"""
Gmail provider via Google API (OAuth2).
"""

from __future__ import annotations

import base64
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from lsm.config.models import OAuthConfig
from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.oauth import OAuth2Client, OAuthTokenStore
from lsm.remote.providers.communication.models import EmailDraft, EmailMessage

logger = get_logger(__name__)


class GmailProvider(BaseRemoteProvider):
    """
    Gmail provider using the Gmail REST API.
    """

    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    API_BASE = "https://gmail.googleapis.com/gmail/v1"
    DEFAULT_TIMEOUT = 20
    DEFAULT_SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint") or self.API_BASE
        self.user_id = config.get("user_id") or "me"
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        oauth = config.get("oauth") or {}
        scopes = oauth.get("scopes") if isinstance(oauth, dict) else None
        if not scopes:
            oauth = dict(oauth) if isinstance(oauth, dict) else {}
            oauth["scopes"] = list(self.DEFAULT_SCOPES)
        self.oauth_config = _coerce_oauth(oauth)
        token_store = OAuthTokenStore(_oauth_root(config))
        self.oauth_client = OAuth2Client(
            provider_name="gmail",
            oauth_config=self.oauth_config,
            auth_url=self.AUTH_URL,
            token_url=self.TOKEN_URL,
            token_store=token_store,
            extra_auth_params={"access_type": "offline", "prompt": "consent"},
        )

    @property
    def name(self) -> str:
        return "gmail"

    def get_name(self) -> str:
        return "Gmail"

    def get_description(self) -> str:
        return "Gmail provider for searching and drafting emails."

    def validate_config(self) -> None:
        self.oauth_config.validate()

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Search query.", "required": False},
            {"name": "from", "type": "string", "description": "Sender filter.", "required": False},
            {"name": "to", "type": "string", "description": "Recipient filter.", "required": False},
            {"name": "unread_only", "type": "boolean", "description": "Unread filter.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "message_id", "type": "string", "description": "Gmail message ID."},
            {"name": "thread_id", "type": "string", "description": "Thread ID."},
            {"name": "from", "type": "string", "description": "Sender."},
            {"name": "to", "type": "array[string]", "description": "Recipients."},
            {"name": "subject", "type": "string", "description": "Email subject."},
            {"name": "received_at", "type": "string", "description": "Received timestamp."},
            {"name": "is_unread", "type": "boolean", "description": "Unread flag."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        messages = self.search_messages(query, max_results=max_results)
        return [message.to_remote_result() for message in messages]

    def search_messages(
        self,
        query: str,
        *,
        max_results: int = 10,
        unread_only: bool = False,
        from_address: Optional[str] = None,
        to_address: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        folder: Optional[str] = None,
    ) -> List[EmailMessage]:
        q = self._build_query(
            query=query,
            unread_only=unread_only,
            from_address=from_address,
            to_address=to_address,
            after=after,
            before=before,
            folder=folder,
        )
        params = {"q": q, "maxResults": max_results}
        url = f"{self.endpoint}/users/{self.user_id}/messages"
        data = self._request("GET", url, params=params)
        messages = data.get("messages", []) if isinstance(data, dict) else []
        results: List[EmailMessage] = []
        for item in messages[:max_results]:
            message_id = str(item.get("id", "")).strip()
            if not message_id:
                continue
            detail = self._get_message(message_id)
            if detail:
                results.append(detail)
        return results

    def create_draft(
        self,
        *,
        recipients: List[str],
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> EmailDraft:
        raw_message = _encode_rfc822(recipients, subject, body)
        payload: Dict[str, Any] = {"message": {"raw": raw_message}}
        if thread_id:
            payload["message"]["threadId"] = thread_id
        url = f"{self.endpoint}/users/{self.user_id}/drafts"
        data = self._request("POST", url, json_data=payload)
        draft_id = str(data.get("id") or data.get("draft", {}).get("id") or "")
        if not draft_id:
            raise ValueError("Gmail draft creation failed")
        return EmailDraft(
            draft_id=draft_id,
            subject=subject,
            recipients=recipients,
            body=body,
            thread_id=thread_id,
            metadata={"provider": "gmail"},
        )

    def send_draft(self, draft_id: str) -> None:
        url = f"{self.endpoint}/users/{self.user_id}/drafts/{draft_id}/send"
        self._request("POST", url, json_data={})

    def delete_message(self, message_id: str) -> None:
        url = f"{self.endpoint}/users/{self.user_id}/messages/{message_id}"
        self._request("DELETE", url, json_data=None)

    def _get_message(self, message_id: str) -> Optional[EmailMessage]:
        url = f"{self.endpoint}/users/{self.user_id}/messages/{message_id}"
        params = {"format": "metadata", "metadataHeaders": ["Subject", "From", "To", "Date"]}
        data = self._request("GET", url, params=params)
        if not isinstance(data, dict):
            return None
        headers = _extract_headers(data.get("payload", {}).get("headers", []))
        subject = headers.get("Subject") or ""
        sender = headers.get("From") or ""
        recipients = _split_recipients(headers.get("To") or "")
        received_at = _parse_date(headers.get("Date"))
        snippet = str(data.get("snippet") or "")
        labels = data.get("labelIds") or []
        is_unread = "UNREAD" in labels
        return EmailMessage(
            message_id=message_id,
            thread_id=str(data.get("threadId") or "") or None,
            subject=subject,
            sender=sender,
            recipients=recipients,
            snippet=snippet,
            received_at=received_at,
            is_unread=is_unread,
            labels=[str(label) for label in labels if str(label).strip()],
            metadata={"provider": "gmail", "source_id": message_id},
        )

    def _build_query(
        self,
        *,
        query: str,
        unread_only: bool,
        from_address: Optional[str],
        to_address: Optional[str],
        after: Optional[datetime],
        before: Optional[datetime],
        folder: Optional[str],
    ) -> str:
        parts = [query] if query else []
        if unread_only:
            parts.append("is:unread")
        if from_address:
            parts.append(f"from:{from_address}")
        if to_address:
            parts.append(f"to:{to_address}")
        if folder:
            parts.append(f"in:{folder}")
        if after:
            parts.append(f"after:{int(after.timestamp())}")
        if before:
            parts.append(f"before:{int(before.timestamp())}")
        return " ".join([part for part in parts if part]).strip()

    def _headers(self) -> Dict[str, str]:
        token = self.oauth_client.get_access_token(allow_interactive=True)
        return {"Authorization": f"Bearer {token}"}

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = requests.request(
            method,
            url,
            params=params,
            json=json_data,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


def _extract_headers(headers: List[Dict[str, str]]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for item in headers:
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if name:
            payload[name] = value
    return payload


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None


def _split_recipients(value: str) -> List[str]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts


def _encode_rfc822(recipients: List[str], subject: str, body: str) -> str:
    to_value = ", ".join(recipients)
    message = f"To: {to_value}\r\nSubject: {subject}\r\n\r\n{body}"
    return base64.urlsafe_b64encode(message.encode("utf-8")).decode("utf-8")


def _coerce_oauth(raw: Dict[str, Any]):
    if isinstance(raw, OAuthConfig):
        return raw
    if not isinstance(raw, dict):
        return OAuthConfig(client_id="", client_secret="", scopes=[])
    return OAuthConfig(
        client_id=str(raw.get("client_id", "")).strip(),
        client_secret=str(raw.get("client_secret", "")).strip(),
        scopes=list(raw.get("scopes") or []),
        redirect_uri=str(raw.get("redirect_uri", OAuthConfig.redirect_uri)).strip(),
        refresh_buffer_seconds=int(
            raw.get("refresh_buffer_seconds", OAuthConfig.refresh_buffer_seconds)
        ),
    )


def _oauth_root(config: Dict[str, Any]) -> Optional[str]:
    value = config.get("global_folder")
    if not value:
        return None
    return str(Path(value) / "oauth_tokens")
