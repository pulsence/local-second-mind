"""
Microsoft Graph Mail provider (Outlook/Office 365).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from lsm.config.models import OAuthConfig
from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.oauth import OAuth2Client, OAuthTokenStore
from lsm.remote.providers.communication.models import EmailDraft, EmailMessage

logger = get_logger(__name__)


class MicrosoftGraphMailProvider(BaseRemoteProvider):
    """
    Outlook mail provider via Microsoft Graph API.
    """

    AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
    TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    API_BASE = "https://graph.microsoft.com/v1.0"
    DEFAULT_TIMEOUT = 20
    DEFAULT_SCOPES = [
        "offline_access",
        "Mail.Read",
        "Mail.ReadWrite",
        "Mail.Send",
    ]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint") or self.API_BASE
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        oauth = config.get("oauth") or {}
        scopes = oauth.get("scopes") if isinstance(oauth, dict) else None
        if not scopes:
            oauth = dict(oauth) if isinstance(oauth, dict) else {}
            oauth["scopes"] = list(self.DEFAULT_SCOPES)
        self.oauth_config = _coerce_oauth(oauth)
        token_store = OAuthTokenStore(_oauth_root(config))
        self.oauth_client = OAuth2Client(
            provider_name="microsoft_graph_mail",
            oauth_config=self.oauth_config,
            auth_url=self.AUTH_URL,
            token_url=self.TOKEN_URL,
            token_store=token_store,
        )

    @property
    def name(self) -> str:
        return "microsoft_graph_mail"

    def get_name(self) -> str:
        return "Microsoft Graph Mail"

    def get_description(self) -> str:
        return "Outlook mail via Microsoft Graph API."

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
            {"name": "message_id", "type": "string", "description": "Message ID."},
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
        params: Dict[str, Any] = {
            "$top": max_results,
            "$orderby": "receivedDateTime desc",
        }
        if query:
            params["$search"] = f"\"{query}\""
        filter_expr = _build_filter(
            unread_only=unread_only,
            from_address=from_address,
            to_address=to_address,
            after=after,
            before=before,
        )
        if filter_expr:
            params["$filter"] = filter_expr

        if folder:
            url = f"{self.endpoint}/me/mailFolders/{folder}/messages"
        else:
            url = f"{self.endpoint}/me/messages"
        data = self._request(url, params=params)
        items = data.get("value", []) if isinstance(data, dict) else []
        results: List[EmailMessage] = []
        for item in items[:max_results]:
            message_id = str(item.get("id") or "").strip()
            if not message_id:
                continue
            results.append(_parse_message(item))
        return results

    def create_draft(
        self,
        *,
        recipients: List[str],
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> EmailDraft:
        payload = {
            "subject": subject,
            "body": {"contentType": "Text", "content": body},
            "toRecipients": [
                {"emailAddress": {"address": address}} for address in recipients
            ],
        }
        url = f"{self.endpoint}/me/messages"
        data = self._request(url, method="POST", json_data=payload)
        draft_id = str(data.get("id") or "").strip()
        if not draft_id:
            raise ValueError("Microsoft Graph draft creation failed")
        return EmailDraft(
            draft_id=draft_id,
            subject=subject,
            recipients=recipients,
            body=body,
            thread_id=thread_id,
            metadata={"provider": "microsoft_graph_mail"},
        )

    def send_draft(self, draft_id: str) -> None:
        url = f"{self.endpoint}/me/messages/{draft_id}/send"
        self._request(url, method="POST", json_data={})

    def delete_message(self, message_id: str) -> None:
        url = f"{self.endpoint}/me/messages/{message_id}"
        self._request(url, method="DELETE", json_data=None)

    def _headers(self) -> Dict[str, str]:
        token = self.oauth_client.get_access_token(allow_interactive=True)
        return {
            "Authorization": f"Bearer {token}",
            "ConsistencyLevel": "eventual",
        }

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
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


def _parse_message(payload: Dict[str, Any]) -> EmailMessage:
    sender = payload.get("from", {}).get("emailAddress", {}).get("address", "")
    recipients = [
        item.get("emailAddress", {}).get("address", "")
        for item in payload.get("toRecipients") or []
    ]
    return EmailMessage(
        message_id=str(payload.get("id") or ""),
        thread_id=str(payload.get("conversationId") or "") or None,
        subject=str(payload.get("subject") or ""),
        sender=str(sender or ""),
        recipients=[value for value in recipients if value],
        snippet=str(payload.get("bodyPreview") or ""),
        received_at=_parse_datetime(payload.get("receivedDateTime")),
        is_unread=not bool(payload.get("isRead", True)),
        metadata={"provider": "microsoft_graph_mail", "source_id": payload.get("id")},
    )


def _build_filter(
    *,
    unread_only: bool,
    from_address: Optional[str],
    to_address: Optional[str],
    after: Optional[datetime],
    before: Optional[datetime],
) -> Optional[str]:
    filters: List[str] = []
    if unread_only:
        filters.append("isRead eq false")
    if from_address:
        filters.append(f"from/emailAddress/address eq '{from_address}'")
    if to_address:
        filters.append(f"toRecipients/any(r:r/emailAddress/address eq '{to_address}')")
    if after:
        filters.append(f"receivedDateTime ge {after.isoformat()}")
    if before:
        filters.append(f"receivedDateTime le {before.isoformat()}")
    if not filters:
        return None
    return " and ".join(filters)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


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
