"""
IMAP/SMTP provider for self-hosted email.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage as MimeEmail
from email.utils import parsedate_to_datetime
import imaplib
import smtplib
from typing import Any, Dict, List, Optional

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.providers.communication.models import EmailDraft, EmailMessage

logger = get_logger(__name__)


class IMAPProvider(BaseRemoteProvider):
    """
    IMAP/SMTP provider for reading and drafting email.
    """

    DEFAULT_PORT = 993
    DEFAULT_SMTP_PORT = 587

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host") or config.get("imap_host")
        self.port = int(config.get("port") or config.get("imap_port") or self.DEFAULT_PORT)
        self.username = config.get("username") or config.get("user")
        self.password = config.get("password") or config.get("pass")
        self.use_ssl = bool(config.get("use_ssl", True))
        self.folder = config.get("folder") or "INBOX"
        self.drafts_folder = config.get("drafts_folder") or "Drafts"
        self.smtp_host = config.get("smtp_host") or self.host
        self.smtp_port = int(config.get("smtp_port") or self.DEFAULT_SMTP_PORT)
        self.smtp_username = config.get("smtp_username") or self.username
        self.smtp_password = config.get("smtp_password") or self.password
        self.smtp_use_ssl = bool(config.get("smtp_use_ssl", False))

    @property
    def name(self) -> str:
        return "imap"

    def get_name(self) -> str:
        return "IMAP"

    def get_description(self) -> str:
        return "IMAP/SMTP provider for self-hosted email."

    def validate_config(self) -> None:
        if not self.host:
            raise ValueError("IMAP host is required")
        if not self.username:
            raise ValueError("IMAP username is required")
        if not self.password:
            raise ValueError("IMAP password is required")

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "IMAP search query.", "required": False},
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
        mailbox = self._connect()
        try:
            mailbox.select(folder or self.folder)
            criteria = _build_criteria(
                query=query,
                unread_only=unread_only,
                from_address=from_address,
                to_address=to_address,
                after=after,
                before=before,
            )
            status, data = mailbox.search(None, *criteria)
            if status != "OK":
                return []
            message_ids = data[0].split()
            results: List[EmailMessage] = []
            for message_id in message_ids[:max_results]:
                msg = _fetch_message(mailbox, message_id)
                if msg:
                    results.append(msg)
            return results
        finally:
            mailbox.logout()

    def create_draft(
        self,
        *,
        recipients: List[str],
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> EmailDraft:
        msg = MimeEmail()
        msg["From"] = self.username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        mailbox = self._connect()
        try:
            mailbox.append(
                self.drafts_folder,
                "",
                imaplib.Time2Internaldate(datetime.now(timezone.utc)),
                msg.as_bytes(),
            )
        finally:
            mailbox.logout()
        draft_id = f"imap-draft-{int(datetime.utcnow().timestamp())}"
        return EmailDraft(
            draft_id=draft_id,
            subject=subject,
            recipients=recipients,
            body=body,
            thread_id=thread_id,
            metadata={"provider": "imap"},
        )

    def send_message(self, recipients: List[str], subject: str, body: str) -> None:
        msg = MimeEmail()
        msg["From"] = self.username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        if self.smtp_use_ssl:
            server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20)
            server.starttls()
        try:
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
        finally:
            server.quit()

    def delete_message(self, message_id: str) -> None:
        mailbox = self._connect()
        try:
            mailbox.select(self.folder)
            mailbox.store(message_id, "+FLAGS", "\\Deleted")
            mailbox.expunge()
        finally:
            mailbox.logout()

    def _connect(self):
        if self.use_ssl:
            mailbox = imaplib.IMAP4_SSL(self.host, self.port)
        else:
            mailbox = imaplib.IMAP4(self.host, self.port)
        mailbox.login(self.username, self.password)
        return mailbox


def _decode_header_value(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _extract_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True) or b""
                return payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        return ""
    payload = message.get_payload(decode=True) or b""
    return payload.decode(message.get_content_charset() or "utf-8", errors="ignore")


def _fetch_message(mailbox, message_id: bytes) -> Optional[EmailMessage]:
    status, data = mailbox.fetch(message_id, "(RFC822 FLAGS)")
    if status != "OK" or not data:
        return None
    raw = data[0][1]
    if not raw:
        return None
    import email

    message = email.message_from_bytes(raw)
    subject = _decode_header_value(message.get("Subject"))
    sender = _decode_header_value(message.get("From"))
    recipients = _decode_header_value(message.get("To"))
    date = _decode_header_value(message.get("Date"))
    snippet = _extract_body(message)[:200]
    received_at = None
    try:
        received_at = parsedate_to_datetime(date)
    except Exception:
        received_at = None

    flags = data[0][0]
    is_unread = b"\\Seen" not in flags

    return EmailMessage(
        message_id=message_id.decode("utf-8"),
        subject=subject,
        sender=sender,
        recipients=[value.strip() for value in recipients.split(",") if value.strip()],
        snippet=snippet,
        received_at=received_at,
        is_unread=is_unread,
        metadata={"provider": "imap", "source_id": message_id.decode("utf-8")},
    )


def _build_criteria(
    *,
    query: str,
    unread_only: bool,
    from_address: Optional[str],
    to_address: Optional[str],
    after: Optional[datetime],
    before: Optional[datetime],
) -> List[str]:
    criteria: List[str] = []
    if unread_only:
        criteria.append("UNSEEN")
    if from_address:
        criteria.extend(["FROM", f'"{from_address}"'])
    if to_address:
        criteria.extend(["TO", f'"{to_address}"'])
    if query:
        criteria.extend(["TEXT", f'"{query}"'])
    if after:
        criteria.extend(["SINCE", after.strftime("%d-%b-%Y")])
    if before:
        criteria.extend(["BEFORE", before.strftime("%d-%b-%Y")])
    if not criteria:
        criteria = ["ALL"]
    return criteria
