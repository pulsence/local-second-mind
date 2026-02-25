"""
Communication providers (email, calendar).
"""

from .gmail import GmailProvider
from .microsoft_graph_mail import MicrosoftGraphMailProvider
from .imap import IMAPProvider
from .models import EmailDraft, EmailMessage

__all__ = [
    "GmailProvider",
    "MicrosoftGraphMailProvider",
    "IMAPProvider",
    "EmailMessage",
    "EmailDraft",
]
