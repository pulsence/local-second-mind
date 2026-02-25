"""
Communication providers (email, calendar).
"""

from .gmail import GmailProvider
from .microsoft_graph_mail import MicrosoftGraphMailProvider
from .imap import IMAPProvider
from .google_calendar import GoogleCalendarProvider
from .microsoft_graph_calendar import MicrosoftGraphCalendarProvider
from .caldav import CalDAVProvider
from .models import EmailDraft, EmailMessage, CalendarEvent

__all__ = [
    "GmailProvider",
    "MicrosoftGraphMailProvider",
    "IMAPProvider",
    "EmailMessage",
    "EmailDraft",
    "CalendarEvent",
    "GoogleCalendarProvider",
    "MicrosoftGraphCalendarProvider",
    "CalDAVProvider",
]
