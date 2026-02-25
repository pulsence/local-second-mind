from __future__ import annotations

from lsm.remote.providers.communication.imap import IMAPProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


class FakeIMAP:
    def __init__(self):
        self.append_called = False

    def login(self, username, password):
        return "OK", []

    def select(self, folder):
        return "OK", []

    def search(self, charset, *criteria):
        _ = charset, criteria
        return "OK", [b"1"]

    def fetch(self, message_id, spec):
        _ = message_id, spec
        raw = (
            b"Subject: Test\r\n"
            b"From: sender@example.com\r\n"
            b"To: me@example.com\r\n"
            b"Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n"
            b"\r\nBody"
        )
        return "OK", [(b"1 (FLAGS ())", raw)]

    def append(self, folder, flags, date_time, data):
        _ = folder, flags, date_time, data
        self.append_called = True
        return "OK", []

    def store(self, message_id, flags, value):
        _ = message_id, flags, value
        return "OK", []

    def expunge(self):
        return "OK", []

    def logout(self):
        return "OK", []


class TestIMAPProvider(RemoteProviderOutputTest):
    def test_search_returns_results(self, monkeypatch):
        fake = FakeIMAP()
        monkeypatch.setattr("imaplib.IMAP4_SSL", lambda host, port: fake)

        provider = IMAPProvider(
            {
                "host": "imap.example.com",
                "username": "user",
                "password": "pass",
            }
        )
        results = provider.search("test", max_results=1)
        assert len(results) == 1
        self.assert_valid_output(results)

    def test_create_draft(self, monkeypatch):
        fake = FakeIMAP()
        monkeypatch.setattr("imaplib.IMAP4_SSL", lambda host, port: fake)

        provider = IMAPProvider(
            {
                "host": "imap.example.com",
                "username": "user",
                "password": "pass",
            }
        )
        draft = provider.create_draft(
            recipients=["me@example.com"],
            subject="Subject",
            body="Body",
        )
        assert draft.draft_id
        assert fake.append_called is True
