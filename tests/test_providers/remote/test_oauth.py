from __future__ import annotations

import socket
import threading
import time
from unittest.mock import Mock, patch

import requests

from lsm.config.models import OAuthConfig
from lsm.remote.oauth import OAuth2Client, OAuthToken, OAuthTokenStore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _mock_post(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = lambda: None
    return response


def test_token_store_roundtrip_encrypts(tmp_path) -> None:
    store = OAuthTokenStore(tmp_path)
    token = OAuthToken(
        access_token="access-token",
        refresh_token="refresh-token",
        expires_at=time.time() + 3600,
        scope="scope.one scope.two",
        scopes=["scope.one", "scope.two"],
    )
    store.save("gmail", token)
    loaded = store.load("gmail")

    assert loaded is not None
    assert loaded.access_token == "access-token"
    token_path = tmp_path / "gmail" / "token.json.enc"
    assert token_path.exists()
    encrypted = token_path.read_bytes()
    assert b"access-token" not in encrypted


def test_oauth_client_refreshes_expired_token(tmp_path) -> None:
    store = OAuthTokenStore(tmp_path)
    expired = OAuthToken(
        access_token="old-token",
        refresh_token="refresh-token",
        expires_at=time.time() - 10,
    )
    store.save("gmail", expired)
    config = OAuthConfig(
        client_id="client",
        client_secret="secret",
        scopes=["scope.one"],
        redirect_uri="http://localhost:9999/callback",
        refresh_buffer_seconds=0,
    )
    client = OAuth2Client(
        provider_name="gmail",
        oauth_config=config,
        auth_url="https://auth.example.com",
        token_url="https://token.example.com",
        token_store=store,
    )

    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_post(
            {"access_token": "new-token", "expires_in": 3600, "token_type": "Bearer"}
        )
        token = client.get_valid_token()
        assert token.access_token == "new-token"


def test_oauth_authorization_flow_stores_token(tmp_path) -> None:
    port = _free_port()
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    store = OAuthTokenStore(tmp_path)
    config = OAuthConfig(
        client_id="client",
        client_secret="secret",
        scopes=["scope.one"],
        redirect_uri=redirect_uri,
    )
    client = OAuth2Client(
        provider_name="gmail",
        oauth_config=config,
        auth_url="https://auth.example.com",
        token_url="https://token.example.com",
        token_store=store,
    )

    def trigger_callback():
        time.sleep(0.2)
        requests.get(f"{redirect_uri}?code=abc123&state=test-state", timeout=3)

    with patch.object(client, "_generate_state", return_value="test-state"):
        with patch("requests.post") as mock_post:
            mock_post.return_value = _mock_post(
                {
                    "access_token": "fresh-token",
                    "refresh_token": "refresh-token",
                    "expires_in": 3600,
                    "scope": "scope.one",
                }
            )
            thread = threading.Thread(target=trigger_callback, daemon=True)
            thread.start()
            token = client.authorize(prompt_callback=lambda url: None, timeout=5)
            assert token.access_token == "fresh-token"
            loaded = store.load("gmail")
            assert loaded is not None
            assert loaded.access_token == "fresh-token"


def test_oauth_scope_management_parses_scopes() -> None:
    token = OAuthToken.from_payload({"access_token": "t", "scope": "a b c"})
    assert token.scopes == ["a", "b", "c"]
