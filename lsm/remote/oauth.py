"""
Shared OAuth2 utilities for remote providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from cryptography.fernet import Fernet, InvalidToken

from lsm.config.models import OAuthConfig
from lsm.logging import get_logger
from lsm.paths import get_global_folder

logger = get_logger(__name__)


_DEFAULT_TOKEN_FILENAME = "token.json.enc"
_DEFAULT_KEY_FILENAME = "oauth.key"
_OAUTH_KEY_ENV = "LSM_OAUTH_KEY"


@dataclass
class OAuthToken:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    token_type: str = "Bearer"
    scope: Optional[str] = None
    scopes: list[str] = None
    id_token: Optional[str] = None
    raw: Dict[str, object] = None

    def __post_init__(self) -> None:
        if self.scopes is None:
            self.scopes = []
        if self.raw is None:
            self.raw = {}

    def is_expired(self, *, buffer_seconds: int = 0) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at <= (time.time() + max(0, int(buffer_seconds)))

    @classmethod
    def from_payload(cls, payload: Dict[str, object], *, fallback_refresh: Optional[str] = None) -> "OAuthToken":
        expires_at = None
        if payload.get("expires_at") is not None:
            try:
                expires_at = float(payload["expires_at"])
            except (TypeError, ValueError):
                expires_at = None
        if expires_at is None and payload.get("expires_in") is not None:
            try:
                expires_at = time.time() + float(payload["expires_in"])
            except (TypeError, ValueError):
                expires_at = None

        scope = payload.get("scope")
        scopes: list[str] = []
        if isinstance(scope, str):
            scopes = [value for value in scope.split() if value]
        elif isinstance(scope, list):
            scopes = [str(value).strip() for value in scope if str(value).strip()]

        refresh_token = payload.get("refresh_token") or fallback_refresh
        token_type = payload.get("token_type") or "Bearer"
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise ValueError("OAuth token response missing access_token")

        return cls(
            access_token=access_token,
            refresh_token=str(refresh_token) if refresh_token else None,
            expires_at=expires_at,
            token_type=str(token_type),
            scope=str(scope) if scope else None,
            scopes=scopes,
            id_token=str(payload.get("id_token")) if payload.get("id_token") else None,
            raw=dict(payload),
        )

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = dict(self.raw or {})
        payload.update(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": self.expires_at,
                "token_type": self.token_type,
                "scope": self.scope,
                "scopes": list(self.scopes or []),
                "id_token": self.id_token,
            }
        )
        return payload


class OAuthTokenStore:
    """
    Encrypted token storage under <global_folder>/oauth_tokens/<provider>/.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        key: Optional[bytes] = None,
    ) -> None:
        base = root if root is not None else get_global_folder() / "oauth_tokens"
        self.root = Path(base).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(self._load_key(key))

    def load(self, provider: str) -> Optional[OAuthToken]:
        token_path = self._token_path(provider)
        if not token_path.exists():
            return None
        try:
            encrypted = token_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
        except (OSError, InvalidToken) as exc:
            logger.warning(f"Failed decrypting OAuth token for {provider}: {exc}")
            return None
        try:
            payload = json.loads(decrypted.decode("utf-8"))
        except Exception as exc:
            logger.warning(f"Failed parsing OAuth token for {provider}: {exc}")
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return OAuthToken.from_payload(payload)
        except Exception as exc:
            logger.warning(f"Invalid OAuth token payload for {provider}: {exc}")
            return None

    def save(self, provider: str, token: OAuthToken) -> Path:
        token_path = self._token_path(provider)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(token.to_payload(), ensure_ascii=True, indent=2).encode("utf-8")
        encrypted = self._fernet.encrypt(payload)
        token_path.write_bytes(encrypted)
        _safe_chmod(token_path)
        return token_path

    def delete(self, provider: str) -> None:
        token_path = self._token_path(provider)
        if token_path.exists():
            token_path.unlink()

    def _token_path(self, provider: str) -> Path:
        safe = _sanitize_provider_name(provider)
        return self.root / safe / _DEFAULT_TOKEN_FILENAME

    def _load_key(self, key: Optional[bytes]) -> bytes:
        if key is not None:
            return key
        env_key = os.getenv(_OAUTH_KEY_ENV)
        if env_key:
            return env_key.encode("utf-8")
        key_path = self.root / _DEFAULT_KEY_FILENAME
        if key_path.exists():
            return key_path.read_bytes().strip()
        generated = Fernet.generate_key()
        key_path.write_bytes(generated)
        _safe_chmod(key_path)
        return generated


class OAuthCallbackServer:
    """
    Local HTTP server to capture OAuth2 authorization codes.
    """

    def __init__(
        self,
        redirect_uri: str,
        *,
        expected_state: Optional[str] = None,
    ) -> None:
        parsed = urlparse(redirect_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        path = parsed.path or "/"
        self._path = path
        self._expected_state = expected_state
        self._event = threading.Event()
        self.code: Optional[str] = None
        self.error: Optional[str] = None

        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] != server._path:
                    self.send_response(404)
                    self.end_headers()
                    return
                query = parse_qs(urlparse(self.path).query)
                code = _first(query.get("code"))
                state = _first(query.get("state"))
                error = _first(query.get("error"))
                if server._expected_state and state != server._expected_state:
                    error = "state_mismatch"
                server.code = code
                server.error = error
                server._event.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                message = "Authorization complete. You can close this tab."
                if error:
                    message = "Authorization failed. You can close this tab."
                self.wfile.write(message.encode("utf-8"))

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        self._httpd = HTTPServer((host, port), Handler)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def wait_for_code(self, timeout: int = 300) -> str:
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("OAuth callback timed out.")
        if self.error:
            raise ValueError(f"OAuth callback error: {self.error}")
        if not self.code:
            raise ValueError("OAuth callback missing authorization code.")
        return self.code

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "OAuthCallbackServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


class OAuth2Client:
    """
    OAuth2 authorization code flow with encrypted token storage.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        oauth_config: OAuthConfig,
        auth_url: str,
        token_url: str,
        token_store: OAuthTokenStore,
        scope_separator: str = " ",
        extra_auth_params: Optional[Dict[str, str]] = None,
        extra_token_params: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider_name = provider_name
        self.oauth_config = oauth_config
        self.auth_url = auth_url
        self.token_url = token_url
        self.token_store = token_store
        self.scope_separator = scope_separator
        self.extra_auth_params = dict(extra_auth_params or {})
        self.extra_token_params = dict(extra_token_params or {})

    def get_access_token(
        self,
        *,
        allow_interactive: bool = False,
        prompt_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 300,
    ) -> str:
        token = self.get_valid_token(
            allow_interactive=allow_interactive,
            prompt_callback=prompt_callback,
            timeout=timeout,
        )
        return token.access_token

    def get_valid_token(
        self,
        *,
        allow_interactive: bool = False,
        prompt_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 300,
    ) -> OAuthToken:
        token = self.token_store.load(self.provider_name)
        if token and not token.is_expired(buffer_seconds=self.oauth_config.refresh_buffer_seconds):
            return token

        if token and token.refresh_token:
            refreshed = self.refresh_token(token)
            if refreshed and not refreshed.is_expired(
                buffer_seconds=self.oauth_config.refresh_buffer_seconds
            ):
                return refreshed

        if not allow_interactive:
            raise RuntimeError(
                f"OAuth token for '{self.provider_name}' is missing or expired. "
                "Run interactive authorization to proceed."
            )
        return self.authorize(prompt_callback=prompt_callback, timeout=timeout)

    def authorize(
        self,
        *,
        prompt_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 300,
        extra_auth_params: Optional[Dict[str, str]] = None,
    ) -> OAuthToken:
        state = self._generate_state()
        auth_url = self.build_authorization_url(state, extra_params=extra_auth_params)
        if prompt_callback is not None:
            prompt_callback(auth_url)
        else:
            logger.info(f"Open OAuth URL for {self.provider_name}: {auth_url}")

        with OAuthCallbackServer(
            self.oauth_config.redirect_uri,
            expected_state=state,
        ) as server:
            code = server.wait_for_code(timeout=timeout)

        token = self.exchange_code(code)
        self.token_store.save(self.provider_name, token)
        return token

    def build_authorization_url(
        self,
        state: Optional[str] = None,
        *,
        extra_params: Optional[Dict[str, str]] = None,
    ) -> str:
        params = {
            "response_type": "code",
            "client_id": self.oauth_config.client_id,
            "redirect_uri": self.oauth_config.redirect_uri,
            "scope": self.scope_separator.join(self.oauth_config.scopes),
        }
        if state:
            params["state"] = state
        params.update(self.extra_auth_params)
        if extra_params:
            params.update(extra_params)
        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code(self, code: str) -> OAuthToken:
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.oauth_config.redirect_uri,
            "client_id": self.oauth_config.client_id,
            "client_secret": self.oauth_config.client_secret,
        }
        payload.update(self.extra_token_params)
        response = requests.post(self.token_url, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("OAuth token response is not a JSON object")
        token = OAuthToken.from_payload(data)
        self.token_store.save(self.provider_name, token)
        return token

    def refresh_token(self, token: OAuthToken) -> OAuthToken:
        if not token.refresh_token:
            raise ValueError("OAuth token has no refresh_token")
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": self.oauth_config.client_id,
            "client_secret": self.oauth_config.client_secret,
        }
        payload.update(self.extra_token_params)
        response = requests.post(self.token_url, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("OAuth refresh response is not a JSON object")
        refreshed = OAuthToken.from_payload(data, fallback_refresh=token.refresh_token)
        self.token_store.save(self.provider_name, refreshed)
        return refreshed

    @staticmethod
    def _generate_state() -> str:
        return secrets.token_urlsafe(16)


def _first(values: Optional[list[str]]) -> Optional[str]:
    if not values:
        return None
    return values[0]


def _sanitize_provider_name(provider: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in provider)
    return cleaned.strip("_") or "provider"


def _safe_chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return
