# -*- coding: utf-8 -*-
"""Keycloak HTTP client for OAuth2/OIDC operations."""
import logging
from typing import Dict, Any, Optional

import httpx

from .config import auth_settings
from .exceptions import TokenExchangeError, UserInfoError, TokenRefreshError

logger = logging.getLogger(__name__)


class KeycloakClient:
    """HTTP client for Keycloak operations."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        realm: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.server_url = server_url or auth_settings.KEYCLOAK_SERVER_URL
        self.realm = realm or auth_settings.KEYCLOAK_REALM
        self.client_id = client_id or auth_settings.KEYCLOAK_CLIENT_ID
        self.client_secret = client_secret or auth_settings.KEYCLOAK_CLIENT_SECRET
        self.timeout = timeout
        self._token_url = f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/token"
        self._userinfo_url = f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/userinfo"
        self._logout_url = f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/logout"

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        if code_verifier:
            data["code_verifier"] = code_verifier

        try:
            async with httpx.AsyncClient(verify=True, timeout=self.timeout) as client:
                response = await client.post(
                    self._token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if response.status_code != 200:
                    raise TokenExchangeError(f"Token exchange failed: {response.status_code}")
                return response.json()
        except httpx.HTTPError as e:
            raise TokenExchangeError(f"HTTP error during token exchange: {e}")

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        try:
            async with httpx.AsyncClient(verify=True, timeout=self.timeout) as client:
                response = await client.post(
                    self._token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if response.status_code != 200:
                    raise TokenRefreshError(f"Token refresh failed: {response.status_code}")
                return response.json()
        except httpx.HTTPError as e:
            raise TokenRefreshError(f"HTTP error during token refresh: {e}")

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Keycloak."""
        try:
            async with httpx.AsyncClient(verify=True, timeout=self.timeout) as client:
                response = await client.get(
                    self._userinfo_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                logger.info(f"[Keycloak] get_user_info status={response.status_code}")
                if response.status_code != 200:
                    logger.error(
                        f"[Keycloak] get_user_info FAILED: "
                        f"{response.status_code} {response.text[:200]}"
                    )
                    raise UserInfoError(
                        f"User info failed: {response.status_code}"
                    )
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"[Keycloak] HTTP error: {e}")
            raise UserInfoError(f"Error getting user info: {e}")

    async def revoke_token(
        self,
        token: str,
        token_type_hint: str = "refresh_token",
    ) -> bool:
        """Revoke access or refresh token."""
        data = {
            "token": token,
            "token_type_hint": token_type_hint,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        try:
            async with httpx.AsyncClient(verify=True, timeout=self.timeout) as client:
                response = await client.post(
                    self._logout_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                return response.status_code in (200, 204)
        except httpx.HTTPError:
            return False


_keycloak_client: KeycloakClient | None = None


def get_keycloak_client() -> KeycloakClient:
    """Get Keycloak client singleton."""
    global _keycloak_client
    if _keycloak_client is None:
        _keycloak_client = KeycloakClient()
    return _keycloak_client
