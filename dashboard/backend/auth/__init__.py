# -*- coding: utf-8 -*-
"""
Authentication module for Keycloak integration.
"""
from .config import auth_settings, AuthSettings
from .keycloak_client import KeycloakClient, get_keycloak_client
from .security import (
    get_current_user_id,
    require_authentication,
    CurrentUserId,
    AuthenticatedUser,
)
from .exceptions import (
    AuthenticationError,
    TokenExchangeError,
    UserInfoError,
    TokenRefreshError,
    TokenValidationError,
)

__all__ = [
    "auth_settings",
    "AuthSettings",
    "KeycloakClient",
    "get_keycloak_client",
    "get_current_user_id",
    "require_authentication",
    "CurrentUserId",
    "AuthenticatedUser",
    "AuthenticationError",
    "TokenExchangeError",
    "UserInfoError",
    "TokenRefreshError",
    "TokenValidationError",
]
