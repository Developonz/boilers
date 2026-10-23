# -*- coding: utf-8 -*-
"""
Authentication exceptions for Keycloak integration.
"""
from typing import Optional


class AuthenticationError(Exception):
    """Base authentication error"""
    def __init__(self, message: str, code: Optional[str] = None):
        self.message = message
        self.code = code or "authentication_error"
        super().__init__(self.message)


class TokenExchangeError(AuthenticationError):
    """Error exchanging authorization code for tokens"""
    def __init__(self, message: str = "Failed to exchange code for token"):
        super().__init__(message, code="token_exchange_error")


class UserInfoError(AuthenticationError):
    """Error getting user info from Keycloak"""
    def __init__(self, message: str = "Failed to get user info"):
        super().__init__(message, code="user_info_error")


class TokenRefreshError(AuthenticationError):
    """Error refreshing access token"""
    def __init__(self, message: str = "Failed to refresh token"):
        super().__init__(message, code="token_refresh_error")


class TokenValidationError(AuthenticationError):
    """Error validating token"""
    def __init__(self, message: str = "Token validation failed"):
        super().__init__(message, code="token_validation_error")
