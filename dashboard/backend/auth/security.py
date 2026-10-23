# -*- coding: utf-8 -*-
"""
Security module for authentication.
Provides token validation and current user resolution.
"""
import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .keycloak_client import get_keycloak_client, KeycloakClient
from .exceptions import TokenValidationError

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_id(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    keycloak: KeycloakClient = Depends(get_keycloak_client),
) -> Optional[str]:
    """
    Get current user ID from Bearer token.
    Returns None for guest users.
    Does NOT require authentication (allows guest access).
    """
    if not creds or creds.scheme.lower() != "bearer":
        return None
    
    token = creds.credentials
    
    try:
        user_info = await keycloak.get_user_info(token)
        return user_info.get("sub")
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        return None


async def require_authentication(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    keycloak: KeycloakClient = Depends(get_keycloak_client),
) -> dict:
    """
    Require authentication - returns user info or raises 401.
    Use this for protected endpoints.
    """
    if not creds or creds.scheme.lower() != "bearer":
        logger.warning(f"[AUTH] No Bearer token provided вЂ” client IP: request details")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = creds.credentials
    logger.info(f"[AUTH] Token received: {token[:20]}...")
    
    try:
        user_info = await keycloak.get_user_info(token)
        logger.info(f"[AUTH] Token valid, user sub: {user_info.get('sub')}")
        return user_info
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Type alias for dependency injection
CurrentUserId = Optional[str]
AuthenticatedUser = dict
