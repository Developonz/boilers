# -*- coding: utf-8 -*-
"""
Configuration for Keycloak authentication.
"""
import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class AuthSettings(BaseSettings):
    """Keycloak authentication settings"""
    
    # =========================================================================
    # Keycloak Configuration
    # =========================================================================
    KEYCLOAK_SERVER_URL: str = Field(
        default="https://kc.enplus.group",
        description="Keycloak server URL"
    )
    KEYCLOAK_REALM: str = Field(
        default="ENTAPP",
        description="Keycloak realm name"
    )
    KEYCLOAK_CLIENT_ID: str = Field(
        default="digitaltoir",
        description="Keycloak client ID"
    )
    KEYCLOAK_CLIENT_SECRET: str = Field(
        default="uXCjG2kNdWCLxL0uzJa6WFbx43lbjzaC",
        description="Keycloak client secret"
    )
    
    # =========================================================================
    # API URLs
    # =========================================================================
    API_URL: str = Field(
        default="http://localhost:8000",
        description="Backend API URL"
    )
    APP_URL: str = Field(
        default="http://localhost:8000",
        description="Frontend application URL"
    )
    
    # =========================================================================
    # CORS
    # =========================================================================
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:8000", "http://localhost:5173", "*"],
        description="Allowed CORS origins"
    )
    
    # =========================================================================
    # JWT / Token Settings
    # =========================================================================
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60, 
        description="Access token expiration in minutes"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    @property
    def keycloak_well_known_url(self) -> str:
        """Get Keycloak .well-known URL for OpenID configuration"""
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}/.well-known/openid-configuration"
    
    @property
    def keycloak_authorization_url(self) -> str:
        """Get Keycloak authorization endpoint"""
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}/protocol/openid-connect/auth"
    
    @property
    def keycloak_token_url(self) -> str:
        """Get Keycloak token endpoint URL"""
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}/protocol/openid-connect/token"
    
    @property
    def keycloak_userinfo_url(self) -> str:
        """Get Keycloak userinfo endpoint URL"""
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}/protocol/openid-connect/userinfo"
    
    @property
    def keycloak_logout_url(self) -> str:
        """Get Keycloak logout endpoint URL"""
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}/protocol/openid-connect/logout"
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


# Global settings instance
auth_settings = AuthSettings()
