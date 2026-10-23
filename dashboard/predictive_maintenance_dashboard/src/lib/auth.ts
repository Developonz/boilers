/**
 * Authentication utilities for Keycloak SSO integration
 */
import { getAuthUrl } from './api-url';

// ============================================================================
// Cookie Utilities (для хранения токенов)
// ============================================================================

const TOKEN_COOKIE_MAX_AGE = 7 * 24 * 60 * 60; // 7 дней в секундах

function setCookie(name: string, value: string, maxAgeSeconds: number = TOKEN_COOKIE_MAX_AGE): void {
  if (typeof document === 'undefined') return;
  const expires = new Date(Date.now() + maxAgeSeconds * 1000).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; expires=${expires}; SameSite=Lax`;
}

function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp(`(?:^|;\\s*)${name}\\s*=\\s*([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function deleteCookie(name: string): void {
  if (typeof document === 'undefined') return;
  document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
}

// ============================================================================
// Token Management
// ============================================================================

const ACCESS_TOKEN_COOKIE = 'access_token';
const REFRESH_TOKEN_COOKIE = 'refresh_token';

export function getAccessToken(): string | null {
  return getCookie(ACCESS_TOKEN_COOKIE);
}

export function getRefreshToken(): string | null {
  return getCookie(REFRESH_TOKEN_COOKIE);
}

export function setTokens(accessToken: string, refreshToken: string): void {
  setCookie(ACCESS_TOKEN_COOKIE, accessToken);
  setCookie(REFRESH_TOKEN_COOKIE, refreshToken);
}

export function clearTokens(): void {
  deleteCookie(ACCESS_TOKEN_COOKIE);
  deleteCookie(REFRESH_TOKEN_COOKIE);
}

export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

// ============================================================================
// Current User
// ============================================================================

export async function getCurrentUser() {
  const token = getAccessToken();
  if (!token) return null;

  try {
    const response = await fetch(`${getAuthUrl()}/me`, {
      headers: { 'Authorization': `Bearer ${token}` },
    });

    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error('[auth] getCurrentUser error:', error);
    return null;
  }
}

// ============================================================================
// Token Refresh
// ============================================================================

export async function refreshAccessToken(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  try {
    const response = await fetch(`${getAuthUrl()}/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      console.warn('[auth] refresh failed:', response.status);
      return false;
    }

    const data = await response.json();
    if (data.access_token) {
      setTokens(data.access_token, data.refresh_token || refreshToken);
      console.log('[auth] token refreshed successfully');
      return true;
    }

    return false;
  } catch (error) {
    console.error('[auth] refresh error:', error);
    return false;
  }
}

// ============================================================================
// Auth Validation
// ============================================================================

export async function validateAuthWithRefresh(): Promise<boolean> {
  const accessToken = getAccessToken();
  const refreshToken = getRefreshToken();

  if (!accessToken && !refreshToken) {
    console.log('[auth] no tokens');
    return false;
  }

  if (accessToken) {
    // Реально проверяем токен на сервере, чтобы знать валидный он или нет
    try {
      const response = await fetch(`${getAuthUrl()}/me`, {
        headers: { 'Authorization': `Bearer ${accessToken}` },
      });
      const body = await response.json().catch(() => ({}));
      if (response.ok && body.authenticated === true) {
        console.log('[auth] token valid');
        return true;
      }
      console.log(`[auth] token invalid (${response.status} auth=${body.authenticated}), attempting refresh`);
      // Токен невалидный — пробуем обновить
    } catch (e) {
      console.warn('[auth] token check failed:', e);
    }
  }

  if (refreshToken) {
    console.log('[auth] attempting refresh');
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      console.log('[auth] refreshed successfully');
      return true;
    }
  }

  console.log('[auth] all attempts failed');
  return false;
}

// ============================================================================
// Logout
// ============================================================================

export async function logout(): Promise<void> {
  const refreshToken = getRefreshToken();
  clearTokens();

  if (refreshToken) {
    try {
      await fetch(`${getAuthUrl()}/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    } catch (error) {
      console.error('[auth] logout error:', error);
    }
  }
}

export function gracefulLogout(reason: string = 'Session expired'): void {
  console.log(`[auth] gracefulLogout: ${reason}`);
  clearTokens();
  if (typeof window !== 'undefined') {
    window.location.href = '/login';
  }
}

// ============================================================================
// Login
// ============================================================================

export function login(redirectAfterLogin?: string): void {
  let loginUrl = `${getAuthUrl()}/login`;
  
  if (redirectAfterLogin) {
    loginUrl += `?next=${encodeURIComponent(redirectAfterLogin)}`;
  }
  
  console.log('[auth] Redirecting to SSO:', loginUrl);
  if (typeof window !== 'undefined') {
    window.location.href = loginUrl;
  }
}

export function requireAuth(): boolean {
  if (!isAuthenticated()) {
    login(window.location.pathname + window.location.search);
    return false;
  }
  return true;
}