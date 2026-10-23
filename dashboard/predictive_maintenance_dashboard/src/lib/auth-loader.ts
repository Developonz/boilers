/**
 * Auth loader utilities for route protection
 */
import { isAuthenticated, validateAuthWithRefresh } from './auth';

/**
 * Require authentication - redirects to login if not authenticated.
 * НЕ вызывает logout/clearTokens — только редирект на /login.
 * Это позволяет пользователю оставаться залогиненным в Keycloak SSO
 * и повторно авторизоваться без полного сброса сессии.
 */
export async function requireAuth(): Promise<boolean> {
  if (typeof window === 'undefined') return false;

  const isAuth = await validateAuthWithRefresh();

  if (!isAuth) {
    const currentPath = window.location.pathname + window.location.search;
    if (currentPath && currentPath !== '/login') {
      sessionStorage.setItem('auth_redirect_after_login', currentPath);
    }
    // Просто редирект — НЕ очищаем cookie (там может быть валидный refresh_token)
    window.location.href = '/login';
    return false;
  }

  return true;
}

/**
 * Redirect to home if already authenticated
 */
export function redirectIfAuthenticated(): void {
  if (typeof window === 'undefined') return;

  if (isAuthenticated()) {
    const savedPath = sessionStorage.getItem('auth_redirect_after_login');
    if (savedPath) {
      sessionStorage.removeItem('auth_redirect_after_login');
      window.location.href = savedPath;
    } else {
      window.location.href = '/';
    }
  }
}
