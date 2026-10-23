/**
 * Auth Callback Page - Handles OAuth callback from Keycloak
 */
import { useEffect, useState, type CSSProperties } from 'react';
import { useSearchParams } from 'react-router-dom';
import { setTokens } from '../lib/auth';

const styles: Record<string, CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  content: {
    textAlign: 'center',
    padding: '40px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
  },
  title: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#333',
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '14px',
    color: '#666',
    marginBottom: '20px',
  },
  errorTitle: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#d32f2f',
    marginBottom: '8px',
  },
  errorMessage: {
    fontSize: '14px',
    color: '#666',
    marginBottom: '16px',
  },
  redirectText: {
    fontSize: '12px',
    color: '#999',
    marginTop: '16px',
  },
  spinner: {
    width: '24px',
    height: '24px',
    border: '3px solid #e0e0e0',
    borderTopColor: '#1976d2',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    margin: '0 auto',
  },
};

/**
 * Parses params from both the query string (?a=b) and the
 * URL fragment (#a=b). Keycloak's implicit flow returns tokens
 * in the fragment, which useSearchParams does not see.
 */
function getCallbackParam(
  searchParams: URLSearchParams,
  key: string,
): string | null {
  if (searchParams.has(key)) return searchParams.get(key);

  const hash = window.location.hash;
  if (hash.startsWith('#') && hash.length > 1) {
    return new URLSearchParams(hash.slice(1)).get(key);
  }

  return null;
}

/**
 * Guards against open redirect: only allow relative paths
 * like "/dashboard", reject "https://evil.com" or "//evil.com".
 */
function isSafeRedirect(url: string | null | undefined): url is string {
  return (
    typeof url === 'string' &&
    url.startsWith('/') &&
    !url.startsWith('//')
  );
}

export default function AuthCallback() {
  const [searchParams] = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const accessToken = getCallbackParam(searchParams, 'access_token');
    const refreshToken = getCallbackParam(searchParams, 'refresh_token');
    const nextUrl = getCallbackParam(searchParams, 'next');
    const errorParam = getCallbackParam(searchParams, 'error');
    const errorDescription = getCallbackParam(
      searchParams,
      'error_description',
    );

    let redirectTimer: number | undefined;

    // Убираем токены/ошибки из адреса: без этого они остаются
    // в истории браузера (кнопка «Назад», share-ссылки).
    const scrubUrl = () => {
      window.history.replaceState(
        null,
        '',
        window.location.pathname + window.location.search,
      );
    };

    if (errorParam) {
      console.error('Auth error:', errorParam, errorDescription);
      scrubUrl();
      setError(errorDescription || errorParam);
      redirectTimer = window.setTimeout(() => {
        window.location.replace('/login');
      }, 3000);
      return () => window.clearTimeout(redirectTimer);
    }

    if (accessToken && refreshToken) {
      setTokens(accessToken, refreshToken);
      scrubUrl();

      const savedPath = sessionStorage.getItem('auth_redirect_after_login');
      const redirectTo = isSafeRedirect(nextUrl)
        ? nextUrl
        : isSafeRedirect(savedPath)
          ? savedPath
          : '/';
      sessionStorage.removeItem('auth_redirect_after_login');

      console.log('[callback] Tokens saved, redirecting to:', redirectTo);

      redirectTimer = window.setTimeout(() => {
        window.location.replace(redirectTo);
      }, 500);
      return () => window.clearTimeout(redirectTimer);
    }

    console.error('Missing tokens in callback');
    scrubUrl();
    setError('Missing authentication tokens');
    redirectTimer = window.setTimeout(() => {
      window.location.replace('/login');
    }, 3000);
    return () => window.clearTimeout(redirectTimer);
  }, [searchParams]);

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.content} role="alert">
          <h1 style={styles.errorTitle}>Ошибка авторизации</h1>
          <p style={styles.errorMessage}>{error}</p>
          <p style={styles.redirectText}>Перенаправление на страницу входа...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* keyframes нельзя объявить через inline-styles,
          поэтому вставляем их через <style> */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
      <div style={styles.container}>
        <div style={styles.content}>
          <h1 style={styles.title}>Авторизация</h1>
          <p style={styles.subtitle}>Пожалуйста, подождите...</p>
          <div style={styles.spinner}></div>
        </div>
      </div>
    </>
  );
}