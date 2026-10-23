/**
 * Login Page - Redirects to Keycloak SSO
 */
import { useEffect, useRef, type CSSProperties } from 'react';
import { login } from '../lib/auth';

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

/** @keyframes нельзя задать inline, поэтому инжектим через <style> */
const keyframes = `
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

export default function LoginPage() {
  const redirected = useRef(false);

  useEffect(() => {
    // Защита от двойного срабатывания эффекта в StrictMode
    if (redirected.current) return;
    redirected.current = true;
    login('/');
  }, []);

  return (
    <div style={styles.container} aria-busy="true">
      <style>{keyframes}</style>
      <div style={styles.content} role="status">
        <h1 style={styles.title}>Предиктивный дашборд</h1>
        <p style={styles.subtitle}>Перенаправление на систему авторизации...</p>
        <div style={styles.spinner} aria-hidden="true" />
      </div>
    </div>
  );
}