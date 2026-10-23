/**
 * App Router - Routing for auth pages and main dashboard
 */
import { useEffect, useState } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import LoginPage from './routes/login-page';
import AuthCallback from './routes/auth-callback';
import Dashboard from './App';
import { validateAuthWithRefresh, isAuthenticated } from './lib/auth';

/**
 * ProtectedRoute - shows children ONLY when authed=true.
 * Before authed=true - loading screen (no dashboard flicker).
 * Key fix: Dashboard renders AFTER token validation+refresh completes.
 */
function ProtectedRoute({ children }) {
  const isRenderProp = typeof children === 'function';
  const [checked, setChecked] = useState(false);
  const [authed, setAuthed] = useState(false);
  const location = useLocation();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!isAuthenticated()) {
        if (!cancelled) { setChecked(true); setAuthed(false); }
        return;
      }
      const ok = await validateAuthWithRefresh();
      if (!cancelled) { setChecked(true); setAuthed(ok); }
    })();
    return () => { cancelled = true; };
  }, []);

  if (!checked) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', color: '#666', fontFamily: 'sans-serif',
      }}>
        Checking authorization...
      </div>
    );
  }

  if (!authed) {
    sessionStorage.setItem('auth_redirect_after_login', location.pathname + location.search);
    return <Navigate to="/login" replace />;
  }

  // Dashboard renders ONLY when authed=true.
  // Key fix: Dashboard useEffect fires AFTER token refresh, so API calls have valid token.
  if (isRenderProp) {
    return <>{children(authed)}</>;
  }
  return children;
}

export default function AppRouter() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/auth/callback" element={<AuthCallback />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            {(authed) => authed ? <Dashboard /> : null}
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
