/**
 * API URL configuration
 * Returns the base API URL based on environment or current location
 */
export function getApiUrl(): string {
  // Check for Vite environment variable first
  if (import.meta && import.meta.env && import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  
  // Fall back to current origin for SPA compatibility
  if (typeof window !== 'undefined' && window.location) {
    return window.location.origin;
  }
  
  // Development fallback
  return 'http://localhost:8000';
}

export function getAuthUrl(): string {
  return `${getApiUrl()}/auth`;
}
