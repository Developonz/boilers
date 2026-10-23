import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// @vitejs/plugin-react с automatic JSX runtime:
// автоматически добавляет `import React from 'react'` в каждый файл с JSX.
// Это гарантирует что React доступен в скомпилированном коде.
export default defineConfig({
  plugins: [
    react({
      // Automatic runtime: JSX превращается в jsx(React, ...) автоматически
      // без необходимости `import React` в каждом файле.
      // Также гарантирует что React импортируется в скомпилированном коде.
      jsx: 'automatic',
    }),
  ],
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-dom/client', 'react-router-dom'],
  },
});
