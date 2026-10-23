/**
 * Точка входа - Predictive Maintenance Dashboard
 * Python + React мониторинг котлоагрегата КА-2
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import AppRouter from './AppRouter.jsx';

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <AppRouter />
    </BrowserRouter>
  </React.StrictMode>
);
