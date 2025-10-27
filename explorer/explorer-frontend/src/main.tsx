import React from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './App';
import './style.css';
import { configureApiBase } from './utils/api';

const rootElement = document.querySelector<HTMLDivElement>('#app');

if (!rootElement) {
  throw new Error('Unable to find root application element.');
}

configureApiBase();

const root = createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
