import { OpenAPI } from '../clients/explorer';

const FALLBACK_BASE_URL = 'http://127.0.0.1:8067';

export const normaliseBaseUrl = (base: string | undefined): string => {
  if (!base || base.trim().length === 0) {
    return FALLBACK_BASE_URL;
  }
  return base.replace(/\/+$/u, '');
};

export const configureApiBase = (): void => {
  const envBase = import.meta?.env?.VITE_EXPLORER_API_BASE as string | undefined;
  OpenAPI.BASE = normaliseBaseUrl(envBase);
};
