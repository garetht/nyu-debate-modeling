import { OpenAPI } from '../clients/explorer';
import { normaliseBaseUrl } from './api';

export const LOG_TAIL_LINE_COUNT = 150;

const toWebSocketBase = (base: string): string => {
  if (base.startsWith('https://')) {
    return `wss://${base.slice('https://'.length)}`;
  }
  if (base.startsWith('http://')) {
    return `ws://${base.slice('http://'.length)}`;
  }
  if (base.startsWith('wss://') || base.startsWith('ws://')) {
    return base;
  }
  return `ws://${base}`;
};

export const buildLogStreamUrl = (subtaskId: number): string => {
  const httpBase = normaliseBaseUrl(OpenAPI.BASE);
  const wsBase = toWebSocketBase(httpBase);
  return `${wsBase}/subtasks/${subtaskId}/logs?last_lines=${LOG_TAIL_LINE_COUNT}`;
};

