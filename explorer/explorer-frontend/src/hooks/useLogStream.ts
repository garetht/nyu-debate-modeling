import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { buildLogStreamUrl, LOG_TAIL_LINE_COUNT } from '../utils/logStream';

export type LogStreamStatus =
  | 'idle'
  | 'connecting'
  | 'active'
  | 'stopping'
  | 'stopped'
  | 'error';

export interface LogStreamState {
  status: LogStreamStatus;
  lines: string[];
  errorMessage?: string;
}

export interface StreamControl {
  start: () => void;
  stop: () => void;
  getStatus: () => LogStreamStatus;
}

interface UseLogStreamParams {
  subtaskId: number;
  registerControl: (subtaskId: number, control: StreamControl) => () => void;
  onStatusChange: (subtaskId: number, status: LogStreamStatus) => void;
}

export const useLogStream = ({
  subtaskId,
  registerControl,
  onStatusChange,
}: UseLogStreamParams) => {
  const [streamState, setStreamState] = useState<LogStreamState>({
    status: 'idle',
    lines: [],
  });
  const socketRef = useRef<WebSocket | null>(null);
  const statusRef = useRef<LogStreamStatus>('idle');

  const setStatus = useCallback(
    (
      status: LogStreamStatus,
      options?: { clearLines?: boolean; errorMessage?: string },
    ) => {
      statusRef.current = status;
      setStreamState((previous) => ({
        status,
        lines: options?.clearLines ? [] : previous.lines,
        errorMessage: options?.errorMessage,
      }));
      onStatusChange(subtaskId, status);
    },
    [onStatusChange, subtaskId],
  );

  const startStream = useCallback(() => {
    const current = socketRef.current;
    if (
      current &&
      (current.readyState === WebSocket.OPEN ||
        current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    setStatus('connecting', { clearLines: true });
    const socket = new WebSocket(buildLogStreamUrl(subtaskId));
    socketRef.current = socket;

    socket.addEventListener('open', () => {
      setStatus('active');
    });

    socket.addEventListener('message', (event: MessageEvent) => {
      if (typeof event.data !== 'string' || event.data.length === 0) {
        return;
      }
      setStreamState((previous) => {
        const nextLines = [...previous.lines, event.data];
        if (nextLines.length > LOG_TAIL_LINE_COUNT) {
          nextLines.splice(0, nextLines.length - LOG_TAIL_LINE_COUNT);
        }
        return { ...previous, lines: nextLines };
      });
    });

    socket.addEventListener('error', () => {
      setStatus('error', {
        errorMessage: 'Unable to stream logs. Check the backend connection.',
      });
    });

    socket.addEventListener('close', () => {
      socketRef.current = null;
      if (statusRef.current !== 'error') {
        setStatus('stopped');
      }
    });
  }, [setStatus, subtaskId]);

  const stopStream = useCallback(() => {
    const current = socketRef.current;
    if (!current) {
      setStatus('stopped');
      return;
    }

    if (
      current.readyState === WebSocket.CLOSING ||
      current.readyState === WebSocket.CLOSED
    ) {
      return;
    }

    setStatus('stopping');
    current.close(1000, 'Client requested stop');
  }, [setStatus]);

  useEffect(() => {
    const unregister = registerControl(subtaskId, {
      start: startStream,
      stop: stopStream,
      getStatus: () => statusRef.current,
    });

    setStatus('idle', { clearLines: true });

    return () => {
      unregister();
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close(1000, 'Component unmounted');
      }
    };
  }, [registerControl, setStatus, startStream, stopStream, subtaskId]);

  const isActive =
    streamState.status === 'active' || streamState.status === 'connecting';
  const buttonDisabled = streamState.status === 'stopping';
  const buttonLabel =
    streamState.status === 'stopping'
      ? 'Stopping…'
      : isActive
        ? 'Stop Log Stream'
        : 'Start Log Stream';

  const handleToggleStream = useCallback(() => {
    if (streamState.status === 'stopping') {
      return;
    }
    if (isActive) {
      stopStream();
    } else {
      startStream();
    }
  }, [isActive, startStream, stopStream, streamState.status]);

  const statusMessage = useMemo(() => {
    switch (streamState.status) {
      case 'connecting':
        return 'Connecting…';
      case 'active':
        return `Streaming last ${LOG_TAIL_LINE_COUNT} lines…`;
      case 'stopping':
        return 'Stopping stream…';
      case 'stopped':
        return 'Stream stopped.';
      case 'error':
        return streamState.errorMessage ?? 'Streaming error.';
      default:
        return 'Stream not started.';
    }
  }, [streamState.errorMessage, streamState.status]);

  const showLogPanel = useMemo(
    () =>
      streamState.status !== 'idle' ||
      streamState.lines.length > 0 ||
      Boolean(streamState.errorMessage),
    [streamState.errorMessage, streamState.lines.length, streamState.status],
  );

  return {
    streamState,
    buttonLabel,
    buttonDisabled,
    handleToggleStream,
    statusMessage,
    showLogPanel,
    logTailLineCount: LOG_TAIL_LINE_COUNT,
  };
};

