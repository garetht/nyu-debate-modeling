import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { createRoot } from 'react-dom/client';

import './style.css';
import type { RunWithSubtasksResponse } from './clients/explorer';
import { DefaultService, OpenAPI } from './clients/explorer';

type LoadingState =
  | { status: 'idle' | 'loading'; runs: [] }
  | { status: 'error'; runs: []; message: string }
  | { status: 'ready'; runs: RunWithSubtasksResponse[] };

type LogStreamStatus = 'idle' | 'connecting' | 'active' | 'stopping' | 'stopped' | 'error';

interface LogStreamState {
  status: LogStreamStatus;
  lines: string[];
  errorMessage?: string;
}

interface StreamControl {
  start: () => void;
  stop: () => void;
  getStatus: () => LogStreamStatus;
}

const LOG_TAIL_LINE_COUNT = 150;

const configureApiBase = (): void => {
  const envBase = import.meta?.env?.VITE_EXPLORER_API_BASE as string | undefined;
  OpenAPI.BASE = envBase && envBase.trim().length > 0 ? envBase : 'http://127.0.0.1:8067';
};

const normaliseBaseUrl = (base: string | undefined): string => {
  if (!base || base.trim().length === 0) {
    return 'http://127.0.0.1:8067';
  }
  return base.replace(/\/+$/u, '');
};

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

const buildLogStreamUrl = (subtaskId: number): string => {
  const httpBase = normaliseBaseUrl(OpenAPI.BASE);
  const wsBase = toWebSocketBase(httpBase);
  return `${wsBase}/subtasks/${subtaskId}/logs?last_lines=${LOG_TAIL_LINE_COUNT}`;
};

const formatDateTime = (value: string): string => {
  try {
    return new Intl.DateTimeFormat('en', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
    }).format(new Date(value));
  } catch {
    return value;
  }
};

const App: React.FC = () => {
  const [state, setState] = useState<LoadingState>({ status: 'idle', runs: [] });

  const loadRuns = useCallback(async () => {
    setState({ status: 'loading', runs: [] });
    try {
      const runs = await DefaultService.listRunsRunsGet();
      setState({ status: 'ready', runs });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error fetching runs.';
      setState({ status: 'error', runs: [], message });
    }
  }, []);

  useEffect(() => {
    configureApiBase();
    void loadRuns();
  }, [loadRuns]);

  const renderContent = (): React.ReactNode => {
    if (state.status === 'loading' || state.status === 'idle') {
      return <LoadingSkeleton />;
    }

    if (state.status === 'error') {
      return <ErrorNotice message={state.message} onRetry={loadRuns} />;
    }

    if (state.runs.length === 0) {
      return <EmptyNotice />;
    }

    return <RunsList runs={state.runs} />;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-12">
        <header className="flex flex-col gap-3 border-b border-slate-800/60 pb-8">
          <p className="text-xs font-semibold uppercase tracking-[0.35em] text-sky-400">
            Run Explorer
          </p>
          <h1 className="text-4xl font-semibold text-slate-50">
            Experiment runs at a glance
          </h1>
          <p className="max-w-3xl text-base text-slate-400">
            Monitor the latest executions, inspect subtasks, and jump straight into logs without leaving the dashboard.
          </p>
        </header>
        <section className="flex-1 py-8">{renderContent()}</section>
        <footer className="border-t border-slate-800/60 pt-6 text-xs text-slate-500">
          Data provided by the Explorer API client.
        </footer>
      </div>
    </div>
  );
};

const LoadingSkeleton: React.FC = () => (
  <div className="space-y-6">
    {Array.from({ length: 3 }).map((_, index) => (
      <div
        key={index}
        className="animate-pulse rounded-3xl border border-slate-800/70 bg-slate-900/30 p-8"
      >
        <div className="h-4 w-24 rounded bg-slate-800/80" />
        <div className="mt-3 h-8 w-2/3 rounded bg-slate-800/60" />
        <div className="mt-5 h-5 w-full rounded bg-slate-800/40" />
        <div className="mt-4 h-32 rounded-2xl bg-slate-900/60" />
      </div>
    ))}
  </div>
);

interface ErrorNoticeProps {
  message: string;
  onRetry: () => void;
}

const ErrorNotice: React.FC<ErrorNoticeProps> = ({ message, onRetry }) => (
  <div className="rounded-3xl border border-red-500/30 bg-red-500/10 p-8 text-red-200 shadow-lg shadow-red-900/30">
    <h2 className="text-xl font-semibold">Something went wrong</h2>
    <p className="mt-3 text-sm text-red-100/80">{message}</p>
    <button
      type="button"
      onClick={onRetry}
      className="mt-6 inline-flex items-center gap-2 rounded-full border border-red-500/50 bg-transparent px-4 py-2 text-sm font-medium text-red-200 transition hover:border-red-400 hover:text-white"
    >
      Try again
    </button>
  </div>
);

const EmptyNotice: React.FC = () => (
  <div className="rounded-3xl border border-slate-800/70 bg-slate-900/40 p-12 text-center text-slate-300 shadow-lg shadow-slate-950/20">
    <h2 className="text-2xl font-semibold text-slate-50">No runs yet</h2>
    <p className="mt-4 text-sm text-slate-400">
      Kick off your first execution to see live status, subtasks, and log links right here.
    </p>
  </div>
);

interface RunsListProps {
  runs: RunWithSubtasksResponse[];
}

const RunsList: React.FC<RunsListProps> = ({ runs }) => (
  <div className="divide-y divide-slate-800/60">
    {runs.map((run) => (
      <RunCard key={run.id} run={run} />
    ))}
  </div>
);

interface RunCardProps {
  run: RunWithSubtasksResponse;
}

const RunCard: React.FC<RunCardProps> = ({ run }) => {
  const controlsRef = useRef<Map<number, StreamControl>>(new Map());
  const [subtaskStatuses, setSubtaskStatuses] = useState<Record<number, LogStreamStatus>>({});

  const handleStatusChange = useCallback((subtaskId: number, status: LogStreamStatus) => {
    setSubtaskStatuses((previous) => ({ ...previous, [subtaskId]: status }));
  }, []);

  const registerControl = useCallback(
    (subtaskId: number, control: StreamControl) => {
      controlsRef.current.set(subtaskId, control);
      setSubtaskStatuses((previous) => ({ ...previous, [subtaskId]: control.getStatus() }));
      return () => {
        controlsRef.current.delete(subtaskId);
        setSubtaskStatuses((previous) => {
          const { [subtaskId]: _removed, ...rest } = previous;
          return rest;
        });
      };
    },
    [],
  );

  const hasActiveStream = useMemo(
    () =>
      Object.values(subtaskStatuses).some(
        (status) => status === 'connecting' || status === 'active',
      ),
    [subtaskStatuses],
  );

  const isStopping = useMemo(
    () => Object.values(subtaskStatuses).some((status) => status === 'stopping'),
    [subtaskStatuses],
  );

  const toggleAllStreams = useCallback(() => {
    const controls = Array.from(controlsRef.current.values());
    if (controls.length === 0) {
      return;
    }
    if (hasActiveStream || isStopping) {
      controls.forEach((control) => control.stop());
    } else {
      controls.forEach((control) => control.start());
    }
  }, [hasActiveStream, isStopping]);

  const toggleLabel = hasActiveStream || isStopping ? 'Stop All Log Streams' : 'Start All Log Streams';
  const toggleDisabled = run.subtasks.length === 0 || controlsRef.current.size === 0;

  return (
    <article className="py-8 first:pt-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.35em] text-sky-400">
            Run #{run.id}
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">{run.run_name}</h2>
          <p className="mt-2 text-sm text-slate-400">Started {formatDateTime(run.created_at)}</p>
        </div>
        <div className="flex flex-col items-end gap-2 text-right text-xs text-slate-400">
          <button
            type="button"
            onClick={toggleAllStreams}
            disabled={toggleDisabled}
            className="inline-flex items-center gap-2 self-end rounded-lg border border-sky-600/60 bg-sky-600/10 px-4 py-2 text-sm font-medium text-sky-200 transition hover:border-sky-400 hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {toggleLabel}
          </button>
          <div className="flex max-w-xs flex-wrap justify-end gap-2 rounded-lg bg-slate-950/80 px-3 py-2 text-right">
            <span className="text-slate-500">YAML</span>
            <span className="break-all font-mono text-[11px] text-slate-200">{run.yaml_path}</span>
          </div>
          <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1 text-[11px] text-slate-200">
            Subtasks
            <span className="rounded-full bg-sky-500/20 px-2 py-0.5 font-mono text-sky-300">
              {run.subtasks.length}
            </span>
          </span>
        </div>
      </div>
      {run.subtasks.length > 0 ? (
        <ul className="mt-6 space-y-4 border-l border-slate-800/60 pl-4 md:pl-6">
          {run.subtasks.map((subtask) => (
            <SubtaskCard
              key={subtask.id}
              subtask={subtask}
              registerControl={registerControl}
              onStatusChange={handleStatusChange}
            />
          ))}
        </ul>
      ) : (
        <div className="mt-6 border-l border-slate-800/60 pl-4 md:pl-6">
          <p className="rounded-xl border border-dashed border-slate-800/70 bg-slate-900/40 px-4 py-6 text-sm text-slate-400">
            No subtasks recorded for this run yet.
          </p>
        </div>
      )}
    </article>
  );
};

interface SubtaskCardProps {
  subtask: RunWithSubtasksResponse['subtasks'][number];
  registerControl: (subtaskId: number, control: StreamControl) => () => void;
  onStatusChange: (subtaskId: number, status: LogStreamStatus) => void;
}

const SubtaskCard: React.FC<SubtaskCardProps> = ({
  subtask,
  registerControl,
  onStatusChange,
}) => {
  const [streamState, setStreamState] = useState<LogStreamState>({
    status: 'idle',
    lines: [],
  });
  const socketRef = useRef<WebSocket | null>(null);
  const statusRef = useRef<LogStreamStatus>('idle');

  const setStatus = useCallback(
    (status: LogStreamStatus, options?: { clearLines?: boolean; errorMessage?: string }) => {
      statusRef.current = status;
      setStreamState((previous) => ({
        status,
        lines: options?.clearLines ? [] : previous.lines,
        errorMessage: options?.errorMessage,
      }));
      onStatusChange(subtask.id, status);
    },
    [onStatusChange, subtask.id],
  );

  const startStream = useCallback(() => {
    const current = socketRef.current;
    if (current && (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setStatus('connecting', { clearLines: true });
    const socket = new WebSocket(buildLogStreamUrl(subtask.id));
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
  }, [setStatus, subtask.id]);

  const stopStream = useCallback(() => {
    const current = socketRef.current;
    if (!current) {
      setStatus('stopped');
      return;
    }

    if (current.readyState === WebSocket.CLOSING || current.readyState === WebSocket.CLOSED) {
      return;
    }

    setStatus('stopping');
    current.close(1000, 'Client requested stop');
  }, [setStatus]);

  useEffect(() => {
    const unregister = registerControl(subtask.id, {
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
  }, [registerControl, setStatus, startStream, stopStream, subtask.id]);

  const isActive = streamState.status === 'active' || streamState.status === 'connecting';
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

  const configuration = useMemo(
    () => JSON.stringify(subtask.configuration ?? {}, null, 2),
    [subtask.configuration],
  );

  const showLogPanel =
    streamState.status !== 'idle' || streamState.lines.length > 0 || Boolean(streamState.errorMessage);

  return (
    <li className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-sm shadow-slate-950/20 transition hover:border-sky-500/60 hover:shadow-sky-500/20">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-sky-400">
            {subtask.base_task_name}
          </p>
          <h3 className="text-lg font-semibold text-slate-100">{subtask.resolved_task_name}</h3>
        </div>
        <div className="text-right text-xs text-slate-400">
          <p className="font-mono text-slate-300">Subtask #{subtask.id}</p>
          <p>{formatDateTime(subtask.created_at)}</p>
        </div>
      </div>
      <div className="mt-3 text-sm text-slate-300">
        <span className="font-medium text-slate-200">Command</span>
        <p className="mt-1 overflow-x-auto rounded bg-slate-950/60 p-3 font-mono text-xs text-slate-200">
          {subtask.command}
        </p>
      </div>
      <div className="mt-5 rounded-xl border border-sky-600/40 bg-sky-950/40 p-4 text-xs text-sky-100 shadow-inner shadow-sky-900/40">
        <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-sky-300">
          Logs Command
        </p>
        <code className="mt-2 block break-words rounded bg-slate-950/60 px-3 py-2 font-mono text-[11px] text-sky-100">
          {subtask.logs_command}
        </code>
      </div>
      <div className="mt-4 flex flex-wrap gap-3 text-xs text-slate-400">
        <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
          <span className="text-slate-500">Task ID</span>
          <span className="font-mono text-slate-200">{subtask.run_task_id}</span>
        </span>
        <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
          <span className="text-slate-500">IP</span>
          <span className="font-mono text-slate-200">{subtask.ip_address}</span>
        </span>
        <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
          <span className="text-slate-500">Log Path</span>
          <span className="font-mono text-slate-200" title={subtask.log_path}>
            {subtask.log_path}
          </span>
        </span>
      </div>
      <details className="mt-5 space-y-3 rounded-lg border border-slate-800/80 bg-slate-950/70 p-4">
        <summary className="cursor-pointer text-sm font-medium text-slate-200 transition hover:text-sky-300">
          Configuration
        </summary>
        <pre className="overflow-x-auto rounded-md bg-slate-900/70 p-3 text-xs leading-relaxed text-slate-200">
          {configuration}
        </pre>
      </details>
      <div className="mt-6 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={handleToggleStream}
            disabled={buttonDisabled}
            className="inline-flex items-center gap-2 rounded-lg border border-sky-600/60 bg-sky-600/10 px-4 py-2 text-sm font-medium text-sky-200 transition hover:border-sky-400 hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {buttonLabel}
          </button>
          <span className="text-xs text-slate-400">
            Streams the latest {LOG_TAIL_LINE_COUNT} lines in real time.
          </span>
        </div>
        {showLogPanel && (
          <div className="space-y-3 rounded-xl border border-sky-800/40 bg-slate-950/70 p-4 shadow-inner shadow-sky-900/40">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
              <span>{statusMessage}</span>
              <span className="font-mono text-[11px] text-slate-500">
                last {LOG_TAIL_LINE_COUNT} lines
              </span>
            </div>
            <pre className="log-stream-container rounded-lg border border-slate-800/80 bg-slate-950/80 p-3 text-xs text-slate-200">
              {streamState.lines.length > 0 ? streamState.lines.join('\n') : 'No log output yet.'}
            </pre>
          </div>
        )}
      </div>
    </li>
  );
};

const rootElement = document.querySelector<HTMLDivElement>('#app');

if (!rootElement) {
  throw new Error('Unable to find root application element.');
}

const root = createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
