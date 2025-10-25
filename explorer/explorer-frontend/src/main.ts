import './style.css';
import type { RunWithSubtasksResponse } from './clients/explorer';
import { DefaultService, OpenAPI } from './clients/explorer';

type LoadingState =
  | {
      status: 'idle' | 'loading';
      runs: [];
    }
  | {
      status: 'error';
      runs: [];
      message: string;
    }
  | {
      status: 'ready';
      runs: RunWithSubtasksResponse[];
    };

const configureApiBase = (): void => {
  const envBase = import.meta?.env?.VITE_EXPLORER_API_BASE as string | undefined;
  OpenAPI.BASE = envBase && envBase.trim().length > 0 ? envBase : 'http://127.0.0.1:8067';
};

const LOG_TAIL_LINE_COUNT = 150;

type LogElements = {
  wrapper: HTMLElement;
  status: HTMLElement;
  content: HTMLElement;
};

type StreamContext = {
  socket: WebSocket;
  button: HTMLButtonElement;
  elements: LogElements;
  closedByClient: boolean;
  runId: number;
  container: HTMLElement;
};

const activeLogStreams = new Map<number, StreamContext>();

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

const findLogElements = (root: ParentNode, subtaskId: number): LogElements | null => {
  const wrapper = root.querySelector<HTMLElement>(`[data-log-wrapper="${subtaskId}"]`);
  const status = root.querySelector<HTMLElement>(`[data-log-status="${subtaskId}"]`);
  const content = root.querySelector<HTMLElement>(`[data-log-content="${subtaskId}"]`);
  if (!wrapper || !status || !content) {
    return null;
  }
  return { wrapper, status, content };
};

const appendLogLine = (element: HTMLElement, line: string): void => {
  const existing = element.textContent ?? '';
  element.textContent = existing.length > 0 ? `${existing}\n${line}` : line;
  element.scrollTop = element.scrollHeight;
};

const isRunStreaming = (runId: number): boolean =>
  Array.from(activeLogStreams.values()).some((context) => context.runId === runId);

const updateRunStreamButton = (runId: number, container: HTMLElement): void => {
  const button = container.querySelector<HTMLButtonElement>(`[data-action="stream-all"]`);
  if (!button) {
    return;
  }
  const anyActive = isRunStreaming(runId);
  button.textContent = anyActive ? 'Stop All Log Streams' : 'Start All Log Streams';
  button.dataset.streaming = anyActive ? 'true' : 'false';
  button.disabled = false;
};

const stopLogStream = (subtaskId: number): void => {
  const context = activeLogStreams.get(subtaskId);
  if (!context) {
    return;
  }
  context.closedByClient = true;
  context.button.disabled = true;
  context.elements.status.textContent = 'Stopping stream…';
  try {
    context.socket.close(1000, 'Client requested stop');
  } catch {
    context.button.disabled = false;
    context.button.dataset.streaming = 'false';
    context.button.textContent = 'Start Log Stream';
    context.elements.status.textContent = 'Stream stopped.';
  }
};

const startLogStream = (
  subtaskId: number,
  runId: number,
  button: HTMLButtonElement,
  root: HTMLElement,
): void => {
  const elements = findLogElements(root, subtaskId);
  if (!elements) {
    return;
  }

  elements.wrapper.classList.remove('hidden');
  elements.status.textContent = 'Connecting…';
  elements.content.textContent = '';
  button.disabled = true;
  button.dataset.streaming = 'pending';
  button.textContent = 'Connecting…';

  const socket = new WebSocket(buildLogStreamUrl(subtaskId));
  const context: StreamContext = {
    socket,
    button,
    elements,
    closedByClient: false,
    runId,
    container: root,
  };

  activeLogStreams.set(subtaskId, context);
  updateRunStreamButton(runId, root);

  socket.addEventListener('open', () => {
    context.button.disabled = false;
    context.button.dataset.streaming = 'true';
    context.button.textContent = 'Stop Log Stream';
    context.elements.status.textContent = `Streaming last ${LOG_TAIL_LINE_COUNT} lines…`;
    updateRunStreamButton(context.runId, context.container);
  });

  socket.addEventListener('message', (event: MessageEvent) => {
    const data = typeof event.data === 'string' ? event.data : '';
    if (data.length > 0) {
      appendLogLine(context.elements.content, data);
    }
  });

  socket.addEventListener('error', () => {
    context.elements.status.textContent = 'Error streaming logs.';
  });

  socket.addEventListener('close', (event: CloseEvent) => {
    activeLogStreams.delete(subtaskId);
    updateRunStreamButton(context.runId, context.container);
    context.button.disabled = false;
    context.button.dataset.streaming = 'false';
    context.button.textContent = 'Start Log Stream';
    context.elements.status.textContent = context.closedByClient
      ? 'Stream stopped.'
      : `Connection closed (code ${event.code}).`;
  });
};

const attachLogStreamHandlers = (root: ParentNode): void => {
  const buttons = root.querySelectorAll<HTMLButtonElement>('[data-action="stream-logs"]');
  buttons.forEach((button) => {
    if (button.dataset.logHandlerAttached === 'true') {
      return;
    }
    button.dataset.logHandlerAttached = 'true';
    button.addEventListener('click', () => {
      const idAttr = button.dataset.subtaskId;
      const runIdAttr = button.dataset.runId;
      if (!idAttr) {
        return;
      }
      const subtaskId = Number.parseInt(idAttr, 10);
      const runId = runIdAttr ? Number.parseInt(runIdAttr, 10) : Number.NaN;
      if (Number.isNaN(subtaskId)) {
        return;
      }
      const runElement = button.closest<HTMLElement>('[data-run-container="true"]');
      if (!runElement || Number.isNaN(runId)) {
        return;
      }
      if (activeLogStreams.has(subtaskId)) {
        stopLogStream(subtaskId);
      } else {
        startLogStream(subtaskId, runId, button, runElement);
      }
    });
  });

  const runButtons = root.querySelectorAll<HTMLButtonElement>('[data-action="stream-all"]');
  runButtons.forEach((button) => {
    if (button.dataset.logHandlerAttached === 'true') {
      return;
    }
    button.dataset.logHandlerAttached = 'true';
    button.addEventListener('click', () => {
      const runIdAttr = button.dataset.runId;
      if (!runIdAttr) {
        return;
      }
      const runId = Number.parseInt(runIdAttr, 10);
      if (Number.isNaN(runId)) {
        return;
      }
      const runElement = button.closest<HTMLElement>('[data-run-container="true"]');
      if (!runElement) {
        return;
      }
      if (isRunStreaming(runId)) {
        button.disabled = true;
        button.textContent = 'Stopping all streams…';
        const activeIds = Array.from(activeLogStreams.entries())
          .filter(([, context]) => context.runId === runId)
          .map(([id]) => id);
        activeIds.forEach((id) => {
          stopLogStream(id);
        });
      } else {
        button.disabled = true;
        button.textContent = 'Starting log streams…';
        const subtaskButtons = runElement.querySelectorAll<HTMLButtonElement>(
          `[data-action="stream-logs"][data-run-id="${runId}"]`,
        );
        subtaskButtons.forEach((subtaskButton) => {
          const subtaskIdAttr = subtaskButton.dataset.subtaskId;
          if (!subtaskIdAttr) {
            return;
          }
          const subtaskId = Number.parseInt(subtaskIdAttr, 10);
          if (Number.isNaN(subtaskId) || activeLogStreams.has(subtaskId)) {
            return;
          }
          startLogStream(subtaskId, runId, subtaskButton, runElement);
        });
        updateRunStreamButton(runId, runElement);
      }
    });
  });
};

const stopAllLogStreams = (): void => {
  const ids = Array.from(activeLogStreams.keys());
  ids.forEach((subtaskId) => {
    stopLogStream(subtaskId);
  });
};

const escapeHtml = (value: string): string =>
  value.replace(/[&<>"']/g, (character: string): string => {
    switch (character) {
      case '&':
        return '&amp;';
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '"':
        return '&quot;';
      case "'":
        return '&#39;';
      default:
        return character;
    }
  });

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

const renderSubtask = (subtask: RunWithSubtasksResponse['subtasks'][number], runId: number): string => {
  const configuration = JSON.stringify(subtask.configuration ?? {}, null, 2);
  return `
    <li class="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-sm shadow-slate-950/20 transition hover:border-sky-500/60 hover:shadow-sky-500/20">
      <div class="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3">
        <div>
          <p class="text-xs font-semibold uppercase tracking-widest text-sky-400">${escapeHtml(
            subtask.base_task_name,
          )}</p>
          <h3 class="text-lg font-semibold text-slate-100">${escapeHtml(
            subtask.resolved_task_name,
          )}</h3>
        </div>
        <div class="text-right text-xs text-slate-400">
          <p class="font-mono text-slate-300">Subtask #${escapeHtml(
            String(subtask.id),
          )}</p>
          <p>${escapeHtml(formatDateTime(subtask.created_at))}</p>
        </div>
      </div>
      <div class="mt-3 text-sm text-slate-300">
        <span class="font-medium text-slate-200">Command</span>
        <p class="mt-1 overflow-x-auto rounded bg-slate-950/60 p-3 font-mono text-xs text-slate-200">${escapeHtml(
          subtask.command,
        )}</p>
      </div>
      <div class="mt-5 rounded-xl border border-sky-600/40 bg-sky-950/40 p-4 text-xs text-sky-100 shadow-inner shadow-sky-900/40">
        <p class="text-[11px] font-semibold uppercase tracking-[0.3em] text-sky-300">Logs Command</p>
        <code class="mt-2 block break-words rounded bg-slate-950/60 px-3 py-2 font-mono text-[11px] text-sky-100">${escapeHtml(
          subtask.logs_command,
        )}</code>
      </div>
      <div class="mt-4 flex flex-wrap gap-3 text-xs text-slate-400">
        <span class="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
          <span class="text-slate-500">Task ID</span>
          <span class="font-mono text-slate-200">${escapeHtml(
            String(subtask.run_task_id),
          )}</span>
        </span>
        <span class="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
          <span class="text-slate-500">IP</span>
          <span class="font-mono text-slate-200">${escapeHtml(
            subtask.ip_address,
          )}</span>
        </span>
        <span class="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
          <span class="text-slate-500">Log Path</span>
          <span class="font-mono text-slate-200" title="${escapeHtml(
            subtask.log_path,
          )}">${escapeHtml(subtask.log_path)}</span>
        </span>
      </div>
      <details class="mt-5 space-y-3 rounded-lg border border-slate-800/80 bg-slate-950/70 p-4">
        <summary class="cursor-pointer text-sm font-medium text-slate-200 transition hover:text-sky-300">
          Configuration
        </summary>
        <pre class="overflow-x-auto rounded-md bg-slate-900/70 p-3 text-xs leading-relaxed text-slate-200">${escapeHtml(
          configuration,
        )}</pre>
      </details>
      <div class="mt-6 space-y-3">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            class="inline-flex items-center gap-2 rounded-lg border border-sky-600/60 bg-sky-600/10 px-4 py-2 text-sm font-medium text-sky-200 transition hover:border-sky-400 hover:bg-sky-500/20"
            data-action="stream-logs"
            data-subtask-id="${escapeHtml(String(subtask.id))}"
            data-run-id="${escapeHtml(String(runId))}"
          >
            Start Log Stream
          </button>
          <span class="text-xs text-slate-400">Streams the latest ${LOG_TAIL_LINE_COUNT} lines in real time.</span>
        </div>
        <div
          class="hidden space-y-3 rounded-xl border border-sky-800/40 bg-slate-950/70 p-4 shadow-inner shadow-sky-900/40"
          data-log-wrapper="${escapeHtml(String(subtask.id))}"
        >
          <div class="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span data-log-status="${escapeHtml(String(subtask.id))}">Stream not started.</span>
            <span class="font-mono text-[11px] text-slate-500">last ${LOG_TAIL_LINE_COUNT} lines</span>
          </div>
          <pre
            class="log-stream-container rounded-lg border border-slate-800/80 bg-slate-950/80 p-3 text-xs text-slate-200"
            data-log-content="${escapeHtml(String(subtask.id))}"
          ></pre>
        </div>
      </div>
    </li>
  `;
};

const renderRun = (run: RunWithSubtasksResponse): string => {
  const subtasksMarkup =
    run.subtasks.length > 0
      ? `<ul class="mt-6 space-y-4">${run.subtasks
          .map((subtask) => renderSubtask(subtask, run.id))
          .join('')}</ul>`
      : `<p class="mt-6 rounded-xl border border-dashed border-slate-800/70 bg-slate-900/40 px-4 py-6 text-sm text-slate-400">No subtasks recorded for this run yet.</p>`;

  return `
    <article
      class="rounded-3xl border border-slate-800/80 bg-slate-900/40 p-8 shadow-lg shadow-slate-950/20 backdrop-blur transition hover:border-sky-500/60 hover:shadow-sky-500/20"
      data-run-id="${escapeHtml(String(run.id))}"
      data-run-container="true"
    >
      <div class="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-4 border-b border-slate-800/60 pb-6">
        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.35em] text-sky-400">Run #${escapeHtml(
            String(run.id),
          )}</p>
          <h2 class="mt-2 text-2xl font-semibold text-slate-50">${escapeHtml(
            run.run_name,
          )}</h2>
          <p class="mt-2 text-sm text-slate-400">Started ${escapeHtml(
            formatDateTime(run.created_at),
          )}</p>
        </div>
        <div class="flex flex-col items-end gap-2 text-right text-xs text-slate-400">
          <button
            type="button"
            class="inline-flex items-center gap-2 self-end rounded-lg border border-sky-600/60 bg-sky-600/10 px-4 py-2 text-sm font-medium text-sky-200 transition hover:border-sky-400 hover:bg-sky-500/20"
            data-action="stream-all"
            data-run-id="${escapeHtml(String(run.id))}"
          >
            Start All Log Streams
          </button>
          <div class="flex max-w-xs flex-wrap justify-end gap-2 rounded-lg bg-slate-950/80 px-3 py-2 text-right">
            <span class="text-slate-500">YAML</span>
            <span class="font-mono text-[11px] text-slate-200 break-all">${escapeHtml(run.yaml_path)}</span>
          </div>
          <span class="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1 text-[11px] text-slate-200">
            Subtasks
            <span class="rounded-full bg-sky-500/20 px-2 py-0.5 font-mono text-sky-300">
              ${escapeHtml(String(run.subtasks.length))}
            </span>
          </span>
        </div>
      </div>
      ${subtasksMarkup}
    </article>
  `;
};

const renderState = (container: HTMLElement, state: LoadingState): void => {
  if (state.status === 'loading' || state.status === 'idle') {
    container.innerHTML = `
      <div class="space-y-6">
        ${Array.from({ length: 3 })
          .map(
            () => `
              <div class="animate-pulse rounded-3xl border border-slate-800/70 bg-slate-900/30 p-8">
                <div class="h-4 w-24 rounded bg-slate-800/80"></div>
                <div class="mt-3 h-8 w-2/3 rounded bg-slate-800/60"></div>
                <div class="mt-5 h-5 w-full rounded bg-slate-800/40"></div>
                <div class="mt-4 h-32 rounded-2xl bg-slate-900/60"></div>
              </div>
            `,
          )
          .join('')}
      </div>
    `;
    return;
  }

  if (state.status === 'error') {
    container.innerHTML = `
      <div class="rounded-3xl border border-red-500/30 bg-red-500/10 p-8 text-red-200 shadow-lg shadow-red-900/30">
        <h2 class="text-xl font-semibold">Something went wrong</h2>
        <p class="mt-3 text-sm text-red-100/80">${escapeHtml(state.message)}</p>
        <button id="retry-fetch" class="mt-6 inline-flex items-center gap-2 rounded-full border border-red-500/50 bg-transparent px-4 py-2 text-sm font-medium text-red-200 transition hover:border-red-400 hover:text-white">
          Try again
        </button>
      </div>
    `;
    return;
  }

  if (state.runs.length === 0) {
    container.innerHTML = `
      <div class="rounded-3xl border border-slate-800/70 bg-slate-900/40 p-12 text-center text-slate-300 shadow-lg shadow-slate-950/20">
        <h2 class="text-2xl font-semibold text-slate-50">No runs yet</h2>
        <p class="mt-4 text-sm text-slate-400">Kick off your first execution to see live status, subtasks, and log links right here.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div class="space-y-8">
      ${state.runs.map((run) => renderRun(run)).join('')}
    </div>
  `;
};

const bootstrap = async (): Promise<void> => {
  configureApiBase();

  const app = document.querySelector<HTMLDivElement>('#app');
  if (!app) {
    throw new Error('Unable to find root application element.');
  }

  app.innerHTML = `
    <div class="min-h-screen bg-slate-950 text-slate-100">
      <div class="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-12">
        <header class="flex flex-col gap-3 border-b border-slate-800/60 pb-8">
          <p class="text-xs font-semibold uppercase tracking-[0.35em] text-sky-400">Run Explorer</p>
          <h1 class="text-4xl font-semibold text-slate-50">Experiment runs at a glance</h1>
          <p class="max-w-3xl text-base text-slate-400">
            Monitor the latest executions, inspect subtasks, and jump straight into logs without leaving the dashboard.
          </p>
        </header>
        <section id="runs-container" class="flex-1 py-8"></section>
        <footer class="border-t border-slate-800/60 pt-6 text-xs text-slate-500">
          Data provided by the Explorer API client.
        </footer>
      </div>
    </div>
  `;

  const runsContainer = document.querySelector<HTMLElement>('#runs-container');
  if (!runsContainer) {
    throw new Error('Unable to find runs container element.');
  }

  const setState = (state: LoadingState): void => {
    stopAllLogStreams();
    renderState(runsContainer, state);

    if (state.status === 'error') {
      const retryButton = document.querySelector<HTMLButtonElement>('#retry-fetch');
      retryButton?.addEventListener('click', () => {
        void loadRuns();
      });
      return;
    }

    if (state.status === 'ready') {
      attachLogStreamHandlers(runsContainer);
    }
  };

  const loadRuns = async (): Promise<void> => {
    setState({ status: 'loading', runs: [] });
    try {
      const runs = await DefaultService.listRunsRunsGet();
      setState({ status: 'ready', runs });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unknown error fetching runs.';
      setState({ status: 'error', runs: [], message });
    }
  };

  await loadRuns();
};

void bootstrap();
