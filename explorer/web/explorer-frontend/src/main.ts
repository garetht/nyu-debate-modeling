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

const renderSubtask = (subtask: RunWithSubtasksResponse['subtasks'][number]): string => {
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
    </li>
  `;
};

const renderRun = (run: RunWithSubtasksResponse): string => {
  const subtasksMarkup =
    run.subtasks.length > 0
      ? `<ul class="mt-6 space-y-4">${run.subtasks
          .map((subtask) => renderSubtask(subtask))
          .join('')}</ul>`
      : `<p class="mt-6 rounded-xl border border-dashed border-slate-800/70 bg-slate-900/40 px-4 py-6 text-sm text-slate-400">No subtasks recorded for this run yet.</p>`;

  return `
    <article class="rounded-3xl border border-slate-800/80 bg-slate-900/40 p-8 shadow-lg shadow-slate-950/20 backdrop-blur transition hover:border-sky-500/60 hover:shadow-sky-500/20">
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
    renderState(runsContainer, state);

    if (state.status === 'error') {
      const retryButton = document.querySelector<HTMLButtonElement>('#retry-fetch');
      retryButton?.addEventListener('click', () => {
        void loadRuns();
      });
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
