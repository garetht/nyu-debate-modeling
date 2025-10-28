import React, { useMemo } from 'react';

import type {
  RunProcessResponse,
  RunSubtaskModelInfo,
  RunWithSubtasksResponse,
} from '../clients/explorer';
import { useLogStream } from '../hooks/useLogStream';
import type { LogStreamStatus, StreamControl } from '../hooks/useLogStream';
import { formatDateTime } from '../utils/date';

interface SubtaskCardProps {
  subtask: RunWithSubtasksResponse['subtasks'][number];
  processes: RunProcessResponse[];
  processesStatus: 'idle' | 'loading' | 'ready' | 'error';
  processesError?: string;
  registerControl: (subtaskId: number, control: StreamControl) => () => void;
  onStatusChange: (subtaskId: number, status: LogStreamStatus) => void;
}

export const SubtaskCard: React.FC<SubtaskCardProps> = ({
  subtask,
  processes,
  processesStatus,
  processesError,
  registerControl,
  onStatusChange,
}) => {
  const {
    streamState,
    buttonLabel,
    buttonDisabled,
    handleToggleStream,
    statusMessage,
    showLogPanel,
    logTailLineCount,
  } = useLogStream({
    subtaskId: subtask.id,
    registerControl,
    onStatusChange,
  });

  const baseTaskConfiguration = subtask.base_task_configuration;
  const baseTaskBadges = useMemo<Array<{ label: string; value: string }>>(() => {
    if (!baseTaskConfiguration) {
      return [];
    }
    return [
      { label: 'Config Type', value: baseTaskConfiguration.config_type },
      { label: 'Task Type', value: baseTaskConfiguration.task_type_name },
    ];
  }, [baseTaskConfiguration]);
  const baseTaskModels = useMemo<
    Array<{ role: 'Debater' | 'Judge'; info: RunSubtaskModelInfo }>
  >(() => {
    if (!baseTaskConfiguration) {
      return [];
    }
    return [
      { role: 'Debater', info: baseTaskConfiguration.debater },
      { role: 'Judge', info: baseTaskConfiguration.judge },
    ];
  }, [baseTaskConfiguration]);

  const configuration = useMemo(
    () => JSON.stringify(subtask.configuration ?? {}, null, 2),
    [subtask.configuration],
  );

  const processesContent = useMemo(() => {
    if (processesStatus === 'loading' || processesStatus === 'idle') {
      return (
        <p className="text-xs text-slate-400">
          Loading process metadata...
        </p>
      );
    }

    if (processesStatus === 'error') {
      return (
        <p className="text-xs text-rose-400">
          Failed to load processes{processesError ? `: ${processesError}` : '.'}
        </p>
      );
    }

    if (processes.length === 0) {
      return (
        <p className="text-xs text-slate-400">
          No processes recorded for this subtask.
        </p>
      );
    }

    return (
      <ul className="space-y-3">
        {processes.map((process, index) => {
          const statusLabel = process.success ? 'Success' : 'Failed';
          const statusClasses = process.success
            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
            : 'bg-rose-500/20 text-rose-300 border-rose-500/30';
          return (
            <li
              key={`${process.subtask_id}-${process.pid ?? index}-${index}`}
              className="space-y-2 rounded-lg border border-slate-800/70 bg-slate-950/60 p-3 shadow-inner shadow-slate-950/30"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-mono text-slate-300">
                  Process #{index + 1}
                </span>
                <span
                  className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${statusClasses}`}
                >
                  {statusLabel}
                </span>
              </div>
              <dl className="space-y-2 text-[11px] text-slate-300">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <dt className="text-slate-500">IP</dt>
                  <dd className="font-mono text-slate-200">
                    {process.ip_address}
                  </dd>
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <dt className="text-slate-500">PID</dt>
                  <dd className="font-mono text-slate-200">
                    {process.pid ?? '—'}
                  </dd>
                </div>
                <div className="flex flex-col gap-1">
                  <dt className="text-slate-500">Command</dt>
                  <dd className="overflow-x-auto rounded-md bg-slate-900/70 p-2 font-mono text-[10px] text-slate-200">
                    {process.command}
                  </dd>
                </div>
                {process.remote_command ? (
                  <div className="flex flex-col gap-1">
                    <dt className="text-slate-500">Remote Command</dt>
                    <dd className="overflow-x-auto rounded-md bg-slate-900/70 p-2 font-mono text-[10px] text-slate-200">
                      {process.remote_command}
                    </dd>
                  </div>
                ) : null}
                {process.ps_line ? (
                  <div className="flex flex-col gap-1">
                    <dt className="text-slate-500">Process Line</dt>
                    <dd className="overflow-x-auto rounded-md bg-slate-900/70 p-2 font-mono text-[10px] text-slate-200">
                      {process.ps_line}
                    </dd>
                  </div>
                ) : null}
                {process.error ? (
                  <div className="flex flex-col gap-1">
                    <dt className="text-slate-500">Error</dt>
                    <dd className="overflow-x-auto rounded-md bg-rose-900/30 p-2 font-mono text-[10px] text-rose-200">
                      {process.error}
                    </dd>
                  </div>
                ) : null}
              </dl>
            </li>
          );
        })}
      </ul>
    );
  }, [processes, processesError, processesStatus]);

  return (
    <li className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-sm shadow-slate-950/20 transition hover:border-sky-500/60 hover:shadow-sky-500/20">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3">
        <div>
          <div className="pt-2 pb-4 text-xs text-slate-400">
            <p className="font-mono text-slate-300">Subtask #{subtask.id}</p>
            <p>Started {formatDateTime(subtask.created_at)}</p>
          </div>
          {baseTaskConfiguration ? (
            <div className="mt-3 space-y-3 text-[11px] text-slate-300">
              {baseTaskBadges.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {baseTaskBadges.map((detail) => (
                    <span
                      key={detail.label}
                      className="inline-flex items-center gap-2 rounded-full bg-slate-950/70 px-3 py-1"
                    >
                      <span className="text-slate-500">{detail.label}</span>
                      <span className="font-mono text-slate-200">{detail.value}</span>
                    </span>
                  ))}
                  <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
                    <span className="text-slate-500">Task ID</span>
                    <span className="font-mono text-slate-200">{subtask.run_task_id}</span>
                  </span>
                  <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1">
                    <span className="text-slate-500">IP</span>
                    <span className="font-mono text-slate-200">{subtask.ip_address}</span>
                  </span>
                </div>
              ) : null}
              <div className="grid gap-3 md:grid-cols-2">
                {baseTaskModels.map(({ role, info }) => (
                  <div
                    key={role}
                    className="rounded-lg border border-slate-800/70 bg-slate-950/60 p-3 shadow-inner shadow-slate-950/30"
                  >
                    <p className="text-[10px] font-semibold uppercase tracking-[0.3em] text-sky-300">
                      {role}
                    </p>
                    <dl className="mt-2 space-y-1 text-[11px] text-slate-300">
                      <div className="flex items-center justify-between gap-2">
                        <dt className="text-slate-500">Key</dt>
                        <dd className="font-mono text-slate-200">{info.key}</dd>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <dt className="text-slate-500">Training</dt>
                        <dd className="font-mono text-slate-200">{info.training_round}</dd>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <dt className="text-slate-500">Model Type</dt>
                        <dd className="font-mono text-slate-200">{info.model_type}</dd>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <dt className="text-slate-500">Model Path</dt>
                        <dd
                          className="max-w-[180px] truncate font-mono text-slate-200"
                          title={info.model_file_path ?? ''}
                        >
                          {info.model_file_path ?? '—'}
                        </dd>
                      </div>
                    </dl>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <details className="mt-3 space-y-2 rounded-lg border border-slate-800/80 bg-slate-950/70 p-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-200 transition hover:text-sky-300">
          Logs Command
        </summary>
        <pre className="overflow-x-auto rounded-md bg-slate-900/70 p-2 text-xs leading-relaxed text-slate-200">
          {subtask.logs_command}
        </pre>
      </details>
      <details className="mt-3 space-y-2 rounded-lg border border-slate-800/80 bg-slate-950/70 p-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-200 transition hover:text-sky-300">
          Configuration
        </summary>
        <pre className="overflow-x-auto rounded-md bg-slate-900/70 p-2 text-xs leading-relaxed text-slate-200">
          {configuration}
        </pre>
      </details>
      <details className="mt-3 space-y-2 rounded-lg border border-slate-800/80 bg-slate-950/70 p-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-200 transition hover:text-sky-300">
          Processes
        </summary>
        <div className="space-y-2">{processesContent}</div>
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
            Streams the latest {logTailLineCount} lines in real time.
          </span>
        </div>
        {showLogPanel && (
          <div className="space-y-3 rounded-xl border border-sky-800/40 bg-slate-950/70 p-4 shadow-inner shadow-sky-900/40">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
              <span>{statusMessage}</span>
              <span className="font-mono text-[11px] text-slate-500">
                last {logTailLineCount} lines
              </span>
            </div>
            <pre className="log-stream-container rounded-lg border border-slate-800/80 bg-slate-950/80 p-3 text-xs text-slate-200">
              {streamState.lines.length > 0
                ? streamState.lines.join('\n')
                : 'No log output yet.'}
            </pre>
          </div>
        )}
      </div>
    </li>
  );
};
