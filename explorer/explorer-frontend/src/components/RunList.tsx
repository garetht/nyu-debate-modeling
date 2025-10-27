import React from 'react';

import type { RunWithSubtasksResponse } from '../clients/explorer';
import { useRunLogControls } from '../hooks/useRunLogControls';
import { formatDateTime } from '../utils/date';
import { SubtaskCard } from './SubtaskCard';

interface RunsListProps {
  runs: RunWithSubtasksResponse[];
}

export const RunsList: React.FC<RunsListProps> = ({ runs }) => (
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
  const {
    registerControl,
    handleStatusChange,
    toggleAllStreams,
    toggleLabel,
    toggleDisabled,
  } = useRunLogControls(run.subtasks.length);

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

