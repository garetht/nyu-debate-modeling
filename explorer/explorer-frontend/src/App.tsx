import React, {
  useCallback,
  useEffect,
  useState,
} from 'react';

import type { RunWithSubtasksResponse } from './clients/explorer';
import { DefaultService } from './clients/explorer';
import { EmptyNotice } from './components/EmptyNotice';
import { ErrorNotice } from './components/ErrorNotice';
import { LoadingSkeleton } from './components/LoadingSkeleton';
import { RunsList } from './components/RunList';

type LoadingState =
  | { status: 'idle' | 'loading'; runs: [] }
  | { status: 'error'; runs: []; message: string }
  | { status: 'ready'; runs: RunWithSubtasksResponse[] };

export const App: React.FC = () => {
  const [state, setState] = useState<LoadingState>({ status: 'idle', runs: [] });

  const loadRuns = useCallback(async () => {
    setState({ status: 'loading', runs: [] });
    try {
      const runs = await DefaultService.listRunsRunsGet();
      setState({ status: 'ready', runs });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unknown error fetching runs.';
      setState({ status: 'error', runs: [], message });
    }
  }, []);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const renderContent = () => {
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

