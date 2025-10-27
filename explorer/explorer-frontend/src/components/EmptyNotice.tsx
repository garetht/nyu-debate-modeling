import React from 'react';

export const EmptyNotice: React.FC = () => (
  <div className="rounded-3xl border border-slate-800/70 bg-slate-900/40 p-12 text-center text-slate-300 shadow-lg shadow-slate-950/20">
    <h2 className="text-2xl font-semibold text-slate-50">No runs yet</h2>
    <p className="mt-4 text-sm text-slate-400">
      Kick off your first execution to see live status, subtasks, and log links right here.
    </p>
  </div>
);

