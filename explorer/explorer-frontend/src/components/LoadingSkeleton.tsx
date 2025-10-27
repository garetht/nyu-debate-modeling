import React from 'react';

export const LoadingSkeleton: React.FC = () => (
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

