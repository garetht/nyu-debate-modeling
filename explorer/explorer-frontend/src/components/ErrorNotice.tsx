import React from 'react';

interface ErrorNoticeProps {
  message: string;
  onRetry: () => void;
}

export const ErrorNotice: React.FC<ErrorNoticeProps> = ({ message, onRetry }) => (
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

