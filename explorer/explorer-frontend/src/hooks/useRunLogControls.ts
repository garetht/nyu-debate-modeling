import {
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';

import type { LogStreamStatus, StreamControl } from './useLogStream';

export const useRunLogControls = (subtaskCount: number) => {
  const controlsRef = useRef<Map<number, StreamControl>>(new Map());
  const [subtaskStatuses, setSubtaskStatuses] = useState<
    Record<number, LogStreamStatus>
  >({});

  const handleStatusChange = useCallback(
    (subtaskId: number, status: LogStreamStatus) => {
      setSubtaskStatuses((previous) => ({ ...previous, [subtaskId]: status }));
    },
    [],
  );

  const registerControl = useCallback(
    (subtaskId: number, control: StreamControl) => {
      controlsRef.current.set(subtaskId, control);
      setSubtaskStatuses((previous) => ({
        ...previous,
        [subtaskId]: control.getStatus(),
      }));
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

  const toggleLabel =
    hasActiveStream || isStopping ? 'Stop All Log Streams' : 'Start All Log Streams';
  const toggleDisabled = subtaskCount === 0 || controlsRef.current.size === 0;

  return {
    registerControl,
    handleStatusChange,
    toggleAllStreams,
    toggleLabel,
    toggleDisabled,
  };
};

