/**
 * Hook for managing live log entries
 */

import { useCallback, useState } from "react";
import type { LogEntry } from "@/components/scraping/LiveStreamPanel";
import { LogLevel, type LogLevelValue } from "@/lib/constants/fetch";

const MAX_LOG_ENTRIES = 50;

interface UseLiveLogOptions {
  maxEntries?: number;
}

interface UseLiveLogReturn {
  logs: LogEntry[];
  addLog: (level: LogLevelValue, message: string) => void;
  clearLogs: () => void;
}

/**
 * Format current time as HH:MM:SS
 */
function formatTime(): string {
  return new Date().toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * Hook to manage live log entries with automatic time formatting
 */
export function useLiveLog(options: UseLiveLogOptions = {}): UseLiveLogReturn {
  const { maxEntries = MAX_LOG_ENTRIES } = options;
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = useCallback(
    (level: LogLevelValue, message: string) => {
      const time = formatTime();
      setLogs((prev) => {
        const newLogs = [...prev, { time, level, message }];
        return newLogs.slice(-maxEntries);
      });
    },
    [maxEntries]
  );

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  return { logs, addLog, clearLogs };
}

// Re-export LogLevel for convenience
export { LogLevel };
export type { LogLevelValue };
