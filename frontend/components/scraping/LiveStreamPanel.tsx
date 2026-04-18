"use client";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface LogEntry {
  time: string;
  level: "INF" | "SUC" | "SYS" | "ERR";
  message: string;
}

interface LiveStreamPanelProps {
  logs: LogEntry[];
  hasActiveJobs?: boolean;
  isConnected?: boolean;
  maxHeight?: string;
}

export function LiveStreamPanel({
  logs,
  hasActiveJobs = false,
  isConnected = false,
  maxHeight,
}: LiveStreamPanelProps) {
  return (
    <Card
      className="flex flex-col overflow-hidden border-[#252a37] bg-[#12151d]"
      style={{ height: maxHeight }}
    >
      <div className="flex flex-shrink-0 items-center justify-between border-b border-[#252a37] p-4">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "h-2 w-2 rounded-full",
              hasActiveJobs
                ? isConnected
                  ? "bg-[#4ade80]"
                  : "bg-[#f59e0b]"
                : "bg-[#444c5e]"
            )}
          />
          <h3 className="text-sm font-medium uppercase tracking-wider text-[#e0e6f0]">
            LIVE STREAM
          </h3>
        </div>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto p-3 font-mono text-xs">
        {logs.length === 0 ? (
          <div className="py-8 text-center text-[#444c5e]">Waiting for activity...</div>
        ) : (
          logs.map((log, idx) => (
            <div key={idx} className="flex gap-2">
              <span className="text-[#6b7280]">[{log.time}]</span>
              <span
                className={cn(
                  "font-semibold",
                  log.level === "INF"
                    ? "text-[#4ade80]"
                    : log.level === "SUC"
                      ? "text-[#22c55e]"
                      : log.level === "SYS"
                        ? "text-[#38bdf8]"
                        : "text-[#f87171]"
                )}
              >
                {log.level}
              </span>
              <span
                className={cn("text-[#e0e6f0]", log.level === "ERR" && "font-semibold")}
              >
                {log.message}
              </span>
            </div>
          ))
        )}
        <div className="py-2 text-center text-[#444c5e]">
          --- End of current buffer ---
        </div>
      </div>
    </Card>
  );
}
