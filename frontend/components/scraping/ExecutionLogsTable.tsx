"use client";

import { Card } from "@/components/ui/card";
import type { FetchJob } from "@/lib/api/fetch";
import { cn } from "@/lib/utils";
import { StatusBadge } from "./StatusBadge";

interface ExecutionLogsTableProps {
  jobs: FetchJob[];
  isLoading?: boolean;
  selectedJobId?: string | null;
  onSelectJob?: (jobId: string) => void;
  maxRows?: number;
}

export function ExecutionLogsTable({
  jobs,
  isLoading = false,
  selectedJobId,
  onSelectJob,
  maxRows = 10,
}: ExecutionLogsTableProps) {
  return (
    <Card className="overflow-hidden border-[#252a37] bg-[#12151d]">
      <div className="flex items-center justify-between border-b border-[#252a37] p-4">
        <h3 className="text-sm font-medium uppercase tracking-wider text-[#9ca3af]">
          RECENT EXECUTION LOGS
        </h3>
        <div className="flex items-center gap-2 text-xs text-[#9ca3af]">
          <span className="text-[#444c5e]">FILTER:</span>
          <span className="text-[#38bdf8]">ALL_ENVIRONMENTS</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-[#161a25] text-left">
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[#9ca3af]">
                ENDPOINT
              </th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[#9ca3af]">
                STATUS
              </th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[#9ca3af]">
                EXECUTION
              </th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[#9ca3af]">
                SIZE
              </th>
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[#9ca3af]">
                TIME
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#252a37]">
            {isLoading ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-[#9ca3af]">
                  Loading...
                </td>
              </tr>
            ) : jobs.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-[#9ca3af]">
                  No jobs found. Start a new fetch job above.
                </td>
              </tr>
            ) : (
              jobs.slice(0, maxRows).map((job) => (
                <tr
                  key={job.id}
                  onClick={() => onSelectJob?.(job.id)}
                  className={cn(
                    "cursor-pointer transition-colors",
                    selectedJobId === job.id ? "bg-[#1e293b]" : "hover:bg-[#161a25]"
                  )}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div
                        className={cn(
                          "h-8 w-1 rounded-full",
                          job.status === "completed"
                            ? "bg-[#4ade80]"
                            : job.status === "running" || job.status === "pending"
                              ? "bg-[#38bdf8]"
                              : "bg-[#f87171]"
                        )}
                      />
                      <span className="max-w-[200px] truncate font-mono text-sm text-[#e0e6f0]">
                        {job.fetch_type === "upcoming"
                          ? "GET /v3/fixtures/upcoming"
                          : job.fetch_type === "results"
                            ? "GET /v3/fixtures/results"
                            : "GET /v3/fixtures/by-date"}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-3 font-mono text-sm text-[#9ca3af]">
                    {job.started_at && job.completed_at
                      ? `${Math.round(new Date(job.completed_at).getTime() - new Date(job.started_at).getTime())}ms`
                      : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-sm text-[#9ca3af]">
                    {job.matches_fetched
                      ? `${(job.matches_fetched * 0.05).toFixed(1)} MB`
                      : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-sm text-[#9ca3af]">
                    {new Date(job.created_at).toLocaleTimeString("en-US", {
                      hour12: false,
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
