"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useFetchProgress } from "@/hooks/useFetchProgress";
import { useLiveLog } from "@/hooks/useLiveLog";
import { useJobFocus } from "@/hooks/useJobFocus";
import { useJobOperations } from "@/hooks/useJobOperations";
import { useFetchJob } from "@/lib/queries/use-fetch";
import {
  CheckCircleIcon,
  DatabaseIcon,
  PlayIcon,
  RefreshCwIcon,
  ZapIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  MetricCard,
  ExecutionLogsTable,
  LiveStreamPanel,
  NextScheduledCard,
  FetchDialog,
  FetchLiveMonitor,
} from "@/components/scraping";
import {
  FetchDialogType,
  type FetchDialogTypeValue,
  TERMINAL_JOB_STATUSES,
} from "@/lib/constants/fetch";

export default function FetchDataPage() {
  const [pageError, setPageError] = useState<string | null>(null);
  const [dialogType, setDialogType] = useState<FetchDialogTypeValue | null>(null);

  // Live logging
  const { logs, addLog } = useLiveLog();

  // Job operations (fetches, mutations, refresh)
  const {
    fetchUpcoming,
    fetchResults,
    fetchByDate,
    importFootballData,
    refreshAll,
    isPending,
    isRefreshing,
    jobsData,
    stats,
  } = useJobOperations({
    addLog,
    onJobCreated: (jobId) => setFocusedJobId(jobId),
    onError: (error) => setPageError(error),
  });

  // Sort jobs by created_at descending
  const jobs = useMemo(
    () =>
      [...(jobsData?.items ?? [])].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      ),
    [jobsData?.items]
  );

  // Job focus management
  const { focusedJobId, focusedJob, activeJobs, liveSocketJobId, setFocusedJobId } =
    useJobFocus({ jobs });

  const { data: liveJobDetails } = useFetchJob(liveSocketJobId ?? null);

  // WebSocket progress tracking
  const { isConnected, currentProgress } = useFetchProgress({
    jobId: liveSocketJobId,
    enabled: Boolean(liveSocketJobId),
    onConnected: () => addLog("SYS", "WebSocket connection established"),
    onDisconnected: () =>
      addLog("SYS", "WebSocket disconnected, polling fallback active"),
    onError: () => addLog("ERR", "WebSocket connection error"),
    onProgress: (progress) => {
      if (progress.message) {
        addLog("INF", progress.message);
      }
      if (progress.status === "running" && progress.matches_fetched) {
        addLog("SUC", `${progress.matches_fetched} matches fetched`);
      }
      if (
        progress.status &&
        TERMINAL_JOB_STATUSES.includes(
          progress.status as (typeof TERMINAL_JOB_STATUSES)[number]
        )
      ) {
        addLog(
          progress.status === "completed" ? "SUC" : "ERR",
          `Job ${progress.status}`
        );
        void refreshAll();
      }
    },
  });

  const monitoredJob =
    liveJobDetails && focusedJob && liveJobDetails.id === focusedJob.id
      ? liveJobDetails
      : focusedJob;

  const focusedProgress =
    monitoredJob && currentProgress?.job_id === monitoredJob.id
      ? currentProgress
      : null;

  // Calculate metrics
  const totalJobs = jobs.length;
  const matchesFetched = stats?.total_matches_fetched ?? 0;
  const successRate = stats?.success_rate ?? 0;

  // Dialog handlers
  const handleDialogConfirm = useCallback(
    (
      data:
        | {
            type: "upcoming";
            scope: "all" | "league";
            tournament_id?: number;
            tournament_url?: string;
            team_id?: number;
          }
        | {
            type: "results";
            tournament_id?: number;
            tournament_url?: string;
            period?: string;
            period_start?: string;
            period_end?: string;
            max_pages?: number;
            team_id?: number;
          }
        | {
            type: "by-date";
            scope: "all" | "league";
            date?: string;
            tournament_id?: number;
            team_id?: number;
          }
        | {
            type: "import";
            division: string;
            season: string;
          }
    ) => {
      if (data.type === "upcoming") {
        void fetchUpcoming({
          tournament_id: data.tournament_id,
          tournament_url: data.tournament_url,
          scope: data.scope,
        });
      } else if (data.type === "results") {
        void fetchResults({
          tournament_id: data.tournament_id,
          tournament_url: data.tournament_url,
          period: data.period,
          period_start: data.period_start,
          period_end: data.period_end,
          max_pages: data.max_pages,
        });
      } else if (data.type === "by-date") {
        void fetchByDate({
          date: data.date,
          tournament_id: data.tournament_id,
          scope: data.scope,
        });
      } else if (data.type === "import") {
        void importFootballData({
          division: data.division,
          season: data.season,
        });
      }
    },
    [fetchUpcoming, fetchResults, fetchByDate, importFootballData]
  );

  // Auto-dismiss error after 6 seconds
  useEffect(() => {
    if (!pageError) return;
    const timer = window.setTimeout(() => setPageError(null), 6000);
    return () => window.clearTimeout(timer);
  }, [pageError]);

  // Initial log
  useEffect(() => {
    addLog("SYS", "Job Monitor initialized");
  }, [addLog]);

  return (
    <div className="min-h-screen bg-[#0a0c12] p-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
        {/* Main Content Area */}
        <div className="space-y-6">
          {/* Hero Section with Action Buttons */}
          <section className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h1 className="text-2xl font-bold text-[#e0e6f0]">
                JOB MONITOR <span className="text-[#4ade80]">&amp;</span> HISTORY
              </h1>
              <p className="mt-1 text-sm text-[#9ca3af]">
                Real-time tracking of OddsPortal scraping jobs across global football
                leagues.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => refreshAll()}
                disabled={isRefreshing}
                className="gap-2 bg-[#38bdf8] font-semibold text-[#0a0c12] hover:bg-[#0ea5e9]"
              >
                <RefreshCwIcon
                  className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                />
                FORCE_REFRESH
              </Button>
              <div className="flex gap-2">
                <Button
                  onClick={() => setDialogType(FetchDialogType.UPCOMING)}
                  disabled={isPending}
                  className="gap-2 bg-[#4ade80] font-semibold text-[#0a0c12] hover:bg-[#22c55e]"
                >
                  <PlayIcon className="h-4 w-4" />
                  UPCOMING
                </Button>
                <Button
                  onClick={() => setDialogType(FetchDialogType.RESULTS)}
                  disabled={isPending}
                  className="gap-2 bg-[#f59e0b] font-semibold text-[#0a0c12] hover:bg-[#d97706]"
                >
                  <PlayIcon className="h-4 w-4" />
                  RESULTS
                </Button>
                <Button
                  onClick={() => setDialogType(FetchDialogType.BY_DATE)}
                  disabled={isPending}
                  className="gap-2 bg-[#8b5cf6] font-semibold text-[#0a0c12] hover:bg-[#7c3aed]"
                >
                  <PlayIcon className="h-4 w-4" />
                  BY DATE
                </Button>
                <Button
                  onClick={() => setDialogType(FetchDialogType.IMPORT)}
                  disabled={isPending}
                  className="gap-2 bg-[#06b6d4] font-semibold text-[#0a0c12] hover:bg-[#0891b2]"
                >
                  <PlayIcon className="h-4 w-4" />
                  IMPORT
                </Button>
              </div>
            </div>
          </section>

          {/* Fetch Dialog */}
          {dialogType && (
            <FetchDialog
              type={dialogType}
              onConfirm={handleDialogConfirm}
              onClose={() => setDialogType(null)}
              isLoading={isPending}
            />
          )}

          {/* Error Alert */}
          {pageError && (
            <Card className="border-[#f87171]/30 bg-[#f87171]/10">
              <CardContent className="flex items-start gap-3 p-4 text-sm text-[#f87171]">
                <span className="font-mono text-xs">[ERR]</span>
                <p>{pageError}</p>
              </CardContent>
            </Card>
          )}

          {/* Metric Cards */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <MetricCard
              label="TOTAL JOBS"
              value={totalJobs.toLocaleString()}
              icon={<PlayIcon className="h-5 w-5 text-[#444c5e]" />}
              valueColor="#e0e6f0"
            />
            <MetricCard
              label="MATCHES FETCHED"
              value={matchesFetched.toLocaleString()}
              icon={<DatabaseIcon className="h-5 w-5 text-[#444c5e]" />}
              valueColor="#4ade80"
            />
            <MetricCard
              label="SUCCESS RATE"
              value={`${successRate.toFixed(1)}%`}
              icon={<CheckCircleIcon className="h-5 w-5 text-[#444c5e]" />}
              valueColor="#38bdf8"
            />
            <MetricCard
              label="LIVE THROUGHPUT"
              value={activeJobs.length > 0 ? `${activeJobs.length}/s` : "0/s"}
              icon={<ZapIcon className="h-5 w-5 text-[#444c5e]" />}
              valueColor="#e0e6f0"
            />
          </div>

          <FetchLiveMonitor
            job={monitoredJob}
            progress={focusedProgress}
            isConnected={isConnected}
            onRefresh={() => {
              void refreshAll();
            }}
            isRefreshing={isRefreshing}
          />

          {/* Execution Logs Table */}
          <ExecutionLogsTable
            jobs={jobs}
            isLoading={isRefreshing}
            selectedJobId={focusedJobId}
            onSelectJob={setFocusedJobId}
          />
        </div>

        {/* Right Panel */}
        <div className="space-y-4">
          {/* Live Stream */}
          <LiveStreamPanel
            logs={logs}
            hasActiveJobs={activeJobs.length > 0}
            isConnected={isConnected}
            maxHeight="calc(100vh - 300px)"
          />

          {/* Next Scheduled */}
          <NextScheduledCard
            onStart={() => setDialogType(FetchDialogType.UPCOMING)}
            isLoading={isPending}
          />
        </div>
      </div>
    </div>
  );
}
