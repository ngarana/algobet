"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress, ProgressValue } from "@/components/ui/progress";
import type { FetchJob } from "@/lib/api/fetch";
import {
  ActivityIcon,
  AlertTriangleIcon,
  CheckCircle2Icon,
  ClockIcon,
  RefreshCwIcon,
  WifiIcon,
  WifiOffIcon,
  XCircleIcon,
} from "lucide-react";

interface FetchProgressMessage {
  job_id: string;
  progress?: number;
  matches_fetched?: number;
  matches_saved?: number;
  message?: string;
  current_page?: number;
  total_pages?: number;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string;
  status?: FetchJob["status"];
}

interface FetchLiveMonitorProps {
  job: FetchJob | null;
  progress: FetchProgressMessage | null;
  isConnected: boolean;
  connectionMessage?: string | null;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

const statusConfig = {
  pending: {
    badge: "secondary" as const,
    icon: ClockIcon,
    tone: "text-amber-500",
    label: "Queued",
  },
  running: {
    badge: "default" as const,
    icon: ActivityIcon,
    tone: "text-sky-500",
    label: "Running",
  },
  completed: {
    badge: "success" as const,
    icon: CheckCircle2Icon,
    tone: "text-emerald-500",
    label: "Completed",
  },
  failed: {
    badge: "destructive" as const,
    icon: XCircleIcon,
    tone: "text-red-500",
    label: "Failed",
  },
  cancelled: {
    badge: "outline" as const,
    icon: AlertTriangleIcon,
    tone: "text-muted-foreground",
    label: "Cancelled",
  },
};

function formatTimestamp(value: string | null | undefined) {
  if (!value) {
    return "Not yet available";
  }

  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(
  startedAt: string | null | undefined,
  completedAt?: string | null
) {
  if (!startedAt) {
    return "Not started";
  }

  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const diffSeconds = Math.max(0, Math.floor((end - start) / 1000));
  const minutes = Math.floor(diffSeconds / 60);
  const seconds = diffSeconds % 60;

  if (minutes === 0) {
    return `${seconds}s`;
  }

  return `${minutes}m ${seconds}s`;
}

function formatJobLabel(job: FetchJob) {
  if (job.tournament_url) {
    const segments = job.tournament_url.split("/").filter(Boolean);
    const normalizedSegments =
      segments[segments.length - 1] === "results" ? segments.slice(0, -1) : segments;
    const tail = normalizedSegments.slice(-2).join(" / ").replace(/-/g, " ");
    return tail || job.tournament_url;
  }

  return "All leagues";
}

export function FetchLiveMonitor({
  job,
  progress,
  isConnected,
  connectionMessage,
  onRefresh,
  isRefreshing = false,
}: FetchLiveMonitorProps) {
  if (!job) {
    return (
      <Card className="border-border/70 bg-card/90 shadow-[0_20px_50px_-34px_rgba(15,23,42,0.55)]">
        <CardHeader>
          <CardTitle>Live Job Monitor</CardTitle>
          <CardDescription>
            Start a fetch operation or select a recent job to inspect live progress
            here.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex min-h-64 items-center justify-center rounded-b-2xl border-t border-dashed border-border/60 bg-muted/20">
          <div className="max-w-sm space-y-2 text-center text-muted-foreground">
            <ActivityIcon className="mx-auto h-10 w-10 text-muted-foreground/70" />
            <p className="font-medium text-foreground">No focused job</p>
            <p className="text-sm">
              The monitor will display live status, messages, progress, and details as
              soon as a job is queued.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const effectiveStatus = progress?.status ?? job.status;
  const currentProgress = progress?.progress ?? job.progress ?? 0;
  const matchesFetched = progress?.matches_fetched ?? job.matches_fetched;
  const matchesSaved =
    progress?.matches_saved ??
    (job.status === "completed" ? job.matches_fetched : undefined);
  const startedAt = progress?.started_at ?? job.started_at;
  const completedAt = progress?.completed_at ?? job.completed_at;
  const message = progress?.message ?? job.message ?? "Waiting for updates...";
  const errorMessage = progress?.error ?? job.errors[0] ?? null;
  const progressPages =
    progress?.current_page && progress?.total_pages
      ? `${progress.current_page}/${progress.total_pages} pages`
      : null;
  const config = statusConfig[effectiveStatus];
  const StatusIcon = config.icon;
  const isLive = effectiveStatus === "running" || effectiveStatus === "pending";

  return (
    <Card className="overflow-hidden border-border/70 bg-card/90 shadow-[0_24px_60px_-34px_rgba(15,23,42,0.58)]">
      <CardHeader className="border-b border-border/60 bg-gradient-to-br from-background via-background to-muted/30">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={config.badge} className="gap-1.5">
                <StatusIcon
                  className={`h-3.5 w-3.5 ${isLive ? "animate-pulse" : ""}`}
                />
                {config.label}
              </Badge>
              <Badge variant="outline" className="border-border/60 bg-background/70">
                {job.fetch_type === "upcoming"
                  ? "Upcoming"
                  : job.fetch_type === "by-date"
                    ? "By Date"
                    : "Results"}
              </Badge>
              <Badge variant="outline" className="border-border/60 bg-background/70">
                {formatJobLabel(job)}
              </Badge>
            </div>
            <div>
              <CardTitle className="text-xl">Live Job Monitor</CardTitle>
              <CardDescription className="mt-1 text-sm leading-6">
                {message}
              </CardDescription>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start">
            <Badge
              variant="outline"
              className={`border-border/60 bg-background/70 ${
                isConnected ? "text-emerald-500" : "text-amber-500"
              }`}
            >
              {isConnected ? (
                <WifiIcon className="mr-1.5 h-3.5 w-3.5" />
              ) : (
                <WifiOffIcon className="mr-1.5 h-3.5 w-3.5" />
              )}
              {isConnected ? "Socket connected" : "Polling fallback"}
            </Badge>
            {onRefresh && (
              <Button
                variant="outline"
                size="sm"
                onClick={onRefresh}
                disabled={isRefreshing}
              >
                <RefreshCwIcon
                  className={`mr-2 h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`}
                />
                Refresh
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-5 p-6">
        {connectionMessage && isLive && (
          <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-300">
            {connectionMessage}
          </div>
        )}

        <div className="rounded-2xl border border-border/60 bg-muted/15 p-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-foreground">Run progress</p>
              <p className="text-sm text-muted-foreground">
                Job ID {job.id.slice(0, 8)}...
              </p>
            </div>
            <p className={`text-2xl font-semibold ${config.tone}`}>
              {currentProgress.toFixed(0)}%
            </p>
          </div>
          <Progress className="h-2.5 bg-muted">
            <ProgressValue value={currentProgress} />
          </Progress>
          {progressPages && (
            <p className="mt-3 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
              {progressPages}
            </p>
          )}
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border border-border/60 bg-background/70 p-4">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
              Matches fetched
            </p>
            <p className="mt-2 text-2xl font-semibold text-foreground">
              {matchesFetched?.toLocaleString() ?? "0"}
            </p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/70 p-4">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
              Matches saved
            </p>
            <p className="mt-2 text-2xl font-semibold text-foreground">
              {matchesSaved?.toLocaleString() ?? "—"}
            </p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/70 p-4">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
              Started
            </p>
            <p className="mt-2 text-sm font-semibold text-foreground">
              {formatTimestamp(startedAt)}
            </p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/70 p-4">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
              Runtime
            </p>
            <p className="mt-2 text-sm font-semibold text-foreground">
              {formatDuration(startedAt, completedAt)}
            </p>
          </div>
        </div>

        {errorMessage && (
          <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-700 dark:text-red-300">
            <p className="font-semibold">Latest failure detail</p>
            <p className="mt-1">{errorMessage}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
