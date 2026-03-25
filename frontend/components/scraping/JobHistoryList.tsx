"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress, ProgressValue } from "@/components/ui/progress";
import type { ScrapingJob } from "@/lib/api/scraping";
import { CheckCircle2Icon, ClockIcon, RefreshCwIcon, XCircleIcon } from "lucide-react";

interface JobHistoryListProps {
  jobs: ScrapingJob[];
  isLoading?: boolean;
  onRefresh?: () => void;
  onSelectJob?: (job: ScrapingJob) => void;
  selectedJobId?: string | null;
}

const statusFilters = [
  "all",
  "pending",
  "running",
  "completed",
  "failed",
  "cancelled",
] as const;

function getStatusMeta(status: ScrapingJob["status"]) {
  switch (status) {
    case "pending":
      return {
        badge: "secondary" as const,
        icon: ClockIcon,
        tone: "text-amber-500",
        label: "Queued",
      };
    case "running":
      return {
        badge: "default" as const,
        icon: RefreshCwIcon,
        tone: "text-sky-500",
        label: "Running",
      };
    case "completed":
      return {
        badge: "success" as const,
        icon: CheckCircle2Icon,
        tone: "text-emerald-500",
        label: "Completed",
      };
    case "failed":
      return {
        badge: "destructive" as const,
        icon: XCircleIcon,
        tone: "text-red-500",
        label: "Failed",
      };
    default:
      return {
        badge: "outline" as const,
        icon: ClockIcon,
        tone: "text-muted-foreground",
        label: "Cancelled",
      };
  }
}

function formatTimestamp(value: string) {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatJobLabel(job: ScrapingJob) {
  if (!job.tournament_url) {
    return job.scraping_type === "upcoming" ? "All football listing" : "Direct crawl";
  }

  const segments = job.tournament_url.split("/").filter(Boolean);
  const normalizedSegments =
    segments[segments.length - 1] === "results" ? segments.slice(0, -1) : segments;
  return normalizedSegments.slice(-2).join(" / ").replace(/-/g, " ");
}

export function JobHistoryList({
  jobs,
  isLoading = false,
  onRefresh,
  onSelectJob,
  selectedJobId,
}: JobHistoryListProps) {
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const filteredJobs = useMemo(() => {
    if (statusFilter === "all") {
      return jobs;
    }

    return jobs.filter((job) => job.status === statusFilter);
  }, [jobs, statusFilter]);

  return (
    <Card className="border-border/70 bg-card/90 shadow-[0_20px_50px_-34px_rgba(15,23,42,0.55)]">
      <CardHeader className="border-b border-border/60 bg-gradient-to-br from-background via-background to-muted/30">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <CardTitle className="text-xl">Recent Jobs</CardTitle>
            <CardDescription className="mt-1 text-sm leading-6">
              Review the newest scraping runs, switch the focused monitor target, and
              spot failed or stale jobs quickly.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-36 border-border/70 bg-background/70">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {statusFilters.map((status) => (
                  <SelectItem key={status} value={status}>
                    {status === "all"
                      ? "All statuses"
                      : status.charAt(0).toUpperCase() + status.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {onRefresh && (
              <Button
                variant="outline"
                size="sm"
                onClick={onRefresh}
                disabled={isLoading}
              >
                <RefreshCwIcon
                  className={`mr-2 h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
                />
                Refresh
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6">
        {isLoading ? (
          <div className="flex min-h-52 items-center justify-center text-sm text-muted-foreground">
            <RefreshCwIcon className="mr-2 h-4 w-4 animate-spin" />
            Loading recent jobs...
          </div>
        ) : filteredJobs.length === 0 ? (
          <div className="flex min-h-52 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
            <ClockIcon className="h-8 w-8" />
            <p className="font-medium text-foreground">No jobs match this filter</p>
            <p className="text-sm">
              Start a scraping run to populate the operator console history.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredJobs.slice(0, 12).map((job) => {
              const meta = getStatusMeta(job.status);
              const StatusIcon = meta.icon;
              const isSelected = job.id === selectedJobId;

              return (
                <button
                  key={job.id}
                  type="button"
                  onClick={() => onSelectJob?.(job)}
                  className={`w-full rounded-2xl border p-4 text-left transition-all ${
                    isSelected
                      ? "border-primary bg-primary/10 shadow-[0_0_0_1px_hsl(var(--primary)/0.35)]"
                      : "border-border/60 bg-background/60 hover:border-primary/40 hover:bg-background"
                  }`}
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant={meta.badge} className="gap-1.5">
                          <StatusIcon
                            className={`h-3.5 w-3.5 ${
                              job.status === "running" ? "animate-spin" : ""
                            }`}
                          />
                          {meta.label}
                        </Badge>
                        <Badge
                          variant="outline"
                          className="border-border/60 bg-background/70"
                        >
                          {job.scraping_type === "upcoming" ? "Upcoming" : "Results"}
                        </Badge>
                        <Badge
                          variant="outline"
                          className="border-border/60 bg-background/70"
                        >
                          {formatJobLabel(job)}
                        </Badge>
                      </div>

                      <div className="space-y-1">
                        <p className="text-sm font-semibold text-foreground">
                          {job.message ?? "No status message"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Created {formatTimestamp(job.created_at)} · ID{" "}
                          {job.id.slice(0, 8)}...
                        </p>
                      </div>
                    </div>

                    <div className="grid gap-3 text-sm text-muted-foreground sm:grid-cols-3 lg:min-w-72">
                      <div>
                        <p className="text-xs uppercase tracking-[0.14em]">Progress</p>
                        <p className={`mt-1 font-semibold ${meta.tone}`}>
                          {job.progress.toFixed(0)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.14em]">Matches</p>
                        <p className="mt-1 font-semibold text-foreground">
                          {job.matches_scraped.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.14em]">Errors</p>
                        <p className="mt-1 font-semibold text-foreground">
                          {job.errors.length}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <Progress className="h-2 bg-muted">
                      <ProgressValue value={job.progress} />
                    </Progress>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
