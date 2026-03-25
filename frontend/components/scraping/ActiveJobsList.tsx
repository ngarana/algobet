"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress, ProgressValue } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  Loader2Icon,
  XIcon,
  ExternalLinkIcon,
} from "lucide-react";
import type { ScrapingJob } from "@/lib/api/scraping";

interface ActiveJobsListProps {
  jobs: ScrapingJob[];
  onCancel?: (jobId: string) => void;
  isCancelling?: boolean;
}

const statusConfig = {
  pending: {
    icon: ClockIcon,
    color: "text-yellow-500",
    bg: "bg-yellow-100",
    badge: "secondary" as const,
  },
  running: {
    icon: Loader2Icon,
    color: "text-blue-500",
    bg: "bg-blue-100",
    badge: "default" as const,
  },
  completed: {
    icon: CheckCircleIcon,
    color: "text-green-500",
    bg: "bg-green-100",
    badge: "success" as const,
  },
  failed: {
    icon: XCircleIcon,
    color: "text-red-500",
    bg: "bg-red-100",
    badge: "destructive" as const,
  },
  cancelled: {
    icon: XIcon,
    color: "text-gray-500",
    bg: "bg-gray-100",
    badge: "outline" as const,
  },
};

export function ActiveJobsList({
  jobs,
  onCancel,
  isCancelling,
}: ActiveJobsListProps) {
  const activeJobs = useMemo(
    () => jobs.filter((j) => j.status === "pending" || j.status === "running"),
    [jobs]
  );

  if (activeJobs.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Loader2Icon className="h-4 w-4 animate-spin" />
          Active Jobs ({activeJobs.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {activeJobs.map((job) => {
          const config = statusConfig[job.status];
          const StatusIcon = config.icon;
          const progressPercent = job.progress ?? 0;

          return (
            <div
              key={job.id}
              className="rounded-lg border p-4 space-y-3"
            >
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Badge variant={config.badge}>
                      <StatusIcon
                        className={`h-3 w-3 mr-1 ${
                          job.status === "running" ? "animate-spin" : ""
                        }`}
                      />
                      {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </Badge>
                    <Badge variant="outline">
                      {job.scraping_type === "upcoming" ? "Upcoming" : "Results"}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground font-mono">
                    ID: {job.id.slice(0, 8)}...
                  </p>
                </div>

                {job.status === "running" && onCancel && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onCancel(job.id)}
                    disabled={isCancelling}
                  >
                    <XIcon className="h-4 w-4" />
                  </Button>
                )}
              </div>

              {job.tournament_url && (
                <a
                  href={job.tournament_url ?? "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-blue-600 hover:underline flex items-center gap-1 truncate"
                >
                  <ExternalLinkIcon className="h-3 w-3 flex-shrink-0" />
                  {job.tournament_url}
                </a>
              )}

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Progress</span>
                  <span className="font-medium">{progressPercent.toFixed(0)}%</span>
                </div>
                <Progress className="h-2">
                  <ProgressValue value={progressPercent} />
                </Progress>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Matches</p>
                  <p className="font-medium">{job.matches_scraped}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Started</p>
                  <p className="font-medium">
                    {job.started_at
                      ? new Date(job.started_at).toLocaleTimeString()
                      : "Queued"}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Errors</p>
                  <p className={`font-medium ${job.errors?.length ? "text-red-500" : ""}`}>
                    {job.errors?.length ?? 0}
                  </p>
                </div>
              </div>

              {job.message && (
                <p className="text-sm text-muted-foreground bg-muted p-2 rounded">
                  {job.message}
                </p>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
