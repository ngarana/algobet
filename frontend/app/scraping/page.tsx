"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  JobHistoryList,
  ScrapeFormCard,
  ScrapingLiveMonitor,
  ScrapingStatsCards,
} from "@/components/scraping";
import { useScrapingProgress } from "@/hooks/useScrapingProgress";
import {
  useScrapeByDate,
  useScrapeResults,
  useScrapeUpcoming,
  useScrapingJobs,
  useScrapingStats,
} from "@/lib/queries/use-scraping";
import type { ScrapingJob } from "@/lib/api/scraping";
import {
  ActivityIcon,
  AlertTriangleIcon,
  CalendarIcon,
  RadarIcon,
  RefreshCwIcon,
} from "lucide-react";

function isActiveJob(job: ScrapingJob) {
  return job.status === "pending" || job.status === "running";
}

export default function ScrapingPage() {
  const [commandTab, setCommandTab] = useState<"upcoming" | "results">("upcoming");
  const [pageError, setPageError] = useState<string | null>(null);
  const [socketMessage, setSocketMessage] = useState<string | null>(null);
  const [focusedJobId, setFocusedJobId] = useState<string | null>(null);

  const {
    data: jobsData,
    isLoading: jobsLoading,
    isFetching: jobsFetching,
    refetch: refreshJobs,
  } = useScrapingJobs();
  const {
    data: stats,
    isLoading: statsLoading,
    isFetching: statsFetching,
    refetch: refreshStats,
  } = useScrapingStats();

  const scrapeUpcomingMutation = useScrapeUpcoming();
  const scrapeResultsMutation = useScrapeResults();
  const scrapeByDateMutation = useScrapeByDate();

  const jobs = useMemo(
    () =>
      [...(jobsData?.items ?? [])].sort(
        (left, right) =>
          new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
      ),
    [jobsData?.items]
  );

  const activeJobs = useMemo(() => jobs.filter(isActiveJob), [jobs]);
  const focusedJob = jobs.find((job) => job.id === focusedJobId) ?? null;
  const liveSocketJobId =
    focusedJob && isActiveJob(focusedJob)
      ? focusedJob.id
      : activeJobs.length > 0
        ? activeJobs[0].id
        : undefined;

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshJobs(), refreshStats()]);
  }, [refreshJobs, refreshStats]);

  const { isConnected, currentProgress } = useScrapingProgress({
    jobId: liveSocketJobId,
    enabled: Boolean(liveSocketJobId),
    onConnected: () => {
      setSocketMessage(null);
    },
    onDisconnected: () => {
      if (liveSocketJobId) {
        setSocketMessage("Live socket disconnected. Polling fallback is still active.");
      }
    },
    onError: () => {
      setSocketMessage(
        "We hit a WebSocket issue, so the console is falling back to scheduled polling."
      );
    },
    onProgress: (progress) => {
      if (
        progress.status &&
        ["completed", "failed", "cancelled"].includes(progress.status)
      ) {
        void refreshAll();
      }
    },
  });

  const focusedProgress =
    focusedJob && currentProgress?.job_id === focusedJob.id ? currentProgress : null;

  useEffect(() => {
    if (jobs.length === 0) {
      if (focusedJobId !== null) {
        setFocusedJobId(null);
      }
      return;
    }

    const focusedStillExists = jobs.some((job) => job.id === focusedJobId);
    if (!focusedStillExists) {
      setFocusedJobId(activeJobs[0]?.id ?? jobs[0]?.id ?? null);
    }
  }, [activeJobs, focusedJobId, jobs]);

  useEffect(() => {
    if (!pageError) {
      return;
    }

    const timer = window.setTimeout(() => setPageError(null), 5000);
    return () => window.clearTimeout(timer);
  }, [pageError]);

  useEffect(() => {
    if (!liveSocketJobId) {
      setSocketMessage(null);
    }
  }, [liveSocketJobId]);

  const handleScrapeUpcoming = useCallback(
    async (data: { leagueIds?: number[] }) => {
      setPageError(null);

      try {
        const job = await scrapeUpcomingMutation.mutateAsync({
          league_ids: data.leagueIds,
        });
        setFocusedJobId(job.id);
        await refreshAll();
      } catch (error) {
        console.error("Error starting upcoming scrape:", error);
        setPageError(
          "Could not start the upcoming scrape. Check your API-Football key configuration."
        );
      }
    },
    [refreshAll, scrapeUpcomingMutation]
  );

  const handleScrapeResults = useCallback(
    async (data: { leagueId?: number; maxResults?: number }) => {
      setPageError(null);

      try {
        const job = await scrapeResultsMutation.mutateAsync({
          league_id: data.leagueId,
          max_results: data.maxResults,
        });
        setFocusedJobId(job.id);
        await refreshAll();
      } catch (error) {
        console.error("Error starting results scrape:", error);
        setPageError(
          "Could not start the results scrape. Check your API-Football key configuration."
        );
      }
    },
    [refreshAll, scrapeResultsMutation]
  );

  const handleScrapeByDate = useCallback(
    async (date?: string) => {
      setPageError(null);

      try {
        const job = await scrapeByDateMutation.mutateAsync({
          date,
        });
        setFocusedJobId(job.id);
        await refreshAll();
      } catch (error) {
        console.error("Error starting by-date scrape:", error);
        setPageError(
          "Could not start the by-date scrape. Check your API-Football key configuration."
        );
      }
    },
    [refreshAll, scrapeByDateMutation]
  );

  const workspaceRefreshing = jobsFetching || statsFetching;
  const isAnyLoading =
    scrapeUpcomingMutation.isPending ||
    scrapeResultsMutation.isPending ||
    scrapeByDateMutation.isPending;

  return (
    <div className="space-y-6 pb-8">
      <section className="relative overflow-hidden rounded-[28px] border border-border/70 bg-gradient-to-br from-background via-background to-muted/40 p-6 shadow-[0_32px_80px_-48px_rgba(15,23,42,0.7)] dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(56,189,248,0.18),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.12),transparent_24%)]" />

        <div className="relative flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="border-border/60 bg-background/70">
                API-Football integration
              </Badge>
              <Badge
                variant="outline"
                className={`border-border/60 bg-background/70 ${
                  activeJobs.length > 0 && isConnected
                    ? "text-emerald-500"
                    : activeJobs.length > 0
                      ? "text-amber-500"
                      : "text-muted-foreground"
                }`}
              >
                <ActivityIcon className="mr-1.5 h-3.5 w-3.5" />
                {activeJobs.length > 0 && isConnected
                  ? "Live socket attached"
                  : activeJobs.length > 0
                    ? "Polling fallback"
                    : "Idle"}
              </Badge>
            </div>

            <div className="space-y-3">
              <h1 className="max-w-3xl text-3xl font-semibold tracking-tight text-foreground dark:text-slate-50">
                Fetch football fixtures and results via API-Football.
              </h1>
              <p className="max-w-3xl text-sm leading-7 text-muted-foreground dark:text-slate-300">
                Select leagues to fetch upcoming matches with odds, or retrieve
                historical results. No web scraping required - data comes directly
                from the API-Football service.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <Button
                variant="outline"
                onClick={() => handleScrapeByDate()}
                disabled={isAnyLoading}
                className="border-primary/50 bg-primary/5 hover:bg-primary/10"
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                Fetch All Today
                <Badge variant="secondary" className="ml-2">
                  1 req
                </Badge>
              </Button>

              <div className="rounded-2xl border border-border/60 bg-background/70 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  Active jobs
                </p>
                <p className="mt-1 text-2xl font-semibold text-foreground dark:text-slate-50">
                  {activeJobs.length}
                </p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-background/70 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  Focused monitor
                </p>
                <p className="mt-1 text-sm font-semibold text-foreground dark:text-slate-50">
                  {focusedJob
                    ? `${focusedJob.scraping_type === "upcoming" ? "Upcoming" : focusedJob.scraping_type === "by-date" ? "By Date" : "Results"} job ${focusedJob.id.slice(0, 8)}...`
                    : "No job selected"}
                </p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-background/70 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  Last queued
                </p>
                <p className="mt-1 text-sm font-semibold text-foreground dark:text-slate-50">
                  {jobs[0]
                    ? new Date(jobs[0].created_at).toLocaleString(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                      })
                    : "No runs yet"}
                </p>
              </div>
            </div>
          </div>

          <Button
            variant="outline"
            onClick={() => refreshAll()}
            disabled={workspaceRefreshing}
            className="border-border/60 bg-background/70"
          >
            <RefreshCwIcon
              className={`mr-2 h-4 w-4 ${workspaceRefreshing ? "animate-spin" : ""}`}
            />
            Refresh console
          </Button>
        </div>
      </section>

      {pageError && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="flex items-start gap-3 p-4 text-sm text-destructive">
            <AlertTriangleIcon className="mt-0.5 h-4 w-4" />
            <p>{pageError}</p>
          </CardContent>
        </Card>
      )}

      <ScrapingStatsCards stats={stats} isLoading={statsLoading} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(380px,0.95fr)]">
        <div className="space-y-6">
          <Tabs
            value={commandTab}
            onValueChange={(value) => setCommandTab(value as "upcoming" | "results")}
            className="space-y-4"
          >
            <TabsList className="grid w-full grid-cols-2 rounded-2xl border border-border/60 bg-muted/40 p-1">
              <TabsTrigger value="upcoming" className="rounded-xl">
                <RadarIcon className="mr-2 h-4 w-4" />
                Upcoming
              </TabsTrigger>
              <TabsTrigger value="results" className="rounded-xl">
                <RefreshCwIcon className="mr-2 h-4 w-4" />
                Historical Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upcoming" className="mt-0">
              <ScrapeFormCard
                type="upcoming"
                onSubmit={handleScrapeUpcoming}
                isLoading={scrapeUpcomingMutation.isPending}
              />
            </TabsContent>

            <TabsContent value="results" className="mt-0">
              <ScrapeFormCard
                type="results"
                onSubmit={handleScrapeResults}
                isLoading={scrapeResultsMutation.isPending}
              />
            </TabsContent>
          </Tabs>
        </div>

        <div className="space-y-6">
          <ScrapingLiveMonitor
            job={focusedJob}
            progress={focusedProgress}
            isConnected={isConnected}
            connectionMessage={socketMessage}
            onRefresh={() => refreshAll()}
            isRefreshing={workspaceRefreshing}
          />

          <JobHistoryList
            jobs={jobs}
            isLoading={jobsLoading}
            onRefresh={() => refreshAll()}
            onSelectJob={(job) => setFocusedJobId(job.id)}
            selectedJobId={focusedJobId}
          />
        </div>
      </div>
    </div>
  );
}
