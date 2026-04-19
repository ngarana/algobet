"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { FetchStats } from "@/lib/api/fetch";
import { ActivityIcon, GaugeIcon, Layers3Icon, ShieldCheckIcon } from "lucide-react";

interface FetchStatsCardsProps {
  stats: FetchStats | undefined;
  isLoading?: boolean;
}

const EMPTY_STATS: FetchStats = {
  total_jobs: 0,
  completed_jobs: 0,
  failed_jobs: 0,
  running_jobs: 0,
  total_matches_fetched: 0,
  average_duration_seconds: null,
  success_rate: 0,
};

function formatDuration(seconds: number | null) {
  if (!seconds) {
    return "No completed runs yet";
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}m ${remainingSeconds}s avg duration`;
}

export function FetchStatsCards({ stats, isLoading = false }: FetchStatsCardsProps) {
  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[...Array(4)].map((_, index) => (
          <Card key={index} className="border-border/70 bg-card/90">
            <CardContent className="space-y-4 p-5">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-28" />
              <Skeleton className="h-4 w-36" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const safeStats = stats ?? EMPTY_STATS;

  const statCards = [
    {
      label: "Total jobs",
      value: safeStats.total_jobs.toLocaleString(),
      helper: `${safeStats.running_jobs} currently active`,
      icon: Layers3Icon,
    },
    {
      label: "Matches fetched",
      value: safeStats.total_matches_fetched.toLocaleString(),
      helper: formatDuration(safeStats.average_duration_seconds),
      icon: ActivityIcon,
    },
    {
      label: "Success rate",
      value: `${safeStats.success_rate.toFixed(1)}%`,
      helper: `${safeStats.completed_jobs} completed / ${safeStats.failed_jobs} failed`,
      icon: ShieldCheckIcon,
    },
    {
      label: "Live throughput",
      value: safeStats.running_jobs.toLocaleString(),
      helper:
        safeStats.running_jobs > 0
          ? "Monitoring in-flight runs"
          : "No live jobs right now",
      icon: GaugeIcon,
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {statCards.map((card) => {
        const Icon = card.icon;

        return (
          <Card
            key={card.label}
            className="overflow-hidden border-border/70 bg-card/90 shadow-[0_20px_50px_-36px_rgba(15,23,42,0.5)]"
          >
            <CardContent className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                    {card.label}
                  </p>
                  <p className="text-3xl font-semibold tracking-tight text-foreground">
                    {card.value}
                  </p>
                  <p className="text-sm text-muted-foreground">{card.helper}</p>
                </div>
                <div className="rounded-2xl border border-border/60 bg-muted/30 p-3">
                  <Icon className="h-5 w-5 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

export function FetchStatsSummary({ stats }: { stats: FetchStats | undefined }) {
  const safeStats = stats ?? EMPTY_STATS;

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
        <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
          Completed jobs
        </p>
        <p className="mt-2 text-2xl font-semibold text-foreground">
          {safeStats.completed_jobs.toLocaleString()}
        </p>
      </div>
      <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
        <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
          Failed jobs
        </p>
        <p className="mt-2 text-2xl font-semibold text-foreground">
          {safeStats.failed_jobs.toLocaleString()}
        </p>
      </div>
    </div>
  );
}
