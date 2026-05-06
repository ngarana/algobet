"use client";

import Link from "next/link";
import { format } from "date-fns";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Match, MatchStatus } from "@/lib/types/api";

interface MatchCardProps {
  match: Match;
}

function getStatusColor(status: MatchStatus): string {
  switch (status) {
    case "LIVE":
      return "bg-red-500 text-white";
    case "FINISHED":
      return "bg-muted text-muted-foreground";
    case "SCHEDULED":
      return "bg-blue-500 text-white";
    default:
      return "bg-muted text-muted-foreground";
  }
}

function getStatusLabel(status: MatchStatus): string {
  switch (status) {
    case "LIVE":
      return "LIVE";
    case "FINISHED":
      return "Finished";
    case "SCHEDULED":
      return "Scheduled";
    default:
      return status;
  }
}

export function MatchCard({ match }: MatchCardProps) {
  const isFinished = match.status === "FINISHED";
  const isLive = match.status === "LIVE";
  const homeName = match.home_team_name ?? `Team ${match.home_team_id}`;
  const awayName = match.away_team_name ?? `Team ${match.away_team_id}`;

  return (
    <Link href={`/matches/${match.id}`}>
      <Card className="cursor-pointer transition-shadow hover:shadow-md">
        <CardContent className="p-4">
          <div className="mb-3 flex items-center justify-between">
            <Badge className={getStatusColor(match.status)}>
              {getStatusLabel(match.status)}
            </Badge>
            <span className="text-sm text-muted-foreground">
              {format(new Date(match.match_date), "MMM d, yyyy HH:mm")}
            </span>
          </div>

          <div className="flex items-center justify-between gap-4">
            {/* Home Team */}
            <div className="flex-1 text-right">
              <p className="font-semibold">{homeName}</p>
              <p className="text-sm text-muted-foreground">Home</p>
            </div>

            {/* Score */}
            <div className="flex items-center gap-2 px-4">
              {isFinished || isLive ? (
                <>
                  <span className="text-2xl font-bold">{match.home_score ?? 0}</span>
                  <span className="text-xl text-muted-foreground">-</span>
                  <span className="text-2xl font-bold">{match.away_score ?? 0}</span>
                </>
              ) : (
                <span className="text-lg text-muted-foreground">vs</span>
              )}
            </div>

            {/* Away Team */}
            <div className="flex-1 text-left">
              <p className="font-semibold">{awayName}</p>
              <p className="text-sm text-muted-foreground">Away</p>
            </div>
          </div>

          {match.tournament_name && (
            <p className="mt-3 text-center text-xs text-muted-foreground">
              {match.tournament_name}
            </p>
          )}

          {/* Odds */}
          {match.odds_home && match.odds_draw && match.odds_away && (
            <div className="mt-3 flex items-center justify-center gap-4 border-t pt-3 text-sm">
              <span className="text-muted-foreground">
                1:{" "}
                <span className="font-medium text-foreground">
                  {match.odds_home.toFixed(2)}
                </span>
              </span>
              <span className="text-muted-foreground">
                X:{" "}
                <span className="font-medium text-foreground">
                  {match.odds_draw.toFixed(2)}
                </span>
              </span>
              <span className="text-muted-foreground">
                2:{" "}
                <span className="font-medium text-foreground">
                  {match.odds_away.toFixed(2)}
                </span>
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
