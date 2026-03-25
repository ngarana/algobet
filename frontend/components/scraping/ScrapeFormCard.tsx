"use client";

import { useState, type FormEvent } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import {
  HistoryIcon,
  Loader2Icon,
  PlayIcon,
  RadarIcon,
  TrophyIcon,
} from "lucide-react";
import { POPULAR_LEAGUES } from "@/lib/api/scraping";

interface ScrapeFormCardProps {
  type: "upcoming" | "results";
  onSubmit: (data: { leagueIds?: number[]; leagueId?: number; maxResults?: number }) => Promise<void>;
  isLoading?: boolean;
}

export function ScrapeFormCard({
  type,
  onSubmit,
  isLoading = false,
}: ScrapeFormCardProps) {
  const isUpcoming = type === "upcoming";
  const [selectedLeagues, setSelectedLeagues] = useState<number[]>(
    isUpcoming ? [39, 140, 135, 78, 61] : [] // Default top 5 leagues for upcoming
  );
  const [selectedLeague, setSelectedLeague] = useState<string>("");
  const [maxResults, setMaxResults] = useState("20");

  const handleLeagueToggle = (leagueId: number) => {
    setSelectedLeagues((prev) =>
      prev.includes(leagueId)
        ? prev.filter((id) => id !== leagueId)
        : [...prev, leagueId]
    );
  };

  const handleSelectAll = () => {
    if (selectedLeagues.length === POPULAR_LEAGUES.length) {
      setSelectedLeagues([]);
    } else {
      setSelectedLeagues(POPULAR_LEAGUES.map((l) => l.id));
    }
  };

  const handleUpcomingSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (selectedLeagues.length === 0) return;

    await onSubmit({ leagueIds: selectedLeagues });
  };

  const handleResultsSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedLeague) return;

    await onSubmit({
      leagueId: parseInt(selectedLeague, 10),
      maxResults: parseInt(maxResults, 10) || 20,
    });
  };

  return (
    <Card className="overflow-hidden border-border/70 bg-card/90 shadow-[0_20px_50px_-32px_rgba(15,23,42,0.6)]">
      <CardHeader className="border-b border-border/60 bg-gradient-to-br from-background via-background to-muted/30">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <Badge
              variant="outline"
              className="w-fit border-border/60 bg-background/70"
            >
              {isUpcoming ? "Upcoming matches" : "Historical results"}
            </Badge>
            <div className="space-y-2">
              <CardTitle className="flex items-center gap-2 text-xl">
                {isUpcoming ? (
                  <RadarIcon className="h-5 w-5 text-primary" />
                ) : (
                  <HistoryIcon className="h-5 w-5 text-primary" />
                )}
                {isUpcoming ? "Fetch Upcoming Matches" : "Fetch Match Results"}
              </CardTitle>
              <CardDescription className="max-w-2xl text-sm leading-6">
                {isUpcoming
                  ? "Select leagues to fetch upcoming fixtures and odds via API-Football. Data is retrieved instantly without web scraping."
                  : "Choose a league and set a limit to fetch recent match results with scores and odds."}
              </CardDescription>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6">
        {isUpcoming ? (
          <form onSubmit={handleUpcomingSubmit} className="space-y-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-base font-semibold">Select Leagues</Label>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={handleSelectAll}
                  className="text-xs"
                >
                  {selectedLeagues.length === POPULAR_LEAGUES.length
                    ? "Deselect All"
                    : "Select All"}
                </Button>
              </div>

              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {POPULAR_LEAGUES.map((league) => (
                  <div
                    key={league.id}
                    className={`flex items-center gap-3 rounded-xl border p-3 transition-colors cursor-pointer ${
                      selectedLeagues.includes(league.id)
                        ? "border-primary/50 bg-primary/5"
                        : "border-border/60 bg-background/70 hover:bg-muted/30"
                    }`}
                    onClick={() => handleLeagueToggle(league.id)}
                  >
                    <Checkbox
                      checked={selectedLeagues.includes(league.id)}
                      onCheckedChange={() => handleLeagueToggle(league.id)}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {league.flag} {league.name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {league.country}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-muted/20 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-foreground">
                  API-Football Integration
                </p>
                <p className="text-sm text-muted-foreground">
                  {selectedLeagues.length} league{selectedLeagues.length !== 1 ? "s" : ""} selected.
                  Fetches up to 10 upcoming fixtures per league.
                </p>
              </div>
              <Button
                type="submit"
                disabled={isLoading || selectedLeagues.length === 0}
                className="min-w-40"
              >
                {isLoading ? (
                  <>
                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                    Fetching...
                  </>
                ) : (
                  <>
                    <PlayIcon className="mr-2 h-4 w-4" />
                    Fetch upcoming
                  </>
                )}
              </Button>
            </div>
          </form>
        ) : (
          <form onSubmit={handleResultsSubmit} className="space-y-5">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,220px)]">
              <div className="space-y-2">
                <Label htmlFor="league-select">Select League</Label>
                <Select value={selectedLeague} onValueChange={setSelectedLeague}>
                  <SelectTrigger className="border-border/70 bg-background/70">
                    <SelectValue placeholder="Choose a league..." />
                  </SelectTrigger>
                  <SelectContent>
                    {POPULAR_LEAGUES.map((league) => (
                      <SelectItem key={league.id} value={league.id.toString()}>
                        {league.flag} {league.name} ({league.country})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="max-results">Max Results</Label>
                <Input
                  id="max-results"
                  type="number"
                  min={1}
                  max={100}
                  value={maxResults}
                  onChange={(e) => setMaxResults(e.target.value)}
                  className="border-border/70 bg-background/70"
                />
              </div>
            </div>

            {selectedLeague && (
              <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
                <div className="flex items-center gap-3">
                  <TrophyIcon className="h-5 w-5 text-primary" />
                  <div>
                    <p className="text-sm font-semibold">
                      {POPULAR_LEAGUES.find((l) => l.id.toString() === selectedLeague)?.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Fetching last {maxResults} completed matches with scores and odds
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-muted/20 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-foreground">
                  Historical Results
                </p>
                <p className="text-sm text-muted-foreground">
                  Fetches completed match results via API-Football with final scores and betting odds.
                </p>
              </div>
              <Button
                type="submit"
                disabled={isLoading || !selectedLeague}
                className="min-w-40"
              >
                {isLoading ? (
                  <>
                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                    Fetching...
                  </>
                ) : (
                  <>
                    <PlayIcon className="mr-2 h-4 w-4" />
                    Fetch results
                  </>
                )}
              </Button>
            </div>
          </form>
        )}
      </CardContent>
    </Card>
  );
}
