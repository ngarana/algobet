"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTournaments, useTournamentSeasons } from "@/lib/queries/use-tournaments";
import { useTeams } from "@/lib/queries/use-teams";
import { FetchDialogType, type FetchDialogTypeValue } from "@/lib/constants/fetch";
import { FETCH_DIALOG_CONFIG } from "@/lib/constants/fetch";
import { PlayIcon } from "lucide-react";

const ALL_SCOPE = "all";
const LEAGUE_SCOPE = "league";
const NO_COUNTRY = "__none__";
const NO_LEAGUE = "__none__";
const NO_TEAM = "__none__";

type FetchDialogSubmitData =
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
    };

interface FetchDialogProps {
  type: FetchDialogTypeValue;
  onConfirm: (data: FetchDialogSubmitData) => void;
  onClose: () => void;
  isLoading?: boolean;
}

export function FetchDialog({
  type,
  onConfirm,
  onClose,
  isLoading = false,
}: FetchDialogProps) {
  const [scope, setScope] = useState<"all" | "league">(
    type === FetchDialogType.RESULTS ? LEAGUE_SCOPE : ALL_SCOPE
  );
  const [selectedCountry, setSelectedCountry] = useState<string>(NO_COUNTRY);
  const [selectedTournamentId, setSelectedTournamentId] = useState<string>(NO_LEAGUE);
  const [selectedTeamId, setSelectedTeamId] = useState<string>(NO_TEAM);
  const [date, setDate] = useState("");
  const [period, setPeriod] = useState("");
  const [maxPages, setMaxPages] = useState("");
  const [resultsInputMode, setResultsInputMode] = useState<"select" | "link">("select");
  const [upcomingInputMode, setUpcomingInputMode] = useState<"select" | "link">(
    "select"
  );
  const [leagueLink, setLeagueLink] = useState("");
  const [upcomingLeagueLink, setUpcomingLeagueLink] = useState("");
  const [periodStart, setPeriodStart] = useState("");
  const [periodEnd, setPeriodEnd] = useState("");
  const [importDivision, setImportDivision] = useState("");
  const [importSeason, setImportSeason] = useState("");

  const config = FETCH_DIALOG_CONFIG[type];
  const { data: tournaments = [], isLoading: tournamentsLoading } = useTournaments();
  const tournamentId =
    selectedTournamentId !== NO_LEAGUE ? Number(selectedTournamentId) : null;
  const teamId = selectedTeamId !== NO_TEAM ? Number(selectedTeamId) : null;
  const { data: seasons = [], isLoading: seasonsLoading } =
    useTournamentSeasons(tournamentId);
  const { data: teams = [], isLoading: teamsLoading } = useTeams(
    tournamentId ? { tournament_id: tournamentId } : undefined
  );

  useEffect(() => {
    if (type === FetchDialogType.RESULTS) {
      setScope(LEAGUE_SCOPE);
    } else {
      setScope(ALL_SCOPE);
    }
    setSelectedCountry(NO_COUNTRY);
    setSelectedTournamentId(NO_LEAGUE);
    setSelectedTeamId(NO_TEAM);
    setDate("");
    setPeriod("");
    setMaxPages("");
    setResultsInputMode("select");
    setUpcomingInputMode("select");
    setLeagueLink("");
    setUpcomingLeagueLink("");
    setPeriodStart("");
    setPeriodEnd("");
    setImportDivision("");
    setImportSeason("");
  }, [type]);

  const countries = useMemo(
    () => [...new Set(tournaments.map((tournament) => tournament.country))].sort(),
    [tournaments]
  );

  const filteredTournaments = useMemo(() => {
    if (selectedCountry === NO_COUNTRY) {
      return [];
    }
    return tournaments
      .filter((tournament) => tournament.country === selectedCountry)
      .sort((left, right) => left.name.localeCompare(right.name));
  }, [selectedCountry, tournaments]);

  useEffect(() => {
    if (
      selectedTournamentId !== NO_LEAGUE &&
      !filteredTournaments.some(
        (tournament) => String(tournament.id) === selectedTournamentId
      )
    ) {
      setSelectedTournamentId(NO_LEAGUE);
    }
  }, [filteredTournaments, selectedTournamentId]);

  const selectedTournament = filteredTournaments.find(
    (tournament) => String(tournament.id) === selectedTournamentId
  );

  const requiresLeague =
    (type === FetchDialogType.RESULTS && resultsInputMode === "select") ||
    (type === FetchDialogType.UPCOMING &&
      scope === LEAGUE_SCOPE &&
      upcomingInputMode === "select") ||
    (type === FetchDialogType.BY_DATE && scope === LEAGUE_SCOPE);
  const hasSeasonSuggestions = seasons.length > 0;

  const handleClose = () => {
    onClose();
  };

  const handleConfirm = () => {
    if (type === FetchDialogType.UPCOMING) {
      if (scope === LEAGUE_SCOPE && upcomingInputMode === "link") {
        onConfirm({
          type: "upcoming",
          scope: LEAGUE_SCOPE,
          tournament_url: upcomingLeagueLink,
        });
      } else {
        onConfirm({
          type: "upcoming",
          scope,
          tournament_id: requiresLeague ? (tournamentId ?? undefined) : undefined,
          team_id: requiresLeague ? (teamId ?? undefined) : undefined,
        });
      }
    } else if (type === FetchDialogType.RESULTS) {
      if (resultsInputMode === "link") {
        onConfirm({
          type: "results",
          tournament_url: leagueLink,
          period_start: periodStart || undefined,
          period_end: periodEnd || undefined,
          max_pages: maxPages ? Number(maxPages) : undefined,
        });
      } else if (tournamentId) {
        onConfirm({
          type: "results",
          tournament_id: tournamentId,
          period: period || undefined,
          period_start: periodStart || undefined,
          period_end: periodEnd || undefined,
          max_pages: maxPages ? Number(maxPages) : undefined,
          team_id: teamId ?? undefined,
        });
      }
    } else if (type === FetchDialogType.BY_DATE) {
      onConfirm({
        type: "by-date",
        scope,
        date: date || undefined,
        tournament_id: requiresLeague ? (tournamentId ?? undefined) : undefined,
        team_id: requiresLeague ? (teamId ?? undefined) : undefined,
      });
    } else if (type === FetchDialogType.IMPORT) {
      onConfirm({
        type: "import",
        division: importDivision,
        season: importSeason,
      });
    }
    handleClose();
  };

  const isConfirmDisabled =
    isLoading ||
    (type === FetchDialogType.UPCOMING &&
      scope === LEAGUE_SCOPE &&
      upcomingInputMode === "link" &&
      !upcomingLeagueLink.trim()) ||
    (type === FetchDialogType.RESULTS && resultsInputMode === "link"
      ? !leagueLink.trim()
      : requiresLeague && !tournamentId) ||
    (type === FetchDialogType.RESULTS &&
      resultsInputMode === "select" &&
      tournamentsLoading) ||
    (type === FetchDialogType.IMPORT && (!importDivision || !importSeason));

  return (
    <Card className="border-[#252a37] bg-[#12151d]">
      <CardContent className="p-6">
        <div className="mb-5 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#e0e6f0]">{config.title}</h2>
            <p className="mt-1 text-sm text-[#9ca3af]">
              {type === FetchDialogType.RESULTS
                ? "Pick a country, league, and season period for historical results."
                : "Choose whether to scrape all leagues or a specific league."}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="text-[#9ca3af] hover:text-[#e0e6f0]"
            aria-label="Close dialog"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          {type !== FetchDialogType.RESULTS && (
            <div className="space-y-2">
              <Label htmlFor="scope">Scope</Label>
              <Select
                value={scope}
                onValueChange={(value: "all" | "league") => {
                  setScope(value);
                  if (value === ALL_SCOPE) {
                    setSelectedCountry(NO_COUNTRY);
                    setSelectedTournamentId(NO_LEAGUE);
                    setSelectedTeamId(NO_TEAM);
                  }
                }}
              >
                <SelectTrigger id="scope" className="border-[#252a37] bg-[#161a25]">
                  <SelectValue placeholder="Select scope" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={ALL_SCOPE}>All leagues</SelectItem>
                  <SelectItem value={LEAGUE_SCOPE}>Specific league</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {type === FetchDialogType.UPCOMING && scope === LEAGUE_SCOPE && (
            <div className="space-y-2">
              <Label>League Selection Mode</Label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-sm text-[#e0e6f0]">
                  <input
                    type="radio"
                    name="upcomingInputMode"
                    checked={upcomingInputMode === "select"}
                    onChange={() => setUpcomingInputMode("select")}
                    className="h-4 w-4 accent-[#4ade80]"
                  />
                  Select from dropdown
                </label>
                <label className="flex items-center gap-2 text-sm text-[#e0e6f0]">
                  <input
                    type="radio"
                    name="upcomingInputMode"
                    checked={upcomingInputMode === "link"}
                    onChange={() => setUpcomingInputMode("link")}
                    className="h-4 w-4 accent-[#4ade80]"
                  />
                  Paste league link
                </label>
              </div>
            </div>
          )}

          {type === FetchDialogType.UPCOMING &&
            scope === LEAGUE_SCOPE &&
            upcomingInputMode === "link" && (
              <div className="space-y-2">
                <Label htmlFor="upcoming-league-link">League URL</Label>
                <Input
                  id="upcoming-league-link"
                  value={upcomingLeagueLink}
                  onChange={(e) => setUpcomingLeagueLink(e.target.value)}
                  placeholder="https://www.oddsportal.com/football/england/premier-league/"
                  className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                />
                <p className="text-xs text-[#9ca3af]">
                  Paste the OddsPortal league URL to scrape upcoming matches.
                </p>
              </div>
            )}

          {type === FetchDialogType.RESULTS && (
            <div className="space-y-2">
              <Label>League Selection Mode</Label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-sm text-[#e0e6f0]">
                  <input
                    type="radio"
                    name="resultsInputMode"
                    checked={resultsInputMode === "select"}
                    onChange={() => setResultsInputMode("select")}
                    className="h-4 w-4 accent-[#f59e0b]"
                  />
                  Select from dropdown
                </label>
                <label className="flex items-center gap-2 text-sm text-[#e0e6f0]">
                  <input
                    type="radio"
                    name="resultsInputMode"
                    checked={resultsInputMode === "link"}
                    onChange={() => setResultsInputMode("link")}
                    className="h-4 w-4 accent-[#f59e0b]"
                  />
                  Paste league link
                </label>
              </div>
            </div>
          )}

          {type === FetchDialogType.RESULTS &&
            resultsInputMode === "link" &&
            requiresLeague && (
              <div className="space-y-2">
                <Label htmlFor="league-link">League Results URL</Label>
                <Input
                  id="league-link"
                  value={leagueLink}
                  onChange={(e) => setLeagueLink(e.target.value)}
                  placeholder="https://www.oddsportal.com/football/england/premier-league/results/"
                  className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                />
                <p className="text-xs text-[#9ca3af]">
                  Paste the full OddsPortal results URL for the league you want to
                  scrape.
                </p>
              </div>
            )}

          {type === FetchDialogType.RESULTS && requiresLeague && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="period-start">Period Start (Optional)</Label>
                <Input
                  id="period-start"
                  value={periodStart}
                  onChange={(e) => setPeriodStart(e.target.value)}
                  placeholder="e.g. 2010-2011"
                  className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="period-end">Period End (Optional)</Label>
                <Input
                  id="period-end"
                  value={periodEnd}
                  onChange={(e) => setPeriodEnd(e.target.value)}
                  placeholder="e.g. 2024-2025"
                  className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                />
              </div>
            </div>
          )}

          {type === FetchDialogType.RESULTS && requiresLeague && (
            <p className="text-xs text-[#9ca3af]">
              Specify a period range to batch scrape multiple seasons. Leave blank for
              single season.
            </p>
          )}

          {requiresLeague &&
            !(type === FetchDialogType.RESULTS && resultsInputMode === "link") && (
              <>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="country">Country</Label>
                    <Select
                      value={selectedCountry}
                      onValueChange={(value) => {
                        setSelectedCountry(value);
                        setSelectedTournamentId(NO_LEAGUE);
                        setSelectedTeamId(NO_TEAM);
                        setPeriod("");
                      }}
                    >
                      <SelectTrigger
                        id="country"
                        className="border-[#252a37] bg-[#161a25]"
                      >
                        <SelectValue
                          placeholder={
                            tournamentsLoading
                              ? "Loading countries..."
                              : "Select country"
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        {countries.map((country) => (
                          <SelectItem key={country} value={country}>
                            {country}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="league">League</Label>
                    <Select
                      value={selectedTournamentId}
                      onValueChange={(value) => {
                        setSelectedTournamentId(value);
                        setSelectedTeamId(NO_TEAM);
                        setPeriod("");
                      }}
                      disabled={selectedCountry === NO_COUNTRY}
                    >
                      <SelectTrigger
                        id="league"
                        className="border-[#252a37] bg-[#161a25]"
                      >
                        <SelectValue
                          placeholder={
                            selectedCountry === NO_COUNTRY
                              ? "Choose country first"
                              : "Select league"
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        {filteredTournaments.map((tournament) => (
                          <SelectItem key={tournament.id} value={String(tournament.id)}>
                            {tournament.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="team">Team (Optional)</Label>
                    <Select
                      value={selectedTeamId}
                      onValueChange={setSelectedTeamId}
                      disabled={selectedTournamentId === NO_LEAGUE}
                    >
                      <SelectTrigger
                        id="team"
                        className="border-[#252a37] bg-[#161a25]"
                      >
                        <SelectValue
                          placeholder={
                            selectedTournamentId === NO_LEAGUE
                              ? "Choose league first"
                              : teamsLoading
                                ? "Loading teams..."
                                : "All Teams"
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={NO_TEAM}>All Teams</SelectItem>
                        {teams.map((team) => (
                          <SelectItem key={team.id} value={String(team.id)}>
                            {team.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {selectedTournament && (
                  <div className="rounded-lg border border-[#252a37] bg-[#161a25] px-3 py-2 text-sm text-[#9ca3af]">
                    Targeting{" "}
                    <span className="text-[#e0e6f0]">{selectedTournament.name}</span>
                    {" · "}
                    {selectedTournament.country}
                  </div>
                )}
              </>
            )}

          {type === FetchDialogType.RESULTS && (
            <>
              {hasSeasonSuggestions ? (
                <div className="space-y-2">
                  <Label htmlFor="period">Period</Label>
                  <Select value={period || undefined} onValueChange={setPeriod}>
                    <SelectTrigger
                      id="period"
                      className="border-[#252a37] bg-[#161a25]"
                      disabled={!tournamentId || seasonsLoading}
                    >
                      <SelectValue
                        placeholder={
                          !tournamentId
                            ? "Choose league first"
                            : seasonsLoading
                              ? "Loading periods..."
                              : "Select season period"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {seasons.map((season) => (
                        <SelectItem key={season.id} value={season.name}>
                          {season.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <div className="space-y-2">
                  <Label htmlFor="period">Period</Label>
                  <Input
                    id="period"
                    value={period}
                    onChange={(event) => setPeriod(event.target.value)}
                    placeholder="e.g. 2023/2024"
                    className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                    disabled={!tournamentId}
                  />
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="max-pages">Max Pages</Label>
                <Input
                  id="max-pages"
                  type="number"
                  min="1"
                  value={maxPages}
                  onChange={(event) => setMaxPages(event.target.value)}
                  placeholder="Optional page cap for faster runs"
                  className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                />
              </div>
            </>
          )}

          {type === FetchDialogType.BY_DATE && (
            <div className="space-y-2">
              <Label htmlFor="date">Date</Label>
              <Input
                id="date"
                type="date"
                value={date}
                onChange={(event) => setDate(event.target.value)}
                className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
              />
            </div>
          )}

          {type === FetchDialogType.IMPORT && (
            <div className="space-y-4">
              <div className="rounded-lg border border-[#06b6d4]/30 bg-[#06b6d4]/10 p-4">
                <p className="text-sm text-[#9ca3af]">
                  Import historical match data and statistics from Football-Data.co.uk.
                  This includes shots, corners, fouls, cards, and more.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="import-division">League</Label>
                <Select
                  value={importDivision}
                  onValueChange={setImportDivision}
                >
                  <SelectTrigger
                    id="import-division"
                    className="border-[#252a37] bg-[#161a25]"
                  >
                    <SelectValue placeholder="Select league" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="E0">Premier League (England)</SelectItem>
                    <SelectItem value="E1">Championship (England)</SelectItem>
                    <SelectItem value="D1">Bundesliga (Germany)</SelectItem>
                    <SelectItem value="D2">2. Bundesliga (Germany)</SelectItem>
                    <SelectItem value="I1">Serie A (Italy)</SelectItem>
                    <SelectItem value="I2">Serie B (Italy)</SelectItem>
                    <SelectItem value="SP1">La Liga (Spain)</SelectItem>
                    <SelectItem value="SP2">La Liga 2 (Spain)</SelectItem>
                    <SelectItem value="F1">Ligue 1 (France)</SelectItem>
                    <SelectItem value="F2">Ligue 2 (France)</SelectItem>
                    <SelectItem value="N1">Eredivisie (Netherlands)</SelectItem>
                    <SelectItem value="B1">First Division A (Belgium)</SelectItem>
                    <SelectItem value="P1">Primeira Liga (Portugal)</SelectItem>
                    <SelectItem value="T1">Super Lig (Turkey)</SelectItem>
                    <SelectItem value="G1">Super League (Greece)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="import-season">Season</Label>
                <Input
                  id="import-season"
                  value={importSeason}
                  onChange={(event) => setImportSeason(event.target.value)}
                  placeholder="e.g. 2324 for 2023/24"
                  className="border-[#252a37] bg-[#161a25] text-[#e0e6f0]"
                />
                <p className="text-xs text-[#9ca3af]">
                  Use format: YYYY (e.g., 2324 for 2023/24, 2425 for 2024/25)
                </p>
              </div>
            </div>
          )}

          <div className="flex gap-2 pt-2">
            <Button
              onClick={handleConfirm}
              disabled={isConfirmDisabled}
              className="font-semibold text-[#0a0c12]"
              style={{ backgroundColor: config.color }}
            >
              <PlayIcon className="mr-2 h-4 w-4" />
              Start Fetch
            </Button>
            <Button
              variant="outline"
              onClick={handleClose}
              className="border-[#252a37] text-[#9ca3af]"
            >
              Cancel
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
