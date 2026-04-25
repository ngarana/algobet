import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Brain, Calendar, TrendingUp, BarChart3, Download } from "lucide-react";
import type { Prediction, PredictionWithMatch } from "@/lib/types/api";
import { ConfidenceIndicator } from "@/components/predictions/ConfidenceIndicator";
import ValueBetIndicator from "@/components/predictions/ValueBetIndicator";

interface PredictionDetailModalProps {
  prediction: Prediction | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExport?: (prediction: Prediction) => void;
}

function getOutcomeLabel(outcome: string): string {
  switch (outcome) {
    case "H":
      return "Home Win";
    case "D":
      return "Draw";
    case "A":
      return "Away Win";
    default:
      return outcome;
  }
}

function getOutcomeColor(outcome: string): string {
  switch (outcome) {
    case "H":
      return "text-blue-600";
    case "D":
      return "text-amber-600";
    case "A":
      return "text-red-600";
    default:
      return "";
  }
}

export default function PredictionDetailModal({
  prediction,
  open,
  onOpenChange,
  onExport,
}: PredictionDetailModalProps) {
  const [activeTab, setActiveTab] = useState("overview");

  if (!prediction) return null;

  const match = prediction.match as PredictionWithMatch["match"] | null;
  const model = prediction.model_version;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Prediction Details
          </DialogTitle>
          <DialogDescription>
            Detailed view of prediction #{prediction.id}
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="match">Match Details</TabsTrigger>
            <TabsTrigger value="model">Model Info</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {match && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">
                    {match.home_team.name} vs {match.away_team.name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">
                        {new Date(match.match_date).toLocaleDateString()}
                      </span>
                    </div>
                    <Badge variant="outline">{match.tournament.name}</Badge>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2 text-center">
                      <div className="text-sm text-muted-foreground">Home</div>
                      <div className="text-2xl font-bold text-blue-600">
                        {(prediction.prob_home * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {match.home_team.name}
                      </div>
                    </div>
                    <div className="space-y-2 text-center">
                      <div className="text-sm text-muted-foreground">Draw</div>
                      <div className="text-2xl font-bold text-amber-600">
                        {(prediction.prob_draw * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="space-y-2 text-center">
                      <div className="text-sm text-muted-foreground">Away</div>
                      <div className="text-2xl font-bold text-red-600">
                        {(prediction.prob_away * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {match.away_team.name}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">
                        Predicted Outcome
                      </span>
                      <Badge className={getOutcomeColor(prediction.predicted_outcome)}>
                        {getOutcomeLabel(prediction.predicted_outcome)}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Confidence</span>
                      <ConfidenceIndicator
                        confidence={prediction.confidence}
                        size="lg"
                      />
                    </div>
                    {prediction.actual_roi !== null &&
                      prediction.actual_roi !== undefined && (
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1 text-sm text-muted-foreground">
                            <TrendingUp className="h-3 w-3" />
                            Actual ROI
                          </span>
                          <span
                            className={`font-bold ${
                              prediction.actual_roi >= 0
                                ? "text-green-600"
                                : "text-red-600"
                            }`}
                          >
                            {prediction.actual_roi >= 0 ? "+" : ""}
                            {prediction.actual_roi.toFixed(2)}%
                          </span>
                        </div>
                      )}
                  </div>
                </CardContent>
              </Card>
            )}

            <ValueBetIndicator
              probHome={prediction.prob_home}
              probDraw={prediction.prob_draw}
              probAway={prediction.prob_away}
              oddsHome={match?.odds_home ?? null}
              oddsDraw={match?.odds_draw ?? null}
              oddsAway={match?.odds_away ?? null}
              predictedOutcome={prediction.predicted_outcome}
            />
          </TabsContent>

          <TabsContent value="match" className="space-y-4">
            {match && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Match Information</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Date</span>
                      <span className="text-sm">
                        {new Date(match.match_date).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Status</span>
                      <Badge
                        variant={match.status === "FINISHED" ? "default" : "secondary"}
                      >
                        {match.status}
                      </Badge>
                    </div>
                    {match.status === "FINISHED" && (
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Score</span>
                        <span className="text-sm font-bold">
                          {match.home_score} - {match.away_score}
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Tournament</span>
                      <span className="text-sm">{match.tournament.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Season</span>
                      <span className="text-sm">{match.season.name}</span>
                    </div>
                  </CardContent>
                </Card>

                {match.h2h_matches && match.h2h_matches.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-base">
                        <BarChart3 className="h-4 w-4" />
                        Head-to-Head (Last {Math.min(match.h2h_matches.length, 5)})
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {match.h2h_matches.slice(0, 5).map((h2hMatch) => (
                          <div
                            key={h2hMatch.id}
                            className="flex items-center justify-between text-sm"
                          >
                            <span className="text-muted-foreground">
                              {new Date(h2hMatch.match_date).toLocaleDateString()}
                            </span>
                            <span>
                              {h2hMatch.home_score} - {h2hMatch.away_score}
                            </span>
                            <Badge variant="outline" className="text-xs">
                              {h2hMatch.status}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">
                        {match.home_team.name} Form
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {match.home_team.current_form ? (
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Win Rate</span>
                            <span>
                              {(match.home_team.current_form.win_rate * 100).toFixed(0)}
                              %
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Avg Goals For</span>
                            <span>
                              {match.home_team.current_form.avg_goals_for.toFixed(1)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">
                              Avg Goals Against
                            </span>
                            <span>
                              {match.home_team.current_form.avg_goals_against.toFixed(
                                1
                              )}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">
                          No form data available
                        </p>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">
                        {match.away_team.name} Form
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {match.away_team.current_form ? (
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Win Rate</span>
                            <span>
                              {(match.away_team.current_form.win_rate * 100).toFixed(0)}
                              %
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Avg Goals For</span>
                            <span>
                              {match.away_team.current_form.avg_goals_for.toFixed(1)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">
                              Avg Goals Against
                            </span>
                            <span>
                              {match.away_team.current_form.avg_goals_against.toFixed(
                                1
                              )}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">
                          No form data available
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="model" className="space-y-4">
            {model && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Brain className="h-4 w-4" />
                      {model.version}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Algorithm</span>
                      <Badge variant="outline">{model.algorithm}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Accuracy</span>
                      <span className="text-sm font-medium">
                        {model.accuracy
                          ? `${(model.accuracy * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Created</span>
                      <span className="text-sm">
                        {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    {model.description && (
                      <div className="pt-2">
                        <span className="text-sm text-muted-foreground">
                          Description
                        </span>
                        <p className="mt-1 text-sm">{model.description}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {model.metrics && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Model Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(model.metrics).map(([key, value]) => (
                          <div key={key} className="flex justify-between text-sm">
                            <span className="text-muted-foreground">{key}</span>
                            <span className="font-mono">
                              {typeof value === "number"
                                ? value.toFixed(4)
                                : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </TabsContent>
        </Tabs>

        <div className="flex justify-end gap-2 border-t pt-4">
          {onExport && (
            <Button variant="outline" size="sm" onClick={() => onExport(prediction)}>
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          )}
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
