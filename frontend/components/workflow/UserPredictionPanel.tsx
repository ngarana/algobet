"use client";

import { useEffect, useState } from "react";
import { Send } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useUpsertUserPrediction } from "@/lib/queries/use-workflow";
import type { PredictedOutcome, UserPrediction } from "@/lib/types/api";

const outcomeLabels: Record<PredictedOutcome, string> = {
  H: "Home win",
  D: "Draw",
  A: "Away win",
};

export function UserPredictionPanel({
  matchId,
  userPrediction,
}: {
  matchId: number;
  userPrediction?: UserPrediction | null;
}) {
  const [pick, setPick] = useState<PredictedOutcome | "">("");
  const [homeScore, setHomeScore] = useState("");
  const [awayScore, setAwayScore] = useState("");
  const mutation = useUpsertUserPrediction();

  useEffect(() => {
    setPick(userPrediction?.pick_1x2 ?? "");
    setHomeScore(
      userPrediction?.home_score !== null && userPrediction?.home_score !== undefined
        ? String(userPrediction.home_score)
        : ""
    );
    setAwayScore(
      userPrediction?.away_score !== null && userPrediction?.away_score !== undefined
        ? String(userPrediction.away_score)
        : ""
    );
  }, [userPrediction]);

  const handleSubmit = () => {
    mutation.mutate({
      match_id: matchId,
      pick_1x2: pick || null,
      home_score: homeScore ? Number(homeScore) : null,
      away_score: awayScore ? Number(awayScore) : null,
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Make Your Own Prediction</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <Label>1X2 pick</Label>
            <Select
              value={pick}
              onValueChange={(value) => setPick(value as PredictedOutcome)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select outcome" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="H">Home win</SelectItem>
                <SelectItem value="D">Draw</SelectItem>
                <SelectItem value="A">Away win</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="home-score">Home score</Label>
            <Input
              id="home-score"
              type="number"
              min="0"
              value={homeScore}
              onChange={(event) => setHomeScore(event.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="away-score">Away score</Label>
            <Input
              id="away-score"
              type="number"
              min="0"
              value={awayScore}
              onChange={(event) => setAwayScore(event.target.value)}
            />
          </div>
        </div>

        {userPrediction && (
          <div className="flex flex-wrap gap-2">
            {userPrediction.pick_1x2 && (
              <Badge variant="secondary">
                Your pick: {outcomeLabels[userPrediction.pick_1x2]}
              </Badge>
            )}
            {userPrediction.model_prediction && (
              <Badge variant="outline">
                Model:{" "}
                {outcomeLabels[userPrediction.model_prediction.predicted_outcome]}
              </Badge>
            )}
            {userPrediction.is_correct_1x2 !== null && (
              <Badge variant={userPrediction.is_correct_1x2 ? "success" : "secondary"}>
                {userPrediction.points} points
              </Badge>
            )}
          </div>
        )}

        <Button onClick={handleSubmit} disabled={mutation.isPending || !pick}>
          <Send className="mr-2 h-4 w-4" />
          {mutation.isPending ? "Saving..." : "Save prediction"}
        </Button>
      </CardContent>
    </Card>
  );
}
