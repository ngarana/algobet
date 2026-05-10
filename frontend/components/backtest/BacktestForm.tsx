"use client";

import { useState, type FormEvent } from "react";
import { Play, Target } from "lucide-react";

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
import { useActiveModel, useModels } from "@/lib/queries/use-models";

interface BacktestFormProps {
  onSubmit: (data: { startDate: string; endDate: string; minMatches: number }) => void;
  isLoading: boolean;
}

export function BacktestForm({ onSubmit, isLoading }: BacktestFormProps) {
  const { data: activeModel } = useActiveModel();
  useModels();

  const today = new Date();
  const oneYearAgo = new Date(today);
  oneYearAgo.setFullYear(today.getFullYear() - 1);

  const [startDate, setStartDate] = useState(oneYearAgo.toISOString().split("T")[0]);
  const [endDate, setEndDate] = useState(today.toISOString().split("T")[0]);
  const [minMatches, setMinMatches] = useState(100);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit({ startDate, endDate, minMatches });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Run Backtest
        </CardTitle>
        <CardDescription>
          Evaluate model performance on historical match data
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
            <Target className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">
              Active Model:{" "}
              {activeModel ? (
                <span className="font-medium">
                  {activeModel.name} ({activeModel.algorithm})
                </span>
              ) : (
                <span className="text-destructive">No active model</span>
              )}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start-date">Start Date</Label>
              <Input
                id="start-date"
                type="date"
                value={startDate}
                onChange={(event) => setStartDate(event.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="end-date">End Date</Label>
              <Input
                id="end-date"
                type="date"
                value={endDate}
                onChange={(event) => setEndDate(event.target.value)}
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="min-matches">Minimum Matches</Label>
            <Input
              id="min-matches"
              type="number"
              min={10}
              max={10000}
              value={minMatches}
              onChange={(event) => setMinMatches(parseInt(event.target.value) || 100)}
            />
          </div>

          <Button type="submit" className="w-full" disabled={isLoading || !activeModel}>
            {isLoading ? (
              <>
                <span className="mr-2 animate-spin">...</span>
                Running Backtest...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Backtest
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
