"use client";

import { BarChart3 } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { PermutationImportanceResponse } from "@/lib/types/ablation";

interface PermutationResultsProps {
  result: PermutationImportanceResponse;
}

export function PermutationResults({ result }: PermutationResultsProps) {
  const sortedFamilies = [...result.families].sort(
    (a, b) => a.importance_rank - b.importance_rank
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Permutation Importance Results</h2>
          <p className="text-sm text-muted-foreground">
            {result.num_samples.toLocaleString()} samples &middot; {result.n_repeats}{" "}
            repeats &middot; Model: {result.model_version}
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="text-2xl font-bold">
              {(result.baseline_accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">Baseline Accuracy</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="text-2xl font-bold">
              {result.baseline_log_loss.toFixed(4)}
            </div>
            <div className="text-sm text-muted-foreground">Baseline Log Loss</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="text-2xl font-bold">{result.families.length}</div>
            <div className="text-sm text-muted-foreground">
              Feature Families Evaluated
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Family Importance Ranking
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {sortedFamilies.map((family) => {
              const barWidth = Math.max((family.importance_score || 0) * 100, 2);
              const isPositive = family.log_loss_increase > 0;
              return (
                <div key={family.family} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-bold">
                        {family.importance_rank}
                      </span>
                      <span className="font-medium">{family.family}</span>
                      <span className="text-xs text-muted-foreground">
                        ({family.features_found.length} feature
                        {family.features_found.length !== 1 ? "s" : ""})
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="text-muted-foreground">
                        &Delta;LL: {isPositive ? "+" : ""}
                        {family.log_loss_increase.toFixed(4)}
                      </span>
                      <span className="text-muted-foreground">
                        &Delta;Acc: {isPositive ? "-" : ""}
                        {(family.accuracy_decrease * 100).toFixed(1)}%
                      </span>
                      <span
                        className={`flex items-center gap-0.5 font-mono ${
                          family.importance_score >= 0.15
                            ? "text-green-600"
                            : family.importance_score >= 0.05
                              ? "text-yellow-600"
                              : "text-muted-foreground"
                        }`}
                      >
                        {(family.importance_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div
                      className={`h-2 rounded-full ${
                        family.importance_score >= 0.15
                          ? "bg-green-500"
                          : family.importance_score >= 0.05
                            ? "bg-yellow-500"
                            : "bg-gray-400"
                      }`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Detailed Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="py-2 text-left">Family</th>
                  <th className="py-2 text-right">Features</th>
                  <th className="py-2 text-right">Baseline LL</th>
                  <th className="py-2 text-right">Permuted LL</th>
                  <th className="py-2 text-right">&Delta; Log Loss</th>
                  <th className="py-2 text-right">&Delta; Accuracy</th>
                  <th className="py-2 text-right">Importance</th>
                </tr>
              </thead>
              <tbody>
                {sortedFamilies.map((family) => (
                  <tr key={family.family} className="border-b">
                    <td className="py-2 font-medium">{family.family}</td>
                    <td className="py-2 text-right text-muted-foreground">
                      {family.features_found.length}
                    </td>
                    <td className="py-2 text-right font-mono">
                      {family.baseline_log_loss.toFixed(4)}
                    </td>
                    <td className="py-2 text-right font-mono">
                      {family.permuted_log_loss.toFixed(4)}
                    </td>
                    <td className="py-2 text-right font-mono">
                      <span
                        className={`${
                          family.log_loss_increase > 0
                            ? "text-red-600"
                            : "text-green-600"
                        }`}
                      >
                        {family.log_loss_increase > 0 ? "+" : ""}
                        {family.log_loss_increase.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2 text-right font-mono">
                      <span
                        className={`${
                          family.accuracy_decrease > 0
                            ? "text-red-600"
                            : "text-green-600"
                        }`}
                      >
                        {family.accuracy_decrease > 0 ? "+" : ""}
                        {(family.accuracy_decrease * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 text-right font-mono">
                      {(family.importance_score * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {result.raw_feature_importance &&
        Object.keys(result.raw_feature_importance).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                Raw Feature Importance (Model-Native)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 text-sm">
                {Object.entries(result.raw_feature_importance)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 20)
                  .map(([name, score]) => (
                    <div key={name} className="flex items-center justify-between">
                      <span className="truncate font-mono text-xs">{name}</span>
                      <span className="font-mono text-xs">{score.toFixed(4)}</span>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        )}
    </div>
  );
}
