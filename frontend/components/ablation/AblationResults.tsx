"use client";

import { ArrowDown, ArrowUp, Layers } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { AblationStudyResponse } from "@/lib/types/ablation";

interface AblationResultsProps {
  result: AblationStudyResponse;
}

export function AblationResults({ result }: AblationResultsProps) {
  const sortedFamilies = [...result.families].sort(
    (a, b) => b.log_loss_delta - a.log_loss_delta
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Leave-One-Out Ablation Results</h2>
          <p className="text-sm text-muted-foreground">
            Baseline: {result.baseline_model_version} ({result.baseline_num_features}{" "}
            features)
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="text-2xl font-bold">
              {(result.baseline_test_metrics.accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">Baseline Test Accuracy</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="text-2xl font-bold">
              {result.baseline_test_metrics.log_loss.toFixed(4)}
            </div>
            <div className="text-sm text-muted-foreground">Baseline Test Log Loss</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Feature Group Impact
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {sortedFamilies.map((family) => {
              const maxLogLossDelta = Math.max(
                ...result.families.map((f) => Math.abs(f.log_loss_delta)),
                0.001
              );
              const barWidth =
                (Math.abs(family.log_loss_delta) / maxLogLossDelta) * 100;
              const isPositive = family.log_loss_delta > 0;
              return (
                <div key={family.family} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{family.family}</span>
                      <span className="text-xs text-muted-foreground">
                        ({family.features_excluded.length} excluded,{" "}
                        {family.num_features_used} remaining)
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {isPositive ? (
                        <ArrowUp className="h-3 w-3 text-red-500" />
                      ) : (
                        <ArrowDown className="h-3 w-3 text-green-500" />
                      )}
                      <span
                        className={`font-mono text-sm ${
                          family.log_loss_delta > 0 ? "text-red-600" : "text-green-600"
                        }`}
                      >
                        {family.log_loss_delta > 0 ? "+" : ""}
                        {family.log_loss_delta.toFixed(4)} LL
                      </span>
                      <span
                        className={`font-mono text-sm ${
                          family.accuracy_delta > 0 ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {family.accuracy_delta > 0 ? "+" : ""}
                        {(family.accuracy_delta * 100).toFixed(1)}% acc
                      </span>
                    </div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div
                      className={`h-2 rounded-full ${
                        isPositive ? "bg-red-500" : "bg-green-500"
                      }`}
                      style={{
                        width: `${Math.max(barWidth, 2)}%`,
                      }}
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
          <CardTitle className="text-lg">Detailed Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="py-2 text-left">Group Excluded</th>
                  <th className="py-2 text-right">Features Used</th>
                  <th className="py-2 text-right">&Delta; Log Loss</th>
                  <th className="py-2 text-right">&Delta; Accuracy</th>
                  <th className="py-2 text-right">Test LL</th>
                  <th className="py-2 text-right">Test Accuracy</th>
                  <th className="py-2 text-right">Model Version</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b bg-muted/50">
                  <td className="py-2 font-medium">Baseline (all features)</td>
                  <td className="py-2 text-right">{result.baseline_num_features}</td>
                  <td className="py-2 text-right font-mono">-</td>
                  <td className="py-2 text-right font-mono">-</td>
                  <td className="py-2 text-right font-mono">
                    {result.baseline_test_metrics.log_loss.toFixed(4)}
                  </td>
                  <td className="py-2 text-right font-mono">
                    {(result.baseline_test_metrics.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 text-right font-mono text-xs">
                    {result.baseline_model_version}
                  </td>
                </tr>
                {result.families.map((family) => (
                  <tr key={family.family} className="border-b">
                    <td className="py-2 font-medium">{family.family}</td>
                    <td className="py-2 text-right text-muted-foreground">
                      {family.num_features_used}
                    </td>
                    <td className="py-2 text-right font-mono">
                      <span
                        className={
                          family.log_loss_delta > 0 ? "text-red-600" : "text-green-600"
                        }
                      >
                        {family.log_loss_delta > 0 ? "+" : ""}
                        {family.log_loss_delta.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2 text-right font-mono">
                      <span
                        className={
                          family.accuracy_delta > 0 ? "text-green-600" : "text-red-600"
                        }
                      >
                        {family.accuracy_delta > 0 ? "+" : ""}
                        {(family.accuracy_delta * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 text-right font-mono">
                      {family.test_metrics.log_loss.toFixed(4)}
                    </td>
                    <td className="py-2 text-right font-mono">
                      {(family.test_metrics.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 text-right font-mono text-xs">
                      {family.model_version}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
