"use client";

import { useCallback, useState } from "react";
import { AlertCircle, FlaskConical, BarChart3 } from "lucide-react";

import {
  AblationForm,
  PermutationResults,
  AblationResults,
} from "@/components/ablation";
import { Card, CardContent } from "@/components/ui/card";
import { useAblation } from "@/lib/queries/use-ablation";
import type {
  AblationRequest,
  AblationResponse,
  PermutationImportanceResponse,
  AblationStudyResponse,
} from "@/lib/types/ablation";

export default function AblationPage() {
  const { mutate: runAblation, isPending, error } = useAblation();
  const [result, setResult] = useState<AblationResponse | null>(null);
  const [submittedMethod, setSubmittedMethod] = useState<
    AblationRequest["method"] | null
  >(null);

  const handleSubmit = useCallback(
    (request: Parameters<typeof runAblation>[0]) => {
      setSubmittedMethod(request.method);
      runAblation(request, {
        onSuccess: (data) => setResult(data),
      });
    },
    [runAblation]
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Feature Importance</h1>
        <p className="text-muted-foreground">
          Analyze which feature families contribute most to model predictions
        </p>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 p-4 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <p>{error.message || "An error occurred"}</p>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-1">
          <AblationForm onSubmit={handleSubmit} isLoading={isPending} />
        </div>

        <div className="space-y-6 lg:col-span-2">
          {isPending && !result ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <FlaskConical className="mb-4 h-12 w-12 animate-pulse" />
                <p className="text-lg font-medium">Running analysis...</p>
                <p className="text-sm">
                  {submittedMethod === "ablation"
                    ? "This may take a few minutes as each feature group is excluded and the model is retrained"
                    : "Shuffling features and measuring performance drop"}
                </p>
              </CardContent>
            </Card>
          ) : result ? (
            result.method === "permutation" ? (
              <PermutationResults result={result as PermutationImportanceResponse} />
            ) : (
              <AblationResults result={result as AblationStudyResponse} />
            )
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <BarChart3 className="mb-4 h-12 w-12" />
                <p className="text-lg font-medium">No results yet</p>
                <p className="text-sm">
                  Configure and run an analysis to see feature importance
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
