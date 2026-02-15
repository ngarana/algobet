"use client";

import { useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCircleIcon, RefreshCwIcon } from "lucide-react";

export default function SchedulesError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Schedules page error:", error);
  }, [error]);

  return (
    <div className="container mx-auto py-10">
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertCircleIcon className="h-5 w-5" />
            Error Loading Schedules
          </CardTitle>
          <CardDescription>
            There was a problem loading the scheduled tasks.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">{error.message}</p>
          <Button onClick={reset} variant="outline">
            <RefreshCwIcon className="mr-2 h-4 w-4" />
            Try Again
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
