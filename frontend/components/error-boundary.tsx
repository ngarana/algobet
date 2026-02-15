"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle } from "lucide-react";

interface ErrorBoundaryProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export function ErrorBoundary({ error, reset }: ErrorBoundaryProps) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Error caught by boundary:", error);
  }, [error]);

  return (
    <div className="flex min-h-[400px] items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10">
            <AlertTriangle className="h-6 w-6 text-destructive" />
          </div>
          <CardTitle className="text-xl">Something went wrong</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-center text-muted-foreground">
            {error.message || "An unexpected error occurred"}
          </p>

          {error.digest && (
            <p className="text-center text-xs text-muted-foreground">
              Error ID: {error.digest}
            </p>
          )}

          <div className="flex justify-center gap-2">
            <Button onClick={reset} variant="default">
              Try again
            </Button>
            <Button onClick={() => (window.location.href = "/")} variant="outline">
              Go home
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Default export for Next.js error.tsx files
export default ErrorBoundary;
