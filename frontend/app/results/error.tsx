"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function ResultsError({ reset }: { reset: () => void }) {
  return (
    <Card className="border-destructive">
      <CardContent className="space-y-4 p-6">
        <p className="text-destructive">Failed to load results review.</p>
        <Button variant="outline" onClick={reset}>
          Try again
        </Button>
      </CardContent>
    </Card>
  );
}
