"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function WatchlistError({ reset }: { reset: () => void }) {
  return (
    <Card className="border-destructive">
      <CardContent className="space-y-4 p-6">
        <p className="text-destructive">Failed to load watchlist.</p>
        <Button variant="outline" onClick={reset}>
          Try again
        </Button>
      </CardContent>
    </Card>
  );
}
