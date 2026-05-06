"use client";

import { Star } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  useAddWatchlistEntry,
  useRemoveWatchlistEntry,
} from "@/lib/queries/use-workflow";
import type { WatchlistEntryType } from "@/lib/types/api";

export function WatchlistToggle({
  entryType,
  entryId,
  watched,
  label,
}: {
  entryType: WatchlistEntryType;
  entryId: number;
  watched: boolean;
  label?: string;
}) {
  const addMutation = useAddWatchlistEntry();
  const removeMutation = useRemoveWatchlistEntry();
  const isPending = addMutation.isPending || removeMutation.isPending;

  const handleClick = () => {
    if (watched) {
      removeMutation.mutate({ entryType, entryId });
      return;
    }
    addMutation.mutate({ entry_type: entryType, entry_id: entryId });
  };

  return (
    <Button
      type="button"
      variant={watched ? "default" : "outline"}
      size="sm"
      onClick={handleClick}
      disabled={isPending}
    >
      <Star className="mr-2 h-4 w-4" />
      {label ?? (watched ? "Watching" : "Watch")}
    </Button>
  );
}
