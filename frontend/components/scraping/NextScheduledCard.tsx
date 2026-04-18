"use client";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PlayIcon, RefreshCwIcon } from "lucide-react";

interface NextScheduledCardProps {
  taskName?: string;
  eta?: string;
  onStart?: () => void;
  isLoading?: boolean;
}

export function NextScheduledCard({
  taskName = "Daily Sync",
  eta = "04:20 MIN",
  onStart,
  isLoading = false,
}: NextScheduledCardProps) {
  return (
    <Card className="border-[#252a37] bg-[#12151d]">
      <div className="p-4">
        <p className="mb-3 text-xs uppercase tracking-wider text-[#9ca3af]">
          NEXT SCHEDULED
        </p>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#252a37]">
              <RefreshCwIcon className="h-4 w-4 text-[#38bdf8]" />
            </div>
            <div>
              <p className="text-sm font-medium text-[#e0e6f0]">{taskName}</p>
              <p className="text-xs text-[#9ca3af]">ETA: {eta}</p>
            </div>
          </div>
          {onStart && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-[#4ade80] hover:bg-[#252a37] hover:text-[#22c55e]"
              onClick={onStart}
              disabled={isLoading}
            >
              <PlayIcon className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
