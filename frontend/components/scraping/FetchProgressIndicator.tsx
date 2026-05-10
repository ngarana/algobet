"use client";

import { Loader2 } from "lucide-react";

interface FetchProgressIndicatorProps {
  isLoading: boolean;
}

export function FetchProgressIndicator({ isLoading }: FetchProgressIndicatorProps) {
  if (!isLoading) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-[#252a37] bg-[#161a25] px-3 py-2 text-sm text-[#9ca3af]">
      <Loader2 className="h-4 w-4 animate-spin" />
      Fetch request is starting
    </div>
  );
}
