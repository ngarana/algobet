"use client";

import { PlayIcon } from "lucide-react";

import { Button } from "@/components/ui/button";

interface FetchConfirmationFooterProps {
  color: string;
  disabled: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function FetchConfirmationFooter({
  color,
  disabled,
  onCancel,
  onConfirm,
}: FetchConfirmationFooterProps) {
  return (
    <div className="flex gap-2 pt-2">
      <Button
        onClick={onConfirm}
        disabled={disabled}
        className="font-semibold text-[#0a0c12]"
        style={{ backgroundColor: color }}
      >
        <PlayIcon className="mr-2 h-4 w-4" />
        Start Fetch
      </Button>
      <Button
        variant="outline"
        onClick={onCancel}
        className="border-[#252a37] text-[#9ca3af]"
      >
        Cancel
      </Button>
    </div>
  );
}
