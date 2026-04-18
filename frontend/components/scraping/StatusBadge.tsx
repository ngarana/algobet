"use client";

import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  status: string;
}

const statusConfig = {
  completed: {
    color: "text-[#4ade80]",
    bg: "bg-[#4ade80]/10",
    label: "SUCCESS",
    dot: "bg-[#4ade80]",
  },
  running: {
    color: "text-[#38bdf8]",
    bg: "bg-[#38bdf8]/10",
    label: "RUNNING",
    dot: "bg-[#38bdf8]",
  },
  pending: {
    color: "text-[#38bdf8]",
    bg: "bg-[#38bdf8]/10",
    label: "PENDING",
    dot: "bg-[#38bdf8]",
  },
  failed: {
    color: "text-[#f87171]",
    bg: "bg-[#f87171]/10",
    label: "FAILED",
    dot: "bg-[#f87171]",
  },
  cancelled: {
    color: "text-[#9ca3af]",
    bg: "bg-[#9ca3af]/10",
    label: "CANCELLED",
    dot: "bg-[#9ca3af]",
  },
};

export function StatusBadge({ status }: StatusBadgeProps) {
  const config =
    statusConfig[status as keyof typeof statusConfig] || statusConfig.pending;
  const { color, bg, label, dot } = config;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs font-medium",
        bg,
        color
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", dot)} />
      {label}
    </span>
  );
}
