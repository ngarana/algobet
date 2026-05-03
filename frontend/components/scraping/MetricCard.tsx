"use client";

import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  valueColor?: string;
}

export function MetricCard({
  label,
  value,
  icon,
  valueColor = "#e0e6f0",
}: MetricCardProps) {
  return (
    <Card className="border-[#252a37] bg-[#12151d] transition-colors hover:bg-[#161a25]">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <p className="text-xs font-medium uppercase tracking-wider text-[#9ca3af]">
              {label}
            </p>
            <p className="text-2xl font-bold" style={{ color: valueColor }}>
              {value}
            </p>
          </div>
          {icon}
        </div>
      </CardContent>
    </Card>
  );
}
