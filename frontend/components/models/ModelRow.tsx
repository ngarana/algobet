"use client";

import { BarChart3, Check, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TableCell, TableRow } from "@/components/ui/table";
import { ModelMetricsPanel } from "./ModelMetricsPanel";
import type { ModelRowProps } from "./types";

export function ModelRow({
  model,
  isActive,
  isExpanded,
  onActivate,
  onDelete,
  onToggleMetrics,
}: ModelRowProps) {
  return (
    <>
      <TableRow className={isActive ? "bg-green-50/70 dark:bg-green-950/20" : ""}>
        <TableCell className="font-mono text-xs">{model.version}</TableCell>
        <TableCell>
          <Badge variant="outline">{model.algorithm}</Badge>
        </TableCell>
        <TableCell className="font-mono">
          {model.accuracy !== null ? `${(model.accuracy * 100).toFixed(1)}%` : "-"}
        </TableCell>
        <TableCell>
          {isActive ? (
            <Badge className="bg-green-600">Active</Badge>
          ) : (
            <Badge variant="secondary">Inactive</Badge>
          )}
        </TableCell>
        <TableCell className="text-sm text-muted-foreground">
          {new Date(model.created_at).toLocaleString()}
        </TableCell>
        <TableCell className="max-w-64 text-sm text-muted-foreground">
          {model.description ?? "-"}
        </TableCell>
        <TableCell>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onToggleMetrics(isExpanded ? null : model)}
            >
              <BarChart3 className="mr-1 h-4 w-4" />
              {isExpanded ? "Hide" : "Metrics"}
            </Button>

            {!isActive && (
              <Button variant="outline" size="sm" onClick={() => onActivate(model.id)}>
                <Check className="mr-1 h-4 w-4" />
                Activate
              </Button>
            )}

            {!isActive && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => {
                  if (confirm(`Delete model ${model.version}?`)) {
                    onDelete(model.id);
                  }
                }}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </TableCell>
      </TableRow>

      {isExpanded && (
        <TableRow>
          <TableCell className="p-0" colSpan={7}>
            <ModelMetricsPanel model={model} onClose={() => onToggleMetrics(null)} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
