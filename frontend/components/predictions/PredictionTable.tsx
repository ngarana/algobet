import { ArrowUpDown } from "lucide-react";
import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import type { Prediction } from "@/lib/types/api";
import PredictionRow from "./PredictionRow";

interface PredictionTableProps {
  predictions: Prediction[];
  originalPredictions: Prediction[];
  filteredPredictions: Prediction[];
  showRoi: boolean;
  isLoading: boolean;
  title: string;
  description: string;
  sortConfig?: {
    key: keyof Prediction | "match" | "probabilities";
    direction: "asc" | "desc";
  } | null;
  onSort?: (key: keyof Prediction | "match" | "probabilities") => void;
  onRefresh?: () => void;
}

function SortableHeader({
  label,
  sortKey,
  sortConfig,
  onSort,
  className,
}: {
  label: string;
  sortKey: keyof Prediction | "match" | "probabilities";
  sortConfig: PredictionTableProps["sortConfig"];
  onSort?: (key: keyof Prediction | "match" | "probabilities") => void;
  className?: string;
}) {
  if (!onSort) {
    return <TableHead className={className}>{label}</TableHead>;
  }

  const isActive = sortConfig?.key === sortKey;
  const direction = sortConfig?.direction;

  return (
    <TableHead className={className}>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onSort(sortKey)}
        className="-ml-3 h-8 font-medium hover:bg-transparent"
      >
        {label}
        <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
        {isActive && <span className="ml-1">{direction === "asc" ? "▲" : "▼"}</span>}
      </Button>
    </TableHead>
  );
}

export default function PredictionTable({
  predictions,
  originalPredictions,
  filteredPredictions,
  showRoi,
  isLoading,
  title,
  description,
  sortConfig,
  onSort,
}: PredictionTableProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-4 md:grid-cols-4">
          {[...Array(4)].map((_, index) => (
            <Skeleton key={index} className="h-20" />
          ))}
        </div>
        <Skeleton className="h-72" />
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="rounded-lg border p-12 text-center text-muted-foreground">
        <p className="text-lg font-medium">No predictions found</p>
        <p className="text-sm">
          Generate predictions for the selected model to populate this view.
        </p>
      </div>
    );
  }

  const filteredCount = filteredPredictions.length;
  const originalCount = originalPredictions.length;
  const showingFiltered = filteredCount !== originalCount;

  return (
    <div className="rounded-lg border">
      <div className="border-b p-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
          {showingFiltered && (
            <Badge variant="secondary" className="self-start sm:self-center">
              Showing {filteredCount} of {originalCount}
            </Badge>
          )}
        </div>
      </div>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <SortableHeader
                label="Match"
                sortKey="match"
                sortConfig={sortConfig}
                onSort={onSort}
              />
              <SortableHeader
                label="Prediction"
                sortKey="predicted_outcome"
                sortConfig={sortConfig}
                onSort={onSort}
              />
              <SortableHeader
                label="Probabilities"
                sortKey="probabilities"
                sortConfig={sortConfig}
                onSort={onSort}
              />
              <SortableHeader
                label="Confidence"
                sortKey="confidence"
                sortConfig={sortConfig}
                onSort={onSort}
              />
              <TableHead>Timing</TableHead>
              {showRoi && (
                <SortableHeader
                  label="ROI"
                  sortKey="actual_roi"
                  sortConfig={sortConfig}
                  onSort={onSort}
                />
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {predictions.map((prediction) => (
              <PredictionRow
                key={prediction.id}
                prediction={prediction}
                showRoi={showRoi}
              />
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
