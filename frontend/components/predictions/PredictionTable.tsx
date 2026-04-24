import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import type { Prediction } from "@/lib/types/api";
import PredictionRow from "./PredictionRow";

interface PredictionTableProps {
  predictions: Prediction[];
  showRoi: boolean;
  isLoading: boolean;
  title: string;
  description: string;
}

export default function PredictionTable({
  predictions,
  showRoi,
  isLoading,
  title,
  description,
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

  return (
    <div className="rounded-lg border">
      <div className="border-b p-6">
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Match</TableHead>
              <TableHead>Prediction</TableHead>
              <TableHead>Probabilities</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead>Timing</TableHead>
              {showRoi && <TableHead>ROI</TableHead>}
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
