import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Download, FileJson, FileSpreadsheet } from "lucide-react";
import type { Prediction } from "@/lib/types/api";

interface ExportButtonProps {
  predictions: Prediction[];
  disabled?: boolean;
}

const AVAILABLE_FIELDS = [
  { id: "id", label: "ID" },
  { id: "match", label: "Match" },
  { id: "predicted_outcome", label: "Predicted Outcome" },
  { id: "prob_home", label: "Home Probability" },
  { id: "prob_draw", label: "Draw Probability" },
  { id: "prob_away", label: "Away Probability" },
  { id: "confidence", label: "Confidence" },
  { id: "max_probability", label: "Max Probability" },
  { id: "actual_roi", label: "Actual ROI" },
  { id: "predicted_at", label: "Predicted At" },
  { id: "model_version_id", label: "Model Version ID" },
];

export default function ExportButton({
  predictions,
  disabled = false,
}: ExportButtonProps) {
  const [open, setOpen] = useState(false);
  const [format, setFormat] = useState<"csv" | "json">("csv");
  const [selectedFields, setSelectedFields] = useState<string[]>(
    AVAILABLE_FIELDS.filter((f) =>
      ["id", "match", "predicted_outcome", "confidence", "actual_roi"].includes(f.id)
    ).map((f) => f.id)
  );

  const handleExport = () => {
    if (format === "csv") {
      exportToCsv();
    } else {
      exportToJson();
    }
    setOpen(false);
  };

  const exportToCsv = () => {
    const headers = AVAILABLE_FIELDS.filter((f) => selectedFields.includes(f.id)).map(
      (f) => f.label
    );

    const rows = predictions.map((p) => {
      const match = p.match as
        | {
            home_team_name?: string;
            away_team_name?: string;
            match_date?: string;
          }
        | null
        | undefined;

      return AVAILABLE_FIELDS.filter((f) => selectedFields.includes(f.id)).map((f) => {
        switch (f.id) {
          case "id":
            return p.id;
          case "match":
            return match ? `${match.home_team_name} vs ${match.away_team_name}` : "N/A";
          case "predicted_outcome":
            return p.predicted_outcome;
          case "prob_home":
            return p.prob_home;
          case "prob_draw":
            return p.prob_draw;
          case "prob_away":
            return p.prob_away;
          case "confidence":
            return p.confidence;
          case "max_probability":
            return p.max_probability;
          case "actual_roi":
            return p.actual_roi ?? "";
          case "predicted_at":
            return p.predicted_at;
          case "model_version_id":
            return p.model_version_id;
          default:
            return "";
        }
      });
    });

    const csvContent = [
      headers.join(","),
      ...rows.map((row) =>
        row
          .map((cell) => {
            if (typeof cell === "string" && cell.includes(",")) {
              return `"${cell}"`;
            }
            return cell;
          })
          .join(",")
      ),
    ].join("\n");

    downloadFile(csvContent, "predictions.csv", "text/csv");
  };

  const exportToJson = () => {
    const data = predictions.map((p) => {
      const match = p.match as
        | {
            home_team_name?: string;
            away_team_name?: string;
          }
        | null
        | undefined;

      const result: Record<string, unknown> = {};

      if (selectedFields.includes("id")) result.id = p.id;
      if (selectedFields.includes("match")) {
        result.match = match
          ? `${match.home_team_name} vs ${match.away_team_name}`
          : "N/A";
      }
      if (selectedFields.includes("predicted_outcome")) {
        result.predicted_outcome = p.predicted_outcome;
      }
      if (selectedFields.includes("prob_home")) result.prob_home = p.prob_home;
      if (selectedFields.includes("prob_draw")) result.prob_draw = p.prob_draw;
      if (selectedFields.includes("prob_away")) result.prob_away = p.prob_away;
      if (selectedFields.includes("confidence")) result.confidence = p.confidence;
      if (selectedFields.includes("max_probability")) {
        result.max_probability = p.max_probability;
      }
      if (selectedFields.includes("actual_roi")) result.actual_roi = p.actual_roi;
      if (selectedFields.includes("predicted_at")) result.predicted_at = p.predicted_at;
      if (selectedFields.includes("model_version_id")) {
        result.model_version_id = p.model_version_id;
      }

      return result;
    });

    const jsonContent = JSON.stringify(data, null, 2);
    downloadFile(jsonContent, "predictions.json", "application/json");
  };

  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const toggleField = (fieldId: string) => {
    setSelectedFields((prev) =>
      prev.includes(fieldId) ? prev.filter((f) => f !== fieldId) : [...prev, fieldId]
    );
  };

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        disabled={disabled || predictions.length === 0}
      >
        <Download className="mr-2 h-4 w-4" />
        Export ({predictions.length})
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export Predictions</DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">Format</Label>
              <div className="flex gap-2">
                <Button
                  variant={format === "csv" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFormat("csv")}
                >
                  <FileSpreadsheet className="mr-2 h-4 w-4" />
                  CSV
                </Button>
                <Button
                  variant={format === "json" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFormat("json")}
                >
                  <FileJson className="mr-2 h-4 w-4" />
                  JSON
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium">Fields to Export</Label>
              <div className="grid grid-cols-2 gap-2">
                {AVAILABLE_FIELDS.map((field) => (
                  <div key={field.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={field.id}
                      checked={selectedFields.includes(field.id)}
                      onCheckedChange={() => toggleField(field.id)}
                    />
                    <Label
                      htmlFor={field.id}
                      className="cursor-pointer text-sm font-normal"
                    >
                      {field.label}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              Exporting {predictions.length} prediction(s) with {selectedFields.length}{" "}
              field(s)
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={selectedFields.length === 0}>
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
