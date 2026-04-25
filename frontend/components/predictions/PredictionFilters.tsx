import { useState } from "react";
import { Search, SlidersHorizontal, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface PredictionFilterState {
  search?: string;
  outcome?: "H" | "D" | "A";
  minConfidence?: number;
  fromDate?: string;
  toDate?: string;
  modelVersionId?: number;
  onlyValueBets?: boolean;
}

interface PredictionFiltersProps {
  filters: PredictionFilterState;
  onFiltersChange: (filters: PredictionFilterState) => void;
  availableModels?: Array<{ id: number; version: string }>;
  activeFilterCount: number;
}

export default function PredictionFilters({
  filters,
  onFiltersChange,
  availableModels = [],
  activeFilterCount,
}: PredictionFiltersProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const updateFilter = (
    key: keyof PredictionFilterState,
    value: string | number | boolean | undefined
  ) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  const clearFilters = () => {
    onFiltersChange({});
  };

  return (
    <div className="space-y-3">
      {/* Quick search and toggle */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search teams..."
            value={filters.search ?? ""}
            onChange={(e) => updateFilter("search", e.target.value || undefined)}
            className="pl-8"
          />
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="gap-2"
        >
          <SlidersHorizontal className="h-4 w-4" />
          Filters
        </Button>
        {activeFilterCount > 0 && (
          <Button variant="ghost" size="sm" onClick={clearFilters}>
            <X className="h-4 w-4" />
            Clear
          </Button>
        )}
      </div>

      {/* Expanded filters */}
      {isExpanded && (
        <div className="grid gap-3 rounded-lg border p-4 md:grid-cols-4">
          <div className="space-y-2">
            <Label>Outcome</Label>
            <Select
              value={filters.outcome ?? "all"}
              onValueChange={(value) =>
                updateFilter(
                  "outcome",
                  value === "all" ? undefined : (value as "H" | "D" | "A")
                )
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="All outcomes" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All outcomes</SelectItem>
                <SelectItem value="H">Home Win</SelectItem>
                <SelectItem value="D">Draw</SelectItem>
                <SelectItem value="A">Away Win</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Min Confidence</Label>
            <Select
              value={String(filters.minConfidence ?? 0)}
              onValueChange={(value) =>
                updateFilter("minConfidence", Number(value) || undefined)
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="Any confidence" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">Any confidence</SelectItem>
                <SelectItem value="0.5">50%+</SelectItem>
                <SelectItem value="0.6">60%+</SelectItem>
                <SelectItem value="0.7">70%+</SelectItem>
                <SelectItem value="0.8">80%+</SelectItem>
                <SelectItem value="0.9">90%+</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>From Date</Label>
            <Input
              type="date"
              value={filters.fromDate ?? ""}
              onChange={(e) => updateFilter("fromDate", e.target.value || undefined)}
            />
          </div>

          <div className="space-y-2">
            <Label>To Date</Label>
            <Input
              type="date"
              value={filters.toDate ?? ""}
              onChange={(e) => updateFilter("toDate", e.target.value || undefined)}
            />
          </div>

          <div className="space-y-2">
            <Label>Model Version</Label>
            <Select
              value={filters.modelVersionId?.toString() ?? "all"}
              onValueChange={(value) =>
                updateFilter(
                  "modelVersionId",
                  value === "all" ? undefined : Number(value)
                )
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="All models" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All models</SelectItem>
                {availableModels.map((model) => (
                  <SelectItem key={model.id} value={String(model.id)}>
                    {model.version}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-end space-x-3">
            <div className="flex items-center gap-2 rounded-md border px-3 py-2">
              <Checkbox
                id="onlyValueBets"
                checked={filters.onlyValueBets || false}
                onCheckedChange={(checked) =>
                  updateFilter("onlyValueBets", checked === true)
                }
              />
              <Label htmlFor="onlyValueBets" className="cursor-pointer">
                Only value bets
              </Label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
