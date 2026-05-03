"use client";

import { useState, useMemo } from "react";
import { Search, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { ModelVersion } from "@/lib/types/api";

interface ModelRegistryProps {
  models: ModelVersion[];
  activeVersion: string | null;
  selectedModelId: number | null;
  onSelectModel: (model: ModelVersion | null) => void;
  onActivate: (id: number) => void;
  onDelete: (id: number) => void;
  isLoading?: boolean;
}

type StatusFilter = "all" | "active" | "inactive";

const ALGORITHMS = ["xgboost", "lightgbm", "random_forest"];

export function ModelRegistry({
  models,
  activeVersion,
  selectedModelId,
  onSelectModel,
  onActivate,
  onDelete,
  isLoading,
}: ModelRegistryProps) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [algorithmFilter, setAlgorithmFilter] = useState<string[]>([]);

  const filteredModels = useMemo(() => {
    return models.filter((model) => {
      if (statusFilter === "active" && !model.is_active) return false;
      if (statusFilter === "inactive" && model.is_active) return false;

      if (algorithmFilter.length > 0 && !algorithmFilter.includes(model.algorithm)) {
        return false;
      }

      if (search) {
        const searchLower = search.toLowerCase();
        return (
          model.version.toLowerCase().includes(searchLower) ||
          (model.description?.toLowerCase().includes(searchLower) ?? false) ||
          model.algorithm.toLowerCase().includes(searchLower)
        );
      }

      return true;
    });
  }, [models, statusFilter, algorithmFilter, search]);

  const toggleAlgorithm = (algo: string) => {
    setAlgorithmFilter((prev) =>
      prev.includes(algo) ? prev.filter((a) => a !== algo) : [...prev, algo]
    );
  };

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-lg border bg-muted" />
        ))}
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <p className="text-lg font-medium">No models found</p>
          <p className="text-sm">Train your first model to get started.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search version, description, algorithm..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge
            variant={statusFilter === "all" ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setStatusFilter("all")}
          >
            All
          </Badge>
          <Badge
            variant={statusFilter === "active" ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setStatusFilter("active")}
          >
            Active
          </Badge>
          <Badge
            variant={statusFilter === "inactive" ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setStatusFilter("inactive")}
          >
            Inactive
          </Badge>

          <div className="flex-1" />

          <div className="flex flex-wrap gap-1">
            {ALGORITHMS.map((algo) => (
              <Badge
                key={algo}
                variant={algorithmFilter.includes(algo) ? "default" : "outline"}
                className="cursor-pointer text-xs"
                onClick={() => toggleAlgorithm(algo)}
              >
                {algo}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      {filteredModels.length === 0 ? (
        <div className="py-8 text-center text-muted-foreground">
          No models match your filters.
        </div>
      ) : (
        <div className="space-y-2">
          {filteredModels.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              isActive={model.version === activeVersion}
              isSelected={selectedModelId === model.id}
              onSelect={() => onSelectModel(model)}
              onActivate={() => onActivate(model.id)}
              onDelete={() => {
                if (confirm(`Delete model ${model.version}?`)) {
                  onDelete(model.id);
                }
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ModelCardProps {
  model: ModelVersion;
  isActive: boolean;
  isSelected: boolean;
  onSelect: () => void;
  onActivate: () => void;
  onDelete: () => void;
}

function ModelCard({
  model,
  isActive,
  isSelected,
  onSelect,
  onActivate,
  onDelete,
}: ModelCardProps) {
  return (
    <Card
      className={`cursor-pointer transition-colors hover:bg-muted/50 ${
        isSelected ? "border-primary" : ""
      }`}
      onClick={onSelect}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-medium">{model.version}</span>
              {isActive && <Badge className="bg-green-600 text-xs">Active</Badge>}
            </div>
            <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
              <Badge variant="outline" className="text-xs">
                {model.algorithm}
              </Badge>
              {model.accuracy !== null && (
                <span>{(model.accuracy * 100).toFixed(1)}% accuracy</span>
              )}
            </div>
            {model.description && (
              <p className="mt-2 line-clamp-2 text-xs text-muted-foreground">
                {model.description}
              </p>
            )}
            <p className="mt-1 text-xs text-muted-foreground">
              {new Date(model.created_at).toLocaleDateString()}
            </p>
          </div>

          <div className="flex flex-col gap-1">
            {!isActive && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    onActivate();
                  }}
                >
                  Activate
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs text-destructive hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
                  }}
                >
                  Delete
                </Button>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
