/**
 * Global filter state store
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { MatchFilters, PredictionFilters } from "@/lib/types/api";

interface FilterState {
  // Match filters
  matchFilters: MatchFilters;
  setMatchFilters: (filters: Partial<MatchFilters>) => void;
  resetMatchFilters: () => void;

  // Prediction filters
  predictionFilters: PredictionFilters;
  setPredictionFilters: (filters: Partial<PredictionFilters>) => void;
  resetPredictionFilters: () => void;

  // Active tournament filter (shared across views)
  activeTournamentId: number | null;
  setActiveTournamentId: (id: number | null) => void;

  // Date range filters
  dateRange: {
    from: string | null;
    to: string | null;
  };
  setDateRange: (from: string | null, to: string | null) => void;
}

const initialMatchFilters: MatchFilters = {
  limit: 50,
  offset: 0,
};

const initialPredictionFilters: PredictionFilters = {};

export const useFilterStore = create<FilterState>()(
  persist(
    (set) => ({
      // Match filters
      matchFilters: initialMatchFilters,
      setMatchFilters: (filters) =>
        set((state) => ({
          matchFilters: { ...state.matchFilters, ...filters },
        })),
      resetMatchFilters: () => set({ matchFilters: initialMatchFilters }),

      // Prediction filters
      predictionFilters: initialPredictionFilters,
      setPredictionFilters: (filters) =>
        set((state) => ({
          predictionFilters: { ...state.predictionFilters, ...filters },
        })),
      resetPredictionFilters: () =>
        set({ predictionFilters: initialPredictionFilters }),

      // Active tournament
      activeTournamentId: null,
      setActiveTournamentId: (id) => set({ activeTournamentId: id }),

      // Date range
      dateRange: {
        from: null,
        to: null,
      },
      setDateRange: (from, to) => set({ dateRange: { from, to } }),
    }),
    {
      name: "algobet-filters",
    }
  )
);
