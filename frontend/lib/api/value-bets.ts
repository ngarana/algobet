/**
 * API functions for value bets
 */

import { apiGet, buildQueryString } from "./client";
import type { ValueBet } from "@/lib/types/api";
import { ValueBetSchema } from "@/lib/types/schemas";
import { z } from "zod";

const valueBetArraySchema = z.array(ValueBetSchema);

interface ValueBetFilters {
  min_ev?: number;
  max_odds?: number;
  days?: number;
  model_version?: string;
  min_confidence?: number;
  max_matches?: number;
}

export async function getValueBets(filters?: ValueBetFilters): Promise<ValueBet[]> {
  const queryString = filters ? buildQueryString(filters) : "";
  return apiGet(`/value-bets${queryString}`, valueBetArraySchema);
}
