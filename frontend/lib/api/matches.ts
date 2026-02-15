/**
 * API functions for matches
 */

import { apiGet, buildQueryString } from "./client";
import type {
  Match,
  MatchDetail,
  MatchFilters,
  PaginatedResponse,
} from "@/lib/types/api";
import {
  MatchSchema,
  MatchDetailSchema,
  createPaginatedResponseSchema,
} from "@/lib/types/schemas";

const matchArraySchema = createPaginatedResponseSchema(MatchSchema);
const matchDetailSchema = MatchDetailSchema;

export async function getMatches(
  filters?: MatchFilters
): Promise<PaginatedResponse<Match>> {
  const queryString = filters ? buildQueryString(filters) : "";
  return apiGet(`/matches${queryString}`, matchArraySchema);
}

export async function getMatch(id: number): Promise<MatchDetail> {
  return apiGet(`/matches/${id}`, matchDetailSchema);
}

export async function getMatchPreview(id: number): Promise<MatchDetail> {
  return apiGet(`/matches/${id}/preview`, matchDetailSchema);
}

export async function getMatchPredictions(id: number): Promise<MatchDetail> {
  return apiGet(`/matches/${id}/predictions`, matchDetailSchema);
}

export async function getMatchH2H(id: number): Promise<PaginatedResponse<Match>> {
  return apiGet(`/matches/${id}/h2h`, matchArraySchema);
}
