/**
 * API client functions for tournament operations
 */

import { apiGet, buildQueryString } from "./client";
import { z } from "zod";
import { TournamentSchema, SeasonSchema } from "@/lib/types/schemas";
import type { Tournament, Season } from "@/lib/types/api";

export const TournamentArraySchema = z.array(TournamentSchema);
export const SeasonArraySchema = z.array(SeasonSchema);

/**
 * Get all tournaments
 */
export async function getTournaments(): Promise<Tournament[]> {
  return apiGet("/tournaments", TournamentArraySchema);
}

/**
 * Get a specific tournament by ID
 */
export async function getTournament(id: number): Promise<Tournament> {
  return apiGet(`/tournaments/${id}`, TournamentSchema);
}

/**
 * Get all seasons for a tournament
 */
export async function getTournamentSeasons(id: number): Promise<Season[]> {
  return apiGet(`/tournaments/${id}/seasons`, SeasonArraySchema);
}
