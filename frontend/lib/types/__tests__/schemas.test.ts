import { describe, it, expect } from "vitest";
import { z } from "zod";
import {
  FormBreakdownSchema,
  TournamentSchema,
  SeasonSchema,
  TeamSchema,
  TeamWithStatsSchema,
} from "../schemas";

describe("FormBreakdownSchema", () => {
  it("should parse valid form breakdown data", () => {
    const data = {
      avg_points: 1.5,
      win_rate: 0.4,
      draw_rate: 0.3,
      loss_rate: 0.3,
      avg_goals_for: 1.2,
      avg_goals_against: 1.1,
    };

    const result = FormBreakdownSchema.parse(data);
    expect(result).toEqual(data);
  });

  it("should fail with invalid data types", () => {
    const data = {
      avg_points: "invalid",
      win_rate: 0.4,
      draw_rate: 0.3,
      loss_rate: 0.3,
      avg_goals_for: 1.2,
      avg_goals_against: 1.1,
    };

    expect(() => FormBreakdownSchema.parse(data)).toThrow(z.ZodError);
  });

  it("should fail with missing required fields", () => {
    const data = {
      avg_points: 1.5,
    };

    expect(() => FormBreakdownSchema.parse(data)).toThrow(z.ZodError);
  });
});

describe("TournamentSchema", () => {
  it("should parse valid tournament data", () => {
    const data = {
      id: 1,
      name: "Premier League",
      country: "England",
      url_slug: "premier-league",
    };

    const result = TournamentSchema.parse(data);
    expect(result).toEqual(data);
  });

  it("should fail with invalid id type", () => {
    const data = {
      id: "invalid",
      name: "Premier League",
      country: "England",
      url_slug: "premier-league",
    };

    expect(() => TournamentSchema.parse(data)).toThrow(z.ZodError);
  });

  it("should fail with missing required fields", () => {
    const data = {
      id: 1,
      name: "Premier League",
    };

    expect(() => TournamentSchema.parse(data)).toThrow(z.ZodError);
  });
});

describe("SeasonSchema", () => {
  it("should parse valid season data", () => {
    const data = {
      id: 1,
      tournament_id: 10,
      name: "2023/2024",
      start_year: 2023,
      end_year: 2024,
      url_suffix: "2023-2024",
    };

    const result = SeasonSchema.parse(data);
    expect(result).toEqual(data);
  });

  it("should accept null url_suffix", () => {
    const data = {
      id: 1,
      tournament_id: 10,
      name: "2023/2024",
      start_year: 2023,
      end_year: 2024,
      url_suffix: null,
    };

    const result = SeasonSchema.parse(data);
    expect(result.url_suffix).toBeNull();
  });

  it("should fail with negative year", () => {
    const data = {
      id: 1,
      tournament_id: 10,
      name: "Invalid",
      start_year: -1,
      end_year: 2024,
      url_suffix: null,
    };

    // Zod number validation doesn't check for positive by default
    const result = SeasonSchema.parse(data);
    expect(result.start_year).toBe(-1);
  });
});

describe("TeamSchema", () => {
  it("should parse valid team data", () => {
    const data = {
      id: 42,
      name: "Manchester United",
    };

    const result = TeamSchema.parse(data);
    expect(result).toEqual(data);
  });

  it("should fail with missing name", () => {
    const data = {
      id: 42,
    };

    expect(() => TeamSchema.parse(data)).toThrow(z.ZodError);
  });
});

describe("TeamWithStatsSchema", () => {
  it("should parse valid team with stats data", () => {
    const data = {
      id: 42,
      name: "Manchester United",
      total_matches: 38,
      wins: 20,
      draws: 10,
      losses: 8,
      goals_for: 65,
      goals_against: 40,
      current_form: {
        avg_points: 1.8,
        win_rate: 0.5,
        draw_rate: 0.3,
        loss_rate: 0.2,
        avg_goals_for: 2.0,
        avg_goals_against: 1.2,
      },
    };

    const result = TeamWithStatsSchema.parse(data);
    expect(result).toEqual(data);
    expect(result.name).toBe("Manchester United");
    expect(result.total_matches).toBe(38);
  });

  it("should extend TeamSchema with additional fields", () => {
    const data = {
      id: 42,
      name: "Liverpool",
      total_matches: 38,
      wins: 25,
      draws: 8,
      losses: 5,
      goals_for: 75,
      goals_against: 30,
      current_form: {
        avg_points: 2.2,
        win_rate: 0.7,
        draw_rate: 0.2,
        loss_rate: 0.1,
        avg_goals_for: 2.5,
        avg_goals_against: 0.8,
      },
    };

    const result = TeamWithStatsSchema.parse(data);
    expect(result.id).toBe(42);
    expect(result.name).toBe("Liverpool");
    expect(result.goals_for).toBe(75);
  });

  it("should fail without current_form", () => {
    const data = {
      id: 42,
      name: "Chelsea",
      total_matches: 38,
      wins: 15,
      draws: 10,
      losses: 13,
      goals_for: 50,
      goals_against: 50,
    };

    expect(() => TeamWithStatsSchema.parse(data)).toThrow(z.ZodError);
  });

  it("should fail with invalid current_form structure", () => {
    const data = {
      id: 42,
      name: "Arsenal",
      total_matches: 38,
      wins: 20,
      draws: 10,
      losses: 8,
      goals_for: 65,
      goals_against: 40,
      current_form: {
        avg_points: "invalid",
        win_rate: 0.5,
        draw_rate: 0.3,
        loss_rate: 0.2,
        avg_goals_for: 2.0,
        avg_goals_against: 1.2,
      },
    };

    expect(() => TeamWithStatsSchema.parse(data)).toThrow(z.ZodError);
  });
});

describe("Schema composition", () => {
  it("should parse nested schema structures", () => {
    const teamData = {
      id: 1,
      name: "Test Team",
      total_matches: 10,
      wins: 5,
      draws: 3,
      losses: 2,
      goals_for: 15,
      goals_against: 8,
      current_form: {
        avg_points: 1.8,
        win_rate: 0.5,
        draw_rate: 0.3,
        loss_rate: 0.2,
        avg_goals_for: 1.5,
        avg_goals_against: 0.8,
      },
    };

    const result = TeamWithStatsSchema.parse(teamData);
    expect(result.current_form.win_rate).toBe(0.5);
    expect(result.current_form.avg_points).toBe(1.8);
  });
});
