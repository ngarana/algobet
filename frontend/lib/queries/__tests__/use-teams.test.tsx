import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import {
  teamKeys,
  useTeams,
  useTeam,
  useTeamForm,
  useTeamMatches,
  useInvalidateTeams,
} from "../use-teams";

// Mock API module
vi.mock("@/lib/api/teams", () => ({
  getTeams: vi.fn(),
  getTeam: vi.fn(),
  getTeamForm: vi.fn(),
  getTeamMatches: vi.fn(),
}));

const { getTeams, getTeam, getTeamForm, getTeamMatches } =
  await import("@/lib/api/teams");

interface WrapperProps {
  children: ReactNode;
}

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return function Wrapper({ children }: WrapperProps) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
};

describe("teamKeys", () => {
  it("should have correct base key", () => {
    expect(teamKeys.all).toEqual(["teams"]);
  });

  it("should generate list key", () => {
    expect(teamKeys.lists()).toEqual(["teams", "list"]);
  });

  it("should generate list key with filters", () => {
    const filters = { tournament_id: 1 };
    expect(teamKeys.list(filters)).toEqual(["teams", "list", { tournament_id: 1 }]);
  });

  it("should generate detail key", () => {
    expect(teamKeys.detail(42)).toEqual(["teams", "detail", 42]);
  });

  it("should generate form key", () => {
    expect(teamKeys.form(42)).toEqual(["teams", "detail", 42, "form"]);
  });

  it("should generate matches key", () => {
    expect(teamKeys.matches(42)).toEqual(["teams", "detail", 42, "matches"]);
  });
});

describe("useTeams", () => {
  it("should fetch teams without filters", async () => {
    const mockTeams = [
      { id: 1, name: "Team A" },
      { id: 2, name: "Team B" },
    ];
    vi.mocked(getTeams).mockResolvedValue(mockTeams);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeams(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeams).toHaveBeenCalledWith(undefined);
    expect(result.current.data).toEqual(mockTeams);
  });

  it("should fetch teams with filters", async () => {
    const mockTeams = [{ id: 1, name: "Filtered Team" }];
    const filters = { tournament_id: 5, limit: 10 };
    vi.mocked(getTeams).mockResolvedValue(mockTeams);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeams(filters), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeams).toHaveBeenCalledWith(filters);
    expect(result.current.data).toEqual(mockTeams);
  });

  it("should handle errors", async () => {
    vi.mocked(getTeams).mockRejectedValue(new Error("Failed to fetch"));

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeams(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeInstanceOf(Error);
  });
});

describe("useTeam", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should fetch team by id", async () => {
    const mockTeam = { id: 42, name: "Test Team" };
    vi.mocked(getTeam).mockResolvedValue(mockTeam);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeam(42), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeam).toHaveBeenCalledWith(42);
    expect(result.current.data).toEqual(mockTeam);
  });

  it("should not fetch when id is falsy", () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeam(0), { wrapper });

    // When enabled is false, the query doesn't run
    expect(result.current.isLoading).toBe(false);
    expect(getTeam).not.toHaveBeenCalled();
  });
});

describe("useTeamForm", () => {
  it("should fetch team form", async () => {
    const mockForm = {
      avg_points: 1.5,
      win_rate: 0.4,
      draw_rate: 0.3,
      loss_rate: 0.3,
    };
    vi.mocked(getTeamForm).mockResolvedValue(mockForm);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeamForm(42), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeamForm).toHaveBeenCalledWith(42, undefined);
  });

  it("should fetch team form with nMatches parameter", async () => {
    const mockForm = { avg_points: 2.0, win_rate: 0.6 };
    vi.mocked(getTeamForm).mockResolvedValue(mockForm);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeamForm(42, 5), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeamForm).toHaveBeenCalledWith(42, 5);
  });
});

describe("useTeamMatches", () => {
  it("should fetch team matches", async () => {
    const mockMatches = [{ id: 1, home_team: "Team A", away_team: "Team B" }];
    vi.mocked(getTeamMatches).mockResolvedValue(mockMatches);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeamMatches(42), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeamMatches).toHaveBeenCalledWith(42, undefined, undefined);
  });

  it("should fetch home matches only", async () => {
    const mockMatches = [{ id: 1, venue: "home" }];
    vi.mocked(getTeamMatches).mockResolvedValue(mockMatches);

    const wrapper = createWrapper();
    const { result } = renderHook(() => useTeamMatches(42, "home", 10), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(getTeamMatches).toHaveBeenCalledWith(42, "home", 10);
  });
});

describe("useInvalidateTeams", () => {
  it("should provide invalidation methods", () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useInvalidateTeams(), { wrapper });

    expect(result.current.invalidateAll).toBeDefined();
    expect(result.current.invalidateList).toBeDefined();
    expect(result.current.invalidateDetail).toBeDefined();
    expect(typeof result.current.invalidateAll).toBe("function");
    expect(typeof result.current.invalidateList).toBe("function");
    expect(typeof result.current.invalidateDetail).toBe("function");
  });

  it("should invalidate all team queries", async () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useInvalidateTeams(), { wrapper });

    // Just verify the method exists and can be called
    expect(() => result.current.invalidateAll()).not.toThrow();
  });

  it("should invalidate specific team detail queries", async () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useInvalidateTeams(), { wrapper });

    expect(() => result.current.invalidateDetail(42)).not.toThrow();
  });
});
