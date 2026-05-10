"""Odds residual feature generator."""

from datetime import datetime
from typing import Any

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class OddsResidualFeatureGenerator(FeatureGenerator):
    """Generate features that compare team form to market-implied expectations.

    These residual features represent how much better or worse each team is
    performing relative to what the betting market predicts -- the "surprise"
    signal.  This is orthogonal to raw odds and raw form features: instead of
    encoding home advantage three times (odds + venue form + naming), the
    model receives the market expectation once (via odds features) and a
    deviation signal (via these residuals).

    Expected points from odds:
        home_expected_pts = 3 * implied_prob_home + 1 * implied_prob_draw
        away_expected_pts = 3 * implied_prob_away + 1 * implied_prob_draw

    Surprise = actual PPG - expected_pts_per_game
    A positive surprise means the team is outperforming the market; negative
    means underperforming.
    """

    def __init__(
        self,
        form_windows: list[int] | None = None,
    ) -> None:
        self.form_windows = form_windows or [5, 10]

    @property
    def name(self) -> str:
        return "odds_residual"

    @property
    def feature_names(self) -> list[str]:
        names = [
            "home_form_surprise",
            "away_form_surprise",
            "home_venue_form_surprise",
            "away_venue_form_surprise",
            "form_surprise_diff",
            "venue_surprise_diff",
            "home_advantage_net",
        ]
        for w in self.form_windows:
            names.extend(
                [
                    f"home_form_surprise_{w}",
                    f"away_form_surprise_{w}",
                ]
            )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        max_window = max(self.form_windows, default=10)
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            odds_home = match.get("odds_home")
            odds_draw = match.get("odds_draw")
            odds_away = match.get("odds_away")

            implied = self._implied_probabilities(odds_home, odds_draw, odds_away)

            home_form = self._ppg(repository, home_team_id, match_date, max_window)
            away_form = self._ppg(repository, away_team_id, match_date, max_window)
            home_venue_form = self._venue_ppg(
                repository, home_team_id, match_date, max_window, is_home=True
            )
            away_venue_form = self._venue_ppg(
                repository, away_team_id, match_date, max_window, is_home=False
            )

            home_exp_pts = 3 * implied["home"] + implied["draw"]
            away_exp_pts = 3 * implied["away"] + implied["draw"]

            match_feats: dict[str, Any] = {
                "match_id": match_id,
                "home_form_surprise": home_form - home_exp_pts,
                "away_form_surprise": away_form - away_exp_pts,
                "home_venue_form_surprise": home_venue_form - home_exp_pts,
                "away_venue_form_surprise": away_venue_form - away_exp_pts,
                "form_surprise_diff": (home_form - home_exp_pts)
                - (away_form - away_exp_pts),
                "venue_surprise_diff": (home_venue_form - home_exp_pts)
                - (away_venue_form - away_exp_pts),
                "home_advantage_net": implied["home"] - implied["away"],
            }

            for w in self.form_windows:
                home_w = self._ppg(repository, home_team_id, match_date, w)
                away_w = self._ppg(repository, away_team_id, match_date, w)
                match_feats[f"home_form_surprise_{w}"] = home_w - home_exp_pts
                match_feats[f"away_form_surprise_{w}"] = away_w - away_exp_pts

            features.append(match_feats)

        return pd.DataFrame(features).set_index("match_id")

    def _implied_probabilities(
        self,
        odds_home: Any,
        odds_draw: Any,
        odds_away: Any,
    ) -> dict[str, float]:
        if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
            return {"home": 0.40, "draw": 0.30, "away": 0.30}
        try:
            h = 1.0 / float(odds_home)
            d = 1.0 / float(odds_draw)
            a = 1.0 / float(odds_away)
            total = h + d + a
            return {"home": h / total, "draw": d / total, "away": a / total}
        except (ZeroDivisionError, ValueError, TypeError):
            return {"home": 0.40, "draw": 0.30, "away": 0.30}

    @staticmethod
    def _ppg(
        repo: MatchRepository,
        team_id: int,
        before_date: datetime,
        limit: int,
    ) -> float:
        matches = repo.get_team_matches(
            team_id=team_id, before_date=before_date, limit=limit
        )
        if not matches:
            return 0.0
        points = 0
        for match in matches:
            is_home = match.home_team_id == team_id
            gf = match.home_score if is_home else match.away_score
            ga = match.away_score if is_home else match.home_score
            gf = gf or 0
            ga = ga or 0
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        return points / len(matches)

    @staticmethod
    def _venue_ppg(
        repo: MatchRepository,
        team_id: int,
        before_date: datetime,
        limit: int,
        is_home: bool,
    ) -> float:
        matches = repo.get_team_matches(
            team_id=team_id,
            before_date=before_date,
            limit=limit,
            home_only=is_home,
            away_only=not is_home,
        )
        if not matches:
            return 0.0
        points = 0
        for match in matches:
            home = match.home_team_id == team_id
            gf = match.home_score if home else match.away_score
            ga = match.away_score if home else match.home_score
            gf = gf or 0
            ga = ga or 0
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        return points / len(matches)
