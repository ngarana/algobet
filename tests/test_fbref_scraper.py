# ruff: noqa: E501

from lxml import html

from algobet.fbref_scraper import FBrefScraper


def test_match_player_stats_use_exact_fbref_team_tables() -> None:
    page = html.fromstring(
        """
        <html>
          <body>
            <div class="scorebox">
              <strong><a href="/en/squads/home123/Home-Team">Home Team</a></strong>
              <strong><a href="/en/squads/away456/Away-Team">Away Team</a></strong>
            </div>
            <span class="venuetime" data-venue-date="2020-09-12">2020-09-12</span>
            <table id="stats_home123_summary">
              <thead>
                <tr>
                  <th data-stat="player">Player</th>
                  <th data-stat="shirtnumber">#</th>
                  <th data-stat="position">Pos</th>
                  <th data-stat="minutes">Min</th>
                  <th data-stat="goals">Gls</th>
                  <th data-stat="assists">Ast</th>
                  <th data-stat="shots">Sh</th>
                  <th data-stat="shots_on_target">SoT</th>
                  <th data-stat="cards_yellow">CrdY</th>
                  <th data-stat="cards_red">CrdR</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th data-stat="player"><a href="/en/players/p1/player">Home Player</a></th>
                  <td data-stat="shirtnumber">9</td>
                  <td data-stat="position">FW</td>
                  <td data-stat="minutes">90</td>
                  <td data-stat="goals">1</td>
                  <td data-stat="assists">0</td>
                  <td data-stat="shots">3</td>
                  <td data-stat="shots_on_target">2</td>
                  <td data-stat="cards_yellow">1</td>
                  <td data-stat="cards_red">0</td>
                </tr>
              </tbody>
            </table>
            <table id="keeper_stats_home123">
              <thead>
                <tr>
                  <th data-stat="player">Player</th>
                  <th data-stat="saves">Saves</th>
                  <th data-stat="gk_goals_against">GA</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th data-stat="player"><a href="/en/players/gk/player">Home Keeper</a></th>
                  <td data-stat="saves">4</td>
                  <td data-stat="gk_goals_against">1</td>
                </tr>
              </tbody>
            </table>
            <table id="stats_away456_summary">
              <thead>
                <tr>
                  <th data-stat="player">Player</th>
                  <th data-stat="shirtnumber">#</th>
                  <th data-stat="position">Pos</th>
                  <th data-stat="minutes">Min</th>
                  <th data-stat="goals">Gls</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th data-stat="player"><a href="/en/players/p2/player">Away Player</a></th>
                  <td data-stat="shirtnumber">10</td>
                  <td data-stat="position">MF</td>
                  <td data-stat="minutes">88</td>
                  <td data-stat="goals">0</td>
                </tr>
              </tbody>
            </table>
          </body>
        </html>
        """
    )

    stats = FBrefScraper(headless=True)._scrape_match_player_stats_from_tree(page)

    assert stats.home_team == "Home Team"
    assert stats.away_team == "Away Team"
    assert stats.match_date is not None
    assert [player.player_name for player in stats.home_players] == ["Home Player"]
    assert [player.player_name for player in stats.away_players] == ["Away Player"]
    assert stats.home_players[0].goals == 1
    assert stats.home_players[0].shots_on_target == 2
    assert stats.home_players[0].team_name == "Home Team"
    assert stats.away_players[0].team_name == "Away Team"
