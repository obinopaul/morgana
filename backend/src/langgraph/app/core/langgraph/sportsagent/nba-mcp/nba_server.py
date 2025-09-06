from mcp.server.fastmcp import FastMCP
import time
import signal
import sys
from nba_api.live.nba.endpoints import scoreboard, boxscore, playbyplay
from nba_api.stats.static import players, teams
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import os
from nba_api.live.nba.endpoints import scoreboard, boxscore, playbyplay
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats, scoreboardv2, teamgamelogs, leaguegamefinder, leaguestandingsv3, teamyearbyyearstats
from nba_api.stats.static import players, teams
from nba_api.stats.library.parameters import SeasonType, SeasonYear

# print(f"Python executable: {sys.executable}", file=sys.stderr)
# print(f"Python path: {sys.path}", file=sys.stderr)
print(f"Current working directory: {os.getcwd()}", file=sys.stderr)

# Handle SIGINT (Ctrl+C) gracefully
def signal_handler(sig, frame):
    print("Shutting down server gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Create an MCP server with increased timeout
mcp = FastMCP(
    name="nba_mcp_server",
    # host="127.0.0.1",
    # port=5000,
    # Add this to make the server more resilient
    timeout=30  # Increase timeout to 30 seconds
)

# -------------------------------------------------------------------
# 1) ScoreBoard Tool (Live Endpoint)
# -------------------------------------------------------------------

class LiveScoreBoardInput(BaseModel):
    dummy_param: Optional[str] = Field(default="", description="Not used.")

@mcp.tool()
def nba_live_scoreboard(dummy_param: Optional[str] = "") -> Dict[str, Any]:
    """Fetch today's NBA scoreboard (live or latest).

    This tool retrieves data from the `nba_api.live.nba.endpoints.scoreboard` endpoint.  It provides
    information about games happening *today* (or the most recent games if no games are live).
    This includes game IDs, start times, scores, period information, and broadcast details.

    **Args:**

        dummy_param (str, optional):  This parameter is not used. It exists for compatibility
            with the MCP framework and should be left as an empty string. Defaults to "".

    **Returns:**

        Dict[str, Any]: A dictionary containing scoreboard data.  The structure follows the
            `nba_api`'s `ScoreBoard` object. Key elements include:

            * "games": A list of dictionaries, one for each game.  Each game dictionary contains:
                * "gameId": (str) The 10-digit game ID.  **Crucially, this is needed for other live tools.**
                * "gameStatus": (int)  A numeric representation of the game status (1 = scheduled, 2 = in progress, 3 = final).
                * "gameStatusText": (str) A textual representation of the game status (e.g., "Final", "Q4 05:30", "8:00 pm ET").
                * "homeTeam": (dict) Information about the home team.
                * "awayTeam": (dict) Information about the away team.
                *  ...and many other fields.

            * "gameDate": (str) - The date of the game
            *  "scoreboard": (dict) - Contains overall scoreboard of that date.

            If an error occurs, the dictionary will contain a single "error" key with a
            description of the problem.
    """
    try:
        sb = scoreboard.ScoreBoard()
        return sb.get_dict()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# 2) BoxScore Tool (Live Endpoint)
# -------------------------------------------------------------------

class LiveBoxScoreInput(BaseModel):
    game_id: str = Field(..., description="A 10-digit NBA game ID (e.g., '0022200017').")

@mcp.tool()
def nba_live_boxscore(game_id: str) -> Dict[str, Any]:
    """Fetch the real-time box score for a given NBA game ID.

    This tool retrieves live box score data from the `nba_api.live.nba.endpoints.boxscore`
    endpoint.  It provides detailed statistics for a *specific* game, including:

    * Player statistics (points, rebounds, assists, etc.)
    * Team statistics (points by quarter, totals)
    * Active players
    * Game officials

    **Args:**

        game_id (str): The 10-digit NBA game ID.  This is typically obtained from
            `nba_live_scoreboard`.  Example: "0022300123"

    **Returns:**

        Dict[str, Any]: A dictionary containing the box score data. The structure follows
            the `nba_api`'s `BoxScore` object. Key elements include:

            * "gameId": The game ID.
            * "gameStatus": Numeric game status.
            *  "boxScoreTraditional": (dict) - contains player and team stats.
            * "teams": A list of two dictionaries (one for each team), containing:
                * "teamId": The team ID.
                * "teamName": The team name.
                * "teamCity": The team city.
                * "players": A list of dictionaries, one for each player, with detailed stats.
            * ...and many other fields.

            If an error occurs, the dictionary will contain a single "error" key.

    """
    if not isinstance(game_id, str):
        game_id = str(game_id)
        
    try:
        bs = boxscore.BoxScore(game_id=game_id)
        return bs.get_dict()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# 3) PlayByPlay Tool (Live Endpoint)
# -------------------------------------------------------------------

class LivePlayByPlayInput(BaseModel):
    game_id: str = Field(..., description="A 10-digit NBA game ID.")

@mcp.tool()
def nba_live_play_by_play(game_id: str) -> Dict[str, Any]:
    """Retrieve the live play-by-play actions for a specific NBA game ID.

    This tool retrieves data from the `nba_api.live.nba.endpoints.playbyplay` endpoint.
    It provides a chronological list of events that occur during a game, including:

    * Scoring plays
    * Fouls
    * Timeouts
    * Substitutions
    * Descriptions of each play

    **Args:**

        game_id (str): The 10-digit NBA game ID.  Obtain this from `nba_live_scoreboard`.
            Example: "0022300123"

    **Returns:**

        Dict[str, Any]: A dictionary containing the play-by-play data. The structure follows
            the `nba_api`'s `PlayByPlay` object.  Key elements include:

            * "gameId": The game ID.
            * "actions": A list of dictionaries, one for each play.  Each play dictionary contains:
                * "actionNumber": A sequential number for the play.
                * "clock": The game clock time when the play occurred.
                * "period": The quarter/overtime period.
                * "teamId": The ID of the team involved in the play (if applicable).
                * "personId": The ID of the player involved in the play (if applicable).
                * "description": A textual description of the play.
                * ...and many other fields.

            If an error occurs, the dictionary will contain a single "error" key.
    """
    if not isinstance(game_id, str):
        game_id = str(game_id)
        
    try:
        pbp = playbyplay.PlayByPlay(game_id=game_id)
        return pbp.get_dict()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# 4) CommonPlayerInfo Tool (Stats Endpoint)
# -------------------------------------------------------------------

class CommonPlayerInfoInput(BaseModel):
    player_id: str = Field(..., description="NBA player ID (e.g., '2544').")

@mcp.tool()
def nba_common_player_info(player_id: str) -> Dict[str, Any]:
    """Retrieve basic information about a player.

    This tool retrieves data from the `nba_api.stats.endpoints.commonplayerinfo` endpoint.
    It provides biographical and basic information about a specific NBA player, including:

    * Player ID
    * Full Name
    * Birthdate
    * Height
    * Weight
    * Current Team
    * Jersey Number
    * Position
    * Draft information
    * College

    **Args:**

        player_id (str): The NBA player ID.  This is typically a number, like "2544" (LeBron James).
            You can use `nba_search_players` (not yet documented here, but in your original code) to
            find a player ID by name.

    **Returns:**

        Dict[str, Any]: A dictionary containing player information.  The structure follows the
            `nba_api`'s `CommonPlayerInfo` object. Key elements include:

            * "CommonPlayerInfo": A list containing a single dictionary with player details.
                * "personId": The player ID.
                * "displayFirstLast": The player's full name.
                * "birthdate": The player's birthdate.
                * "height": Player height.
                * "weight": Player weight.
                * "teamId":  The ID of the player's current team.
                * "teamName": The name of the player's current team.
                * ... and many other fields

             * "ResultSets": (list) - Contains the results in sets.

            If an error occurs, the dictionary will contain a single "error" key.

    """
    if not isinstance(player_id, str):
        player_id = str(player_id)
        
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        return info.get_dict()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# 5) PlayerCareerStats Tool (Stats Endpoint)
# -------------------------------------------------------------------

class PlayerCareerStatsInput(BaseModel):
    player_id: str = Field(..., description="NBA player ID.")
    per_mode: Optional[str] = Field(default="PerGame", description="One of 'Totals', 'PerGame', 'Per36'.")

@mcp.tool()
def nba_player_career_stats(player_id: str, per_mode: str = "PerGame") -> Dict[str, Any]:
    """Obtain an NBA player's career statistics.

    This tool retrieves career statistics (regular season, playoffs, and potentially All-Star games)
    from the `nba_api.stats.endpoints.playercareerstats` endpoint.  It provides aggregated
    statistics for a player across their entire career or specific seasons.

    **Args:**

        player_id (str): The NBA player ID (e.g., "2544").
        per_mode (str, optional):  Determines the statistical aggregation.  Valid options are:
            * "PerGame":  Stats averaged per game played.
            * "Totals":  Total career statistics.
            * "Per36": Stats per 36 minutes played.
            Defaults to "PerGame".

    **Returns:**

        Dict[str, Any]: A dictionary containing the player's career statistics. The structure
            follows the `nba_api`'s `PlayerCareerStats` object.  Key elements include:

            * "SeasonTotalsRegularSeason": A list of dictionaries, one for each season the
              player played in the regular season.  Each dictionary contains aggregated stats
              for that season (e.g., games played, points, rebounds, assists, etc.).
            * "CareerTotalsRegularSeason":  A list containing a single dictionary with the
              player's total career regular season stats.
            * "SeasonTotalsPostSeason", "CareerTotalsPostSeason": Similar data for playoff games.
            * "SeasonTotalsAllStarSeason", "CareerTotalsAllStarSeason": Similar data for All-Star games.
            * "resultSets" (list): Contains different sets of career stats.

            If an error occurs, the dictionary will contain a single "error" key.

    """
    # Convert player_id to a string if it's not already.
    if not isinstance(player_id, str):
        player_id = str(player_id)
        
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id, per_mode36=per_mode)
        return career.get_dict()
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------
# 8) List All Active Players
# -------------------------------------------------------------------
class ListActivePlayersInput(BaseModel):
    # no arguments needed
    dummy: str = "unused"

@mcp.tool()
def nba_list_active_players(dummy: str = "") -> List[Dict[str, Any]]:
    """Return a list of all currently active NBA players.

    This tool uses the `nba_api.stats.static.players` module to retrieve a list of all players
    marked as active in the NBA API's database.

    **Args:**

        dummy (str, optional): This parameter is not used. It is included for compatibility with the MCP framework.

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents an active player.
            Each player dictionary contains:
            * "id": (int) The player's ID.
            * "full_name": (str) The player's full name.
            * "first_name": (str) The player's first name.
            * "last_name": (str) The player's last name.
            * "is_active": (bool) Always True for this function.
            If there's an issue, a list containing a dictionary with an "error" key is returned.

    """
    try:
        all_active = players.get_active_players()
        return all_active
    except Exception as e:
        return [{"error": str(e)}]

# -------------------------------------------------------------------
# 9) List Today’s Games (Stats vs. Live)
# -------------------------------------------------------------------

class TodayGamesInput(BaseModel):
    game_date: str = Field(..., description="A date in 'YYYY-MM-DD' format.")
    league_id: str = Field(default="00", description="League ID (default=00 for NBA).")

@mcp.tool()
def nba_list_todays_games(game_date: str, league_id: str = "00") -> Dict[str, Any]:
    """Returns scoreboard data from stats.nba.com for a given date.

    This tool retrieves game information for a specific date from the
    `nba_api.stats.endpoints.scoreboardv2` endpoint.  It's similar to `nba_live_scoreboard`,
    but it allows you to query for games on *any* date (past, present, or future), not just
    today's games.

    **Args:**

        game_date (str): The date for which to retrieve game information, in "YYYY-MM-DD" format.
            Example: "2023-12-25"
        league_id (str, optional):  The league ID.  "00" represents the NBA. Defaults to "00".

    **Returns:**

        Dict[str, Any]: A dictionary containing game data for the specified date.  The structure
            follows the `nba_api`'s `ScoreboardV2` object (but is normalized). Key elements:

            * "GameHeader": A list of dictionaries, one for each game, containing:
                * "GAME_DATE_EST":  The game date in YYYY-MM-DD format.
                * "GAME_ID": The 10-digit game ID. **Important for other tools.**
                * "HOME_TEAM_ID": The ID of the home team.
                * "VISITOR_TEAM_ID": The ID of the away team.
                * "GAME_STATUS_TEXT": Textual game status (e.g., "Final", "8:00 PM ET").
                * ...and other fields.
            * "LineScore": A list of dictionaries with detailed scoring information for each team
               in each game.
            *  "SeriesStandings": A list of series standings
            *   "LastMeeting": A list of last meeting
            * ... other fields

            If an error occurs, the dictionary will contain a single "error" key.

    """
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date, league_id=league_id)
        return sb.get_normalized_dict()
    except Exception as e:
        return {"error": str(e)}

# # -------------------------------------------------------------------
# # 10) TeamGameLogsTool: Fetch a Team's Game Logs
# # -------------------------------------------------------------------

# class TeamGameLogsInput(BaseModel):
#     team_id: str = Field(..., description="The NBA Team ID.")
#     season: str = Field(default="2022-23", description="Season in 'YYYY-YY' format.")
#     season_type: str = Field(default="Regular Season", description="'Regular Season', 'Playoffs', etc.")

# @mcp.tool()
# def nba_team_game_logs(team_id: str, season: str, season_type: str) -> List[Dict[str, Any]]:
#     """Fetch a list of all games for a given Team ID in a specified season."""
#     try:
#         logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season, season_type_nullable=season_type)
#         df = logs.get_data_frames()[0]
#         selected_columns = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
#         partial_df = df[selected_columns]
#         return partial_df.to_dict("records")
#     except Exception as e:
#         return [{"error": str(e)}]

# -------------------------------------------------------------------
# 11) team_game_logs_by_name_tool: Fetch a Team's Game Logs by Name
# -------------------------------------------------------------------

class TeamGameLogsByNameInput(BaseModel):
    team_name: str = Field(..., description="Partial or full NBA team name.")
    season: str = Field(default="2022-23", description="Season in 'YYYY-YY' format.")
    season_type: str = Field(default="Regular Season", description="'Regular Season', 'Playoffs', etc.")

@mcp.tool()
def nba_team_game_logs_by_name(team_name: str, season: str, season_type: str) -> List[Dict[str, Any]]:
    """Fetch a team's game logs by providing the team name.

    This tool retrieves a team's game log (list of games) for a given season and season type,
    using the team's *name* as input.  This avoids needing to know the team's numeric ID.  It uses
    the `nba_api.stats.static.teams` module to find the team and then the
    `nba_api.stats.endpoints.teamgamelogs` endpoint to get the game log.

    **Args:**

        team_name (str): The full or partial name of the NBA team (e.g., "Lakers", "Los Angeles Lakers").
        season (str): The season in "YYYY-YY" format (e.g., "2023-24").
        season_type (str): The type of season.  Valid options are:
            * "Regular Season"
            * "Playoffs"
            * "Pre Season"
            * "All Star"

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a game in the
            team's game log.  The selected columns are:

            * "TEAM_ID": The team's numeric ID.
            * "GAME_ID": The 10-digit game ID.
            * "GAME_DATE": The date of the game.
            * "MATCHUP":  A string showing the matchup (e.g., "LAL vs. GSW").
            * "WL":  The game result ("W" for win, "L" for loss, or None if the game hasn't been played).

            If no team is found or an error occurs, the list will contain a single dictionary
            with an "error" key.

    """
    try:
        found = teams.find_teams_by_full_name(team_name)
        if not found:
            return [{"error": f"No NBA team found matching name '{team_name}'."}]
        best_match = found[0]
        team_id = best_match["id"]
        logs = teamgamelogs.TeamGameLogs(team_id_nullable=str(team_id), season_nullable=season, season_type_nullable=season_type)
        df = logs.get_data_frames()[0]
        columns_we_want = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
        partial_df = df[columns_we_want]
        return partial_df.to_dict("records")
    except Exception as e:
        return [{"error": str(e)}]

# -------------------------------------------------------------------
# 12) nba_fetch_game_results: Fetch Game Results for a Team
# -------------------------------------------------------------------
class GameResultsInput(BaseModel):

    team_id: str = Field(..., description="A valid NBA team ID.")
    dates: List[str] = Field(..., description="A list of dates in 'YYYY-MM-DD' format.", min_items=1)

@mcp.tool()
def nba_fetch_game_results(team_id: str, dates: List[str]) -> List[Dict[str, Any]]:
    """Fetch game results for a given NBA team ID and date range.

    This tool retrieves game results and statistics for a specified team within a given range of dates.
    It leverages the `nba_api.stats.endpoints.leaguegamefinder` to efficiently find games and then filters
    the results to include only the dates requested.

    **Args:**

        team_id (str): The NBA team ID (e.g., "1610612744" for the Golden State Warriors).
        dates (List[str]): A list of dates in "YYYY-MM-DD" format, representing the date range for which
                           to fetch game results. The order of dates does not matter; the function will
                           automatically determine the start and end dates.  Must contain at least one date.

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a game played by
            the specified team within the provided date range.  Includes all columns returned by
            the `nba_api`'s `LeagueGameFinder`.

            If an error occurs or no games are found, a list with a single dictionary containing an "error"
            key is returned.

    """
    # Convert player_id to a string if it's not already.
    if not isinstance(team_id, str):
        team_id = str(team_id)
        
    try:
        date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_type_nullable=SeasonType.regular,
            date_from_nullable=min(date_objects).strftime('%m/%d/%Y'),
            date_to_nullable=max(date_objects).strftime('%m/%d/%Y')
        )
        games = gamefinder.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        start_date = min(date_objects)
        end_date = max(date_objects)
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += timedelta(days=1)
        games = games[games['GAME_DATE'].dt.date.isin([d.date() for d in all_dates])]
        return games.to_dict('records')
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------------
# nba_team_standings: Retrieve NBA Team Standings
# -------------------------------------------------------------------------
class LeagueStandingsInput(BaseModel):
    season: str = Field(default=SeasonYear.default, description="The NBA season (e.g., '2023-24').")
    season_type: str = Field(default="Regular Season", description="The season type (e.g., 'Regular Season').")

@mcp.tool()
def nba_team_standings(season: str = SeasonYear.default, season_type: str = "Regular Season") -> List[Dict[str, Any]]:
    """Fetch the NBA team standings for a given season and season type.

    Retrieves team standings data from `nba_api.stats.endpoints.leaguestandingsv3`.  This includes
    wins, losses, win percentage, conference and division rankings, and other relevant information.

    **Args:**

        season (str, optional): The NBA season in "YYYY-YY" format (e.g., "2023-24"). Defaults to the
            current season as defined by `nba_api.stats.library.parameters.SeasonYear.default`.
        season_type (str, optional): The type of season. Valid options include:
            * "Regular Season"
            * "Playoffs"
            * "Pre Season"
            * "All Star"
            Defaults to "Regular Season".

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries, each representing a team's standing and
            associated statistics.  The structure is based on the `nba_api`'s `LeagueStandingsV3`
            data frame output. Includes fields like:

            * "TeamID": The team's ID.
            * "TeamCity": The team's city.
            * "TeamName": The team's name.
            * "Conference": The team's conference (e.g., "East", "West").
            * "ConferenceRecord": The team's record within its conference.
            * "PlayoffRank": The team's rank for playoff seeding within its conference.
            * "WINS": Number of wins.
            * "LOSSES": Number of losses.
            * "Win_PCT": Win percentage.
            * ...and many other statistical fields.
            If an error occurs, returns a list containing a single dictionary with an "error" key.
    """
    try:
        standings = leaguestandingsv3.LeagueStandingsV3(season=season, season_type=season_type)
        return standings.get_data_frames()[0].to_dict('records')
    except Exception as e:
        return [{"error": str(e)}]

# -------------------------------------------------------------------------
# nba_team_stats_by_name: Retrieve NBA Team Stats by Team Name
# -------------------------------------------------------------------------
class TeamStatsInput(BaseModel):
    team_name: str = Field(..., description="The NBA team name (e.g., 'Cleveland Cavaliers').")
    season_type: str = Field(default="Regular Season", description="The season type (e.g., 'Regular Season').")
    per_mode: str = Field(default="PerGame", description="Options are Totals, PerGame, Per48, Per40, etc.")

    @field_validator("team_name")
    def validate_team_name(cls, value):
        found_teams = teams.find_teams_by_full_name(value)
        if not found_teams:
            raise ValueError(f"No NBA team found with the name '{value}'.")
        return value

@mcp.tool()
def nba_team_stats_by_name(team_name: str, season_type: str = "Regular Season", per_mode: str = "PerGame") -> List[Dict[str, Any]]:
    """Fetches NBA team statistics from stats.nba.com using the team name.

    This tool retrieves detailed team statistics for a specified team, season type, and aggregation
    method.  It first uses `nba_api.stats.static.teams` to find the team ID based on the provided
    name, then uses `nba_api.stats.endpoints.teamyearbyyearstats` to get the statistics.

    **Args:**

        team_name (str): The full or partial name of the NBA team (e.g., "Celtics", "Boston Celtics").
                          This argument is validated to ensure a team with provided name exists.
        season_type (str, optional): The type of season. Valid options are:
            * "Regular Season"
            * "Playoffs"
            * "Pre Season"
             * "All Star"
            Defaults to "Regular Season".
        per_mode (str, optional):  Determines how the statistics are aggregated.  Valid options include:
            * "Totals":  Total season statistics.
            * "PerGame":  Stats averaged per game.
            * "Per48": Stats per 48 minutes.
            * "Per40": Stats per 40 minutes.
            * "Per36" : Stats per 36 minutes
            * ...and other per-minute options.
            Defaults to "PerGame".

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries. If the data for provided `team_name` is not found
            or is empty, this will contain a single error dictionary. Otherwise, each dictionary represents
            a season for the team and includes a wide range of statistics, based on the
            `nba_api`'s `TeamYearByYearStats` data frame output. Some key fields include:

            * "TEAM_ID": The team's ID.
            * "TEAM_CITY": The team's city.
            * "TEAM_NAME": The team's name.
            * "YEAR": The year of the season.
            * "WINS":, "LOSSES":, "Win_PCT": Basic win-loss information.
            * Numerous statistical fields (e.g., "PTS", "REB", "AST", "STL", "BLK", etc.)

    """
    try:
        found_teams = teams.find_teams_by_full_name(team_name)
        if not found_teams:
            return [{"error": f"No NBA team found with the name '{team_name}'."}]
        team_id = found_teams[0]['id']
        team_stats = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id, per_mode_simple=per_mode, season_type_all_star=season_type)
        team_stats_data = team_stats.get_data_frames()[0]
        if team_stats_data.empty:
            return [{"error": f"No stats found for {team_name},  season_type {season_type}."}]
        return team_stats_data.to_dict('records')
    except Exception as e:
        return [{"error": str(e)}]

# -------------------------------------------------------------------
# 15) nba_all_teams_stats: Retrieve NBA Team Stats for All Teams
# -------------------------------------------------------------------
class AllTeamsStatsInput(BaseModel):
    years: List[str] = Field(default=["2023"], description="A list of NBA season years (e.g., ['2022', '2023']).")
    season_type: str = Field(default="Regular Season", description="The season type (e.g., 'Regular Season').")

    @field_validator("years")
    def validate_years(cls, value):
        for year in value:
            if not year.isdigit() or len(year) != 4:
                raise ValueError("Each year must be a 4-digit string (e.g., '2023')")
        return value

@mcp.tool()
def nba_all_teams_stats(years: List[str] = ["2023"], season_type: str = "Regular Season") -> List[Dict[str, Any]]:
    """Fetch the NBA team statistics for all teams for a given list of season years and a season type.

    This tool retrieves comprehensive team statistics for *all* NBA teams across one or more seasons.
    It uses the `nba_api.stats.endpoints.leaguestandingsv3` endpoint to gather the data.  This is
    useful for comparing teams or tracking league-wide trends over time.

    **Args:**

        years (List[str], optional): A list of NBA season years in "YYYY" format (e.g., ["2022", "2023"]).
            Defaults to ["2023"].  Each year must be a 4-digit string.
        season_type (str, optional): The type of season. Valid options are:
            * "Regular Season"
            * "Playoffs"
            * "Pre Season"
            * "All Star"
            Defaults to "Regular Season".

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a team's
            statistics *for a specific season*.  The data includes a wide range of statistics,
            similar to `nba_team_standings`, but aggregated for all teams. Key fields:

            * "TeamID": The team's ID.
            * "TeamCity": The team's city.
            * "TeamName": The team's name.
            * "Conference": The team's conference.
            * "ConferenceRecord":  Record within the conference.
            * "WINS", "LOSSES", "Win_PCT": Win-loss statistics.
            * "Season": The year of the season (taken from the input `years`).
            * ...and many other statistical fields.
            If no data available, returns an error message.
    """
    all_seasons_stats = []
    try:
        for year in years:
            team_stats = leaguestandingsv3.LeagueStandingsV3(
                season=year,
                season_type=season_type,
                league_id='00',
            )
            team_stats_data = team_stats.get_data_frames()[0]
            if team_stats_data.empty:
                all_seasons_stats.append({"error": f"No stats found for season {year}, season_type {season_type}."})
                continue
            for col in ['PlayoffRank', 'ConferenceRank', 'DivisionRank', 'WINS', 'LOSSES', 'ConferenceGamesBack', 'DivisionGamesBack']:
                if col in team_stats_data.columns:
                    try:
                        team_stats_data[col] = pd.to_numeric(team_stats_data[col], errors='coerce')
                    except (ValueError, TypeError):
                        pass
            team_stats_data['Season'] = year
            all_seasons_stats.extend(team_stats_data.to_dict('records'))
        return all_seasons_stats
    except Exception as e:
        return [{"error": str(e)}]

# -------------------------------------------------------------------
# 16) nba_player_game_logs: Retrieve NBA Player Game Logs and stats
# -------------------------------------------------------------------
@mcp.tool()
def nba_player_game_logs(player_id: str, date_range: List[str], season_type: str = "Regular Season") -> List[Dict[str, Any]]:
    """Obtain an NBA player's game statistics for dates within a specified date range.

    This tool retrieves individual game statistics for a given player within a specific date range. It uses
    the `nba_api.stats.endpoints.leaguegamefinder` to find games played by the player and filters the
    results to include only games within the specified dates.

    **Args:**

        player_id (str): The NBA player ID (e.g., "2544" for LeBron James).
        date_range (List[str]): A list containing two strings representing the start and end dates
            of the desired range, in "YYYY-MM-DD" format. Example: ["2024-01-01", "2024-01-31"]
        season_type (str, optional): The type of season. Valid options are:
            * "Regular Season"
            * "Playoffs"
            * "Pre Season"
            * "All Star"
            Defaults to "Regular Season".

    **Returns:**

        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a game played
            by the specified player within the provided date range. Includes all columns returned
            by the underlying `nba_api` call, including detailed game statistics.  Key fields:

            *   "PLAYER_ID": The player's ID
            *   "PLAYER_NAME": The player's name.
            *   "TEAM_ID": The ID of the player's team.
            *   "TEAM_ABBREVIATION": Team abbreviation.
            *   "GAME_ID": The 10-digit Game ID.
            *   "GAME_DATE": The date of the game.
            *   "MATCHUP": Text showing the matchup.
            *   "WL": Win ('W') or Loss ('L')
            *   "MIN": Minutes played.
            *   ...and many other statistical fields (PTS, REB, AST, etc.).

            If no games are found or an error occurs, returns a list containing a single dictionary
            with an "error" key.
    """
    # Convert player_id to a string if it's not already.
    if not isinstance(player_id, str):
        player_id = str(player_id)
        
    try:
        start_date_str, end_date_str = date_range
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        gamefinder = leaguegamefinder.LeagueGameFinder(
            player_id_nullable=player_id,
            season_type_nullable=season_type,
            date_from_nullable=start_date.strftime('%m/%d/%Y'),
            date_to_nullable=end_date.strftime('%m/%d/%Y')
        )
        games = gamefinder.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += timedelta(days=1)
        games = games[games['GAME_DATE'].dt.date.isin([d.date() for d in all_dates])]
        return games.to_dict('records')
    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    try:
        print("Starting MCP server 'nba_mcp_server' on 127.0.0.1:5000")
        # Use this approach to keep the server running
        mcp.run()
    except Exception as e:
        print(f"Error: {e}")
        # Sleep before exiting to give time for error logs
        time.sleep(5)