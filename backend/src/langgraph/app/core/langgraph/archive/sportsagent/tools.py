"""This module provides example tools for for the LangChain platform.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast, Dict, Literal
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional, Dict, Any
from langchain.tools.base import StructuredTool
import os
from datetime import datetime, timedelta

import re
import pandas as pd 
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.tools import BaseTool, Tool
import mlbstatsapi
import requests
import logging
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import leaguestandingsv3
from nba_api.stats.library.parameters import SeasonTypeAllStar, SeasonYear, Season
from nba_api.stats.endpoints import teamyearbyyearstats
from nba_api.stats.static import teams



#---------------------------------------------------------------------
from app.react_agent.configuration import Configuration
#---------------------------------------------------------------------

load_dotenv()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# If you're using something like LangChain Tools, uncomment or adjust the import:
# from langchain.tools import Tool

# -------------------------------------------------------------------
# 1) Get MLB 2024 (or any) Regular Season Schedule
# -------------------------------------------------------------------

class MLBGetScheduleInput(BaseModel):
    """
    Input schema for fetching MLB schedule data.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/schedule
    """
    sportId: Optional[int] = Field(1, description="Sport ID for MLB is 1.")
    season: Optional[str] = Field("2024", description="The season year. Example: '2024'.")
    gameType: Optional[str] = Field("R", description="Game type. Examples: R (Regular), P (Postseason), S (Spring).")
    date: Optional[str] = Field(
        None, 
        description="Specific date in MM/DD/YYYY format to get the schedule for that day."
    )
    # Add additional parameters as needed (e.g., fields, hydrate, etc.)

class MLBGetScheduleTool:
    """
    A tool to call the MLB StatsAPI /schedule endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/schedule"
        # No API key required for MLB endpoints.

    def run_get_schedule(
        self,
        sportId: int = 1,
        season: str = "2024",
        gameType: str = "R",
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GET request to /schedule with optional query parameters.
        """
        params = {
            "sportId": sportId,
            "season": season,
            "gameType": gameType
        }
        if date:
            params["date"] = date

        try:
            resp = requests.get(self.base_url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_schedule_tool = StructuredTool(
    name="mlb_get_schedule",
    func=MLBGetScheduleTool().run_get_schedule,
    description="BASEBALL MLB: Calls the MLB StatsAPI to get the schedule for a given season, date, and game type.",
    args_schema=MLBGetScheduleInput
)

# -------------------------------------------------------------------
# 2) Get Team Roster
# -------------------------------------------------------------------

class MLBGetTeamRosterInput(BaseModel):
    """
    Input schema for fetching a specific team's roster.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/teams/{teamId}/roster
    """
    teamId: int = Field(..., description="Team ID. Example: 119 for LA Dodgers.")
    season: Optional[str] = Field(default = "2025", description="Season year. Example: '2024'.")

class MLBGetTeamRosterTool:
    """
    A tool to call the MLB StatsAPI /teams/{teamId}/roster endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/teams"

    def run_get_team_roster(self, teamId: int, season: str = "2024") -> Dict[str, Any]:
        """
        GET request to /teams/{teamId}/roster with optional season parameter.
        """
        url = f"{self.base_url}/{teamId}/roster"
        params = {
            "season": season
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_team_roster_tool = StructuredTool(
    name="mlb_get_team_roster",
    func=MLBGetTeamRosterTool().run_get_team_roster,
    description="BASEBALL MLB: Fetches a team's roster for a given season using the MLB StatsAPI.",
    args_schema=MLBGetTeamRosterInput
)

# -------------------------------------------------------------------
# 3) Get Team Information
# -------------------------------------------------------------------

class MLBGetTeamInfoInput(BaseModel):
    """
    Input schema for fetching detailed team info.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/teams/{teamId}
    """
    teamId: int = Field(..., description="Team ID. Example: 119 for LA Dodgers.")
    season: Optional[str] = Field(..., description="Season year. Example: '2024'.")

class MLBGetTeamInfoTool:
    """
    A tool to call the MLB StatsAPI /teams/{teamId} endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/teams"

    def run_get_team_info(self, teamId: int, season: Optional[str] = None) -> Dict[str, Any]:
        """
        GET request to /teams/{teamId} with optional season parameter.
        """
        url = f"{self.base_url}/{teamId}"
        params = {}
        if season:
            params["season"] = season

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_team_info_tool = StructuredTool(
    name="mlb_get_team_info",
    func=MLBGetTeamInfoTool().run_get_team_info,
    description="BASEBALL MLB: Fetches detailed information about a given MLB team from the StatsAPI.",
    args_schema=MLBGetTeamInfoInput
)

# -------------------------------------------------------------------
# 4) Get Player Information
# -------------------------------------------------------------------

class MLBGetPlayerInfoInput(BaseModel):
    """
    Input schema for fetching a specific player's info.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/people/{playerId}
    """
    playerId: int = Field(..., description="Player ID. Example: 660271 for Shohei Ohtani.")
    season: Optional[str] = Field(..., description="Season year. Example: '2024'.")

class MLBGetPlayerInfoTool:
    """
    A tool to call the MLB StatsAPI /people/{playerId} endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/people"

    def run_get_player_info(self, playerId: int, season: Optional[str] = None) -> Dict[str, Any]:
        """
        GET request to /people/{playerId} with optional season parameter.
        """
        url = f"{self.base_url}/{playerId}"
        params = {}
        if season:
            params["season"] = season

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_player_info_tool = StructuredTool(
    name="mlb_get_player_info",
    func=MLBGetPlayerInfoTool().run_get_player_info,
    description="BASEBALL MLB: Fetches detailed information about a specific MLB player.",
    args_schema=MLBGetPlayerInfoInput
)

# -------------------------------------------------------------------
# 5) Get Live Game Data (GUMBO Feed)
# -------------------------------------------------------------------

class MLBGetLiveGameDataInput(BaseModel):
    """
    Input schema for fetching GUMBO live feed (entire game state).
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
    """
    game_pk: int = Field(..., description="Game primary key (e.g., 716463).")

class MLBGetLiveGameDataTool:
    """
    A tool to call the MLB StatsAPI /game/{game_pk}/feed/live endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def run_get_live_game_data(self, game_pk: int) -> Dict[str, Any]:
        """
        GET request to /game/{game_pk}/feed/live to get the GUMBO feed for a specific game.
        """
        url = f"{self.base_url}/{game_pk}/feed/live"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_live_game_data_tool = StructuredTool(
    name="mlb_get_live_game_data",
    func=MLBGetLiveGameDataTool().run_get_live_game_data,
    description="BASEBALL MLB: Fetches the GUMBO live feed for a specified MLB game.",
    args_schema=MLBGetLiveGameDataInput
)

# -------------------------------------------------------------------
# 6) Get Game Timestamps
# -------------------------------------------------------------------

class MLBGetGameTimestampsInput(BaseModel):
    """
    Input schema for fetching the timestamps of GUMBO updates for a given game.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live/timestamps
    """
    game_pk: int = Field(..., description="Game primary key (e.g., 716463).")

class MLBGetGameTimestampsTool:
    """
    A tool to call the MLB StatsAPI /game/{game_pk}/feed/live/timestamps endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def run_get_game_timestamps(self, game_pk: int) -> Dict[str, Any]:
        """
        GET request to /game/{game_pk}/feed/live/timestamps for update timestamps.
        """
        url = f"{self.base_url}/{game_pk}/feed/live/timestamps"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_game_timestamps_tool = StructuredTool(
    name="mlb_get_game_timestamps",
    func=MLBGetGameTimestampsTool().run_get_game_timestamps,
    description="BASEBALL MLB: Fetches the list of GUMBO update timestamps for a given MLB game.",
    args_schema=MLBGetGameTimestampsInput
)

# game_data_tools = [mlb_get_schedule_tool, mlb_get_live_game_data_tool, mlb_get_game_timestamps_tool]
# team_tools = [mlb_get_team_roster_tool, mlb_get_team_info_tool]
# player_tools = [mlb_get_player_info_tool]

# -------------------------------------------------------------------
# End of Tools
# -------------------------------------------------------------------

# You now have six robust tools for common MLB StatsAPI queries:
# 1) mlb_get_schedule_tool
# 2) mlb_get_team_roster_tool
# 3) mlb_get_team_info_tool
# 4) mlb_get_player_info_tool
# 5) mlb_get_live_game_data_tool
# 6) mlb_get_game_timestamps_tool

# You can import and use them as needed in your project. 
# For example:
# result = mlb_get_schedule_tool.func(sportId=1, season="2024", gameType="R", date="03/28/2024")
# print(result)


# -------------------------------------------------------------------
# 7) Get Team ID From Team Name
# -------------------------------------------------------------------

class MLBGetTeamIdInput(BaseModel):
    """
    Input schema for retrieving MLB team ID(s) by a team name string.
    This uses mlb.get_team_id(team_name, search_key=...) under the hood.
    """
    team_name: str = Field(..., description="Full or partial team name, e.g. 'Oakland Athletics'.")
    search_key: Optional[str] = Field(
        "name",
        description="Which search field to match on; defaults to 'name'."
    )


class MLBGetTeamIdTool:
    """
    A tool that calls python-mlb-statsapi's Mlb.get_team_id().
    Returns a list of matching team IDs.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_team_id(self, team_name: str, search_key: str = "name") -> Dict[str, Any]:
        """
        Returns: A dict with the list of matching team IDs and a success/error message.
        """
        try:
            team_ids = self.client.get_team_id(team_name, search_key=search_key)
            return {
                "team_name": team_name,
                "matching_team_ids": team_ids
            }
        except Exception as e:
            return {"error": f"Unable to retrieve team ID(s): {str(e)}"}


mlb_get_team_id_tool = StructuredTool(
    name="mlb_get_team_id",
    func=MLBGetTeamIdTool().run_get_team_id,
    description="BASEBALL MLB: Get a list of MLB team ID(s) by providing a team name.",
    args_schema=MLBGetTeamIdInput
)



# -------------------------------------------------------------------
# 8) Get Player ID From Full Name
# -------------------------------------------------------------------

class MLBGetPlayerIdInput(BaseModel):
    """
    Input schema for retrieving MLB player ID(s) by a player name string.
    """
    player_name: str = Field(..., description="Player's name, e.g. 'Shohei Ohtani' or 'Ty France'.")
    sport_id: Optional[int] = Field(default = 1, description="Sport ID, defaults to 1 for MLB.")
    search_key: Optional[str] = Field(
        default = "fullname",
        description="Which search field to match on; typically 'fullname'."
    )


class MLBGetPlayerIdTool:
    """
    A tool that calls python-mlb-statsapi's Mlb.get_people_id().
    Returns a list of matching player IDs.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_player_id(
        self,
        player_name: str,
        sport_id: int = 1,
        search_key: str = "fullname"
    ) -> Dict[str, Any]:
        try:
            player_ids = self.client.get_people_id(
                fullname=player_name,
                sport_id=sport_id,
                search_key=search_key
            )
            # return {
            #     "player_name": player_name,
            #     "matching_player_ids": player_ids
            # }
    
            if player_ids:
                return f"Player: {player_name}, Matching Player IDs: {', '.join(map(str, player_ids))}"
            else:
                return f"No matching player IDs found for: {player_name}"
            
        except Exception as e:
            return {"error": f"Unable to retrieve player ID(s): {str(e)}"}


mlb_get_player_id_tool = StructuredTool(
    name="mlb_get_player_id",
    func=MLBGetPlayerIdTool().run_get_player_id,
    description="BASEBALL MLB: Get a list of MLB player IDs by providing a full player name.",
    args_schema=MLBGetPlayerIdInput
)



from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import mlbstatsapi

    
# -------------------------------------------------------------------
# 10) Get Game PK (IDs) By Date
# -------------------------------------------------------------------


class MLBGetGameIdsByDateInput(BaseModel):
    """
    Input schema to retrieve a list of game_pk IDs for a given date.
    """
    date: str = Field(..., description="Date in YYYY-MM-DD format.")
    sport_id: Optional[int] = Field(default = 1, description="Sport ID for MLB is 1.")
    team_id: Optional[int] = Field(..., description="Filter by a specific team's ID if desired.")


class MLBGetGameIdsByDateTool:
    """
    A tool that calls python-mlb-statsapi's Mlb.get_scheduled_games_by_date().
    Returns a list of game IDs for the given date (and optional team).
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_game_ids_by_date(
        self,
        date: str,
        sport_id: int = 1,
        team_id: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            # This method returns a list of game IDs. If no games found, might be an empty list.
            game_ids = self.client.get_scheduled_games_by_date(
                date=date,
                sport_id=sport_id,
                team_id=team_id
            )
            return {
                "requested_date": date,
                "sport_id": sport_id,
                "team_id": team_id,
                "game_ids": game_ids
            }
        except Exception as e:
            return {"error": f"Unable to retrieve game IDs: {str(e)}"}


mlb_get_game_ids_by_date_tool = StructuredTool(
    name="mlb_get_game_ids_by_date",
    func=MLBGetGameIdsByDateTool().run_get_game_ids_by_date,
    description="BASEBALL MLB: Get a list of MLB game_pk (IDs) scheduled on a specific date using python-mlb-statsapi.",
    args_schema=MLBGetGameIdsByDateInput
)




# -------------------------------------------------------------------
# 11) Get a Single Game’s “PK” by Searching Team & Date
# -------------------------------------------------------------------

class MLBFindOneGameIdInput(BaseModel):
    """
    Input schema to find the first game PK matching a team on a certain date.
    """
    date: str = Field(..., description="Date in YYYY-MM-DD format.")
    team_name: str = Field(..., description="Team name, e.g. 'Seattle Mariners'.")


class MLBFindOneGameIdTool:
    """
    A tool that:
      1) Gets the team_id from the name (using get_team_id).
      2) Then calls get_scheduled_games_by_date(date=..., team_id=TEAM_ID).
      3) Returns the first found game_pk or all of them if you prefer.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_find_one_game_id(self, date: str, team_name: str) -> Dict[str, Any]:
        try:
            # 1) Find the team_id
            team_ids = self.client.get_team_id(team_name)
            if not team_ids:
                return {"error": f"No team ID found for '{team_name}'."}
            team_id = team_ids[0]

            # 2) Grab the game IDs for that date/team
            game_ids = self.client.get_scheduled_games_by_date(
                date=date,
                sport_id=1,
                team_id=team_id
            )

            if not game_ids:
                return {
                    "date": date,
                    "team_name": team_name,
                    "error": "No games found for this date/team."
                }

            # For demonstration: just return the first game
            return {
                "date": date,
                "team_id": team_id,
                "found_game_ids": game_ids,
                "first_game_id": game_ids[0]
            }

        except Exception as e:
            return {"error": f"Unable to find game ID: {str(e)}"}


mlb_find_one_game_id_tool = StructuredTool(
    name="mlb_find_one_game_id",
    func=MLBFindOneGameIdTool().run_find_one_game_id,
    description="BASEBALL MLB: Search for the first MLB game_pk on a given date for a given team name.",
    args_schema=MLBFindOneGameIdInput
)




# -------------------------------------------------------------------
# 12) Get Venue ID By Name
# -------------------------------------------------------------------

class MLBGetVenueIdInput(BaseModel):
    venue_name: str = Field(..., description="Venue name, e.g. 'PNC Park' or 'Wrigley Field'.")
    search_key: Optional[str] = Field(default = "name", description="Search field to match on.")


class MLBGetVenueIdTool:
    """
    A tool to call Mlb.get_venue_id(...), returning a list of matching venue IDs.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_venue_id(self, venue_name: str, search_key: str = "name") -> Dict[str, Any]:
        try:
            venue_ids = self.client.get_venue_id(venue_name, search_key=search_key)
            return {
                "venue_name": venue_name,
                "matching_venue_ids": venue_ids
            }
        except Exception as e:
            return {"error": f"Unable to retrieve venue ID(s): {str(e)}"}


mlb_get_venue_id_tool = StructuredTool(
    name="mlb_get_venue_id",
    func=MLBGetVenueIdTool().run_get_venue_id,
    description="BASEBALL MLB: Get a list of venue IDs for a stadium name (e.g. 'Wrigley Field').",
    args_schema=MLBGetVenueIdInput
)



# -------------------------------------------------------------------
# 13) Tavily Search Tool
# -------------------------------------------------------------------
# Define Input Schema# Define Input Schema
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

# Define the Tool
class TavilySearchTool:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a web search using the Tavily search engine.
        """
        try:
            # Initialize the Tavily search tool with the configured max_results
            search_tool = TavilySearchResults(max_results=self.max_results)

            # Perform the search
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

# Create the LangChain Tool
tavily_search_tool = StructuredTool(
    name="tavily_search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)


# -------------------------------------------------------------------
# ID Lookup Tools
# -------------------------------------------------------------------
# team_id_lookup_tools = [mlb_get_team_id_tool], 
team_tools = [mlb_get_team_id_tool, mlb_get_team_roster_tool, mlb_get_team_info_tool]
player_tools = [mlb_get_player_id_tool, mlb_get_player_info_tool, tavily_search_tool]

# game_id_lookup_tools = [mlb_get_game_ids_by_date_tool, mlb_find_one_game_id_tool, mlb_get_venue_id_tool, tavily_search_tool]
# game_data_tools = [mlb_get_game_ids_by_date_tool, mlb_get_schedule_tool, mlb_get_live_game_data_tool, mlb_get_game_timestamps_tool, tavily_search_tool]

game_info_tools = [mlb_get_game_ids_by_date_tool, mlb_find_one_game_id_tool, tavily_search_tool]
game_data_tools = [mlb_get_game_ids_by_date_tool, mlb_get_schedule_tool, mlb_get_live_game_data_tool]




# ---------------------------------------------------- NBA TOOLS ------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------
# 1) ScoreBoard Tool (Live Endpoint)
# -------------------------------------------------------------------
# Retrieves today’s scoreboard data from the live endpoint.

from langchain.tools import StructuredTool
from nba_api.live.nba.endpoints import scoreboard

# ========== 1) Define Input Schema ==========
class LiveScoreBoardInput(BaseModel):
    """
    Schema for fetching the current day scoreboard (live games).
    No extra parameters for scoreboard, but you can add filters if needed.
    """
    dummy_param: Optional[str] = Field(
        default="",
        description="Not used, but placeholder for expansions if needed."
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchScoreBoardTool:
    """
    Fetch today's scoreboard from the NBA Live endpoint.
    """
    def __init__(self):
        pass  # Any initial config can go here if needed

    def run(self, dummy_param: Optional[str] = "") -> Dict[str, Any]:
        """
        Gets the scoreboard data for today's NBA games.
        Returns it as a dictionary.
        """
        try:
            sb = scoreboard.ScoreBoard()  # Instantiate scoreboard
            data_dict = sb.get_dict()     # Dictionary of scoreboard data
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_live_scoreboard = StructuredTool(
    name="nba_live_scoreboard",
    description=(
        "BASKETBALL NBA:"
        "Fetch today's NBA scoreboard (live or latest). "
        "Useful for retrieving the current day's games, scores, period, status, etc."
    ),
    func=NBAFetchScoreBoardTool().run,
    args_schema=LiveScoreBoardInput
)


# -------------------------------------------------------------------
# 2) BoxScore Tool (Live Endpoint)
# -------------------------------------------------------------------
# Given a valid NBA game_id, retrieve the real-time box score from the live endpoint.

from nba_api.live.nba.endpoints import boxscore

# ========== 1) Define Input Schema ==========
class LiveBoxScoreInput(BaseModel):
    """
    Schema for fetching box score data using live/nba/endpoints/boxscore.
    """
    game_id: str = Field(
        ...,
        description="A 10-digit NBA game ID (e.g., '0022200017')."
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchBoxScoreTool:
    """
    Fetches a real-time box score for a given game ID from NBA Live endpoints.
    """
    def __init__(self):
        pass

    def run(self, game_id: str) -> Dict[str, Any]:
        """
        Return the box score as a dictionary.
        """
        try:
            bs = boxscore.BoxScore(game_id=game_id)
            data_dict = bs.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_live_boxscore = StructuredTool(
    name="nba_live_boxscore",
    description=(
        "BASKETBALL NBA: "
        "Fetch the real-time (live) box score for a given NBA game ID. "
        "Provides scoring, stats, team info, and player data."
    ),
    func=NBAFetchBoxScoreTool().run,
    args_schema=LiveBoxScoreInput
)



# -------------------------------------------------------------------
# 3) PlayByPlay Tool (Live Endpoint)
# -------------------------------------------------------------------
# Pulls the real-time play-by-play feed for a given game_id.

from nba_api.live.nba.endpoints import playbyplay

# ========== 1) Define Input Schema ==========
class LivePlayByPlayInput(BaseModel):
    """
    Schema for live PlayByPlay data retrieval.
    """
    game_id: str = Field(
        ...,
        description="A 10-digit NBA game ID for which to fetch play-by-play actions."
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchPlayByPlayTool:
    """
    Fetch real-time play-by-play data from the NBA Live endpoint for the given game ID.
    """
    def __init__(self):
        pass

    def run(self, game_id: str) -> Dict[str, Any]:
        """
        Return the play-by-play feed as a dictionary.
        """
        try:
            pbp = playbyplay.PlayByPlay(game_id=game_id)
            data_dict = pbp.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_live_play_by_play = StructuredTool(
    name="nba_live_play_by_play",
    description=(
        "BASKETBALL NBA: "
        "Retrieve the live play-by-play actions for a specific NBA game ID. "
        "Useful for real-time game event tracking."
    ),
    func=NBAFetchPlayByPlayTool().run,
    args_schema=LivePlayByPlayInput
)

# -------------------------------------------------------------------
# 4) CommonPlayerInfo Tool (Stats Endpoint)
# -------------------------------------------------------------------
# Retrieve standard information about an NBA player (e.g., birthdate, height, years of experience).
from nba_api.stats.endpoints import commonplayerinfo

# ========== 1) Define Input Schema ==========
class CommonPlayerInfoInput(BaseModel):
    """
    Pydantic schema for requesting common player info from stats.nba.com
    """
    player_id: str = Field(
        ...,
        description="NBA player ID (e.g., '2544' for LeBron James)."
    )


# ========== 2) Define the Tool Class ==========
class NBACommonPlayerInfoTool:
    """
    Retrieve player's basic profile data from the stats.nba.com endpoint.
    """
    def __init__(self):
        pass

    def run(self, player_id: str, league_id: str = "00") -> Dict[str, Any]:
        """
        Return data as dictionary, including personal info, stats, etc.
        """
        try:
            info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id
            )
            data_dict = info.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_common_player_info = StructuredTool(
    name="nba_common_player_info",
    description=(
        "BASKETBALL NBA: "
        "Retrieve basic information about a player (height, weight, birthdate, "
        "team, experience, etc.) from NBA stats endpoints."
    ),
    func=NBACommonPlayerInfoTool().run,
    args_schema=CommonPlayerInfoInput
)


# -------------------------------------------------------------------
# 5) PlayerCareerStats Tool (Stats Endpoint)
# -------------------------------------------------------------------
# Retrieves career stats for a given player (split by season and possibly by team).

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from nba_api.stats.endpoints import playercareerstats

# ========== 1) Define Input Schema ==========
class PlayerCareerStatsInput(BaseModel):
    """
    Schema for retrieving a player's aggregated career stats.
    """
    player_id: str = Field(
        ...,
        description="NBA player ID (e.g., '203999' for Nikola Jokic)."
    )
    per_mode: Optional[str] = Field(
        default="PerGame",
        description="One of 'Totals', 'PerGame', 'Per36', etc."
    )

# ========== 2) Define the Tool Class ==========
class NBAPlayerCareerStatsTool:
    """
    Pull aggregated career stats (regular season & playoff) for an NBA player from stats.nba.com.
    """
    def __init__(self):
        pass

    def run(self, player_id: str, per_mode: str = "PerGame") -> Dict[str, Any]:
        """
        Returns a dictionary containing the player's career data.
        """
        try:
            career = playercareerstats.PlayerCareerStats(
                player_id=player_id,
                per_mode36=per_mode  # param name is per_mode36 in the library
            )
            data_dict = career.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_player_career_stats = StructuredTool(
    name="nba_player_career_stats",
    description=(
        "BASKETBALL NBA: "
        "Obtain an NBA player's career statistics (regular season, playoffs, etc.) "
        "from the stats.nba.com endpoints. Usage requires a valid player_id."
    ),
    func=NBAPlayerCareerStatsTool().run,
    args_schema=PlayerCareerStatsInput
)


# -------------------------------------------------------------------
# 6) Search Players by Name
# -------------------------------------------------------------------
from nba_api.stats.static import players

# ========== 1) Define Input Schema ==========
class SearchPlayersByNameInput(BaseModel):
    name_query: str = Field(
        ...,
        description="Full or partial name of the player to look up (e.g. 'LeBron', 'Curry', 'James')."
    )

# ========== 2) Define the Tool Class ==========
class NBAPlayerSearchTool:
    """
    Searches NBA players by name (case-insensitive) using the static library in nba_api.
    Returns a list of matches with IDs, full names, etc.
    """
    def __init__(self):
        pass

    def run(self, name_query: str) -> List[Dict[str, Any]]:
        """
        Returns a list of player dicts: 
        [
          {
            'id': <player_id>,
            'full_name': 'FirstName LastName',
            'first_name': ...,
            'last_name': ...,
            'is_active': ...
          }, 
          ...
        ]
        """
        try:
            results = players.find_players_by_full_name(name_query)
            return results
        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
nba_search_players = StructuredTool(
    name="nba_search_players",
    description=(
        "BASKETBALL NBA: "
        "Search NBA players by partial or full name. "
        "Returns a list of matches with 'id' fields which can be used as 'player_id'."
    ),
    func=NBAPlayerSearchTool().run,
    args_schema=SearchPlayersByNameInput
)


# -------------------------------------------------------------------
# 7) Search Teams by Name
# -------------------------------------------------------------------
from nba_api.stats.static import teams

# ========== 1) Define Input Schema ==========
class SearchTeamsByNameInput(BaseModel):
    name_query: str = Field(
        ...,
        description="Full or partial team name (e.g. 'Lakers', 'Cavaliers')."
    )

# ========== 2) Define the Tool Class ==========
class NBATeamSearchTool:
    """
    Searches NBA teams by partial or full name using the static library in nba_api.
    """
    def __init__(self):
        pass

    def run(self, name_query: str) -> List[Dict[str, Any]]:
        """
        Returns a list of team dicts:
        [
          {
            'id': <team_id>,
            'full_name': 'Los Angeles Lakers',
            'abbreviation': 'LAL',
            'nickname': 'Lakers',
            'city': 'Los Angeles',
            'state': 'California',
            'year_founded': 1948
          },
          ...
        ]
        """
        try:
            results = teams.find_teams_by_full_name(name_query)
            return results
        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
nba_search_teams = StructuredTool(
    name="nba_search_teams",
    description=(
        "BASKETBALL NBA: "
        "Search NBA teams by partial or full name. "
        "Returns a list of matches with 'id' used as 'team_id'."
    ),
    func=NBATeamSearchTool().run,
    args_schema=SearchTeamsByNameInput
)


# -------------------------------------------------------------------
# 8) List All Active Players
# -------------------------------------------------------------------
from nba_api.stats.static import players

# ========== 1) Define Input Schema ==========
class ListActivePlayersInput(BaseModel):
    # no arguments needed
    dummy: str = "unused"

# ========== 2) Define the Tool Class ==========
class NBAListActivePlayersTool:
    """
    Lists all active NBA players as a big dictionary list, 
    each containing 'id', 'full_name', 'is_active', etc.
    """
    def __init__(self):
        pass

    def run(self, dummy: str = "") -> List[Dict[str, Any]]:
        try:
            all_active = players.get_active_players()
            return all_active
        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
nba_list_active_players = StructuredTool(
    name="nba_list_active_players",
    description=(
        "BASKETBALL NBA: "
        "Return a list of all currently active NBA players with their IDs and names. "
        "No input needed."
    ),
    func=NBAListActivePlayersTool().run,
    args_schema=ListActivePlayersInput
)


# -------------------------------------------------------------------
# 9) List Today’s Games (Stats vs. Live)
# -------------------------------------------------------------------
from nba_api.stats.endpoints import scoreboardv2

# ========== 1) Define Input Schema ==========
class TodayGamesInput(BaseModel):
    game_date: str = Field(
        ...,
        description="A date in 'YYYY-MM-DD' format to fetch scheduled or completed games."
    )

    league_id: str = Field(
        default="00",
        description="League ID (default=00 for NBA)."
    )

# ========== 2) Define the Tool Class ==========
class NBATodayGamesTool:
    """
    Lists the scoreboard from stats.nba.com for a given date, returning the games data set.
    """
    def __init__(self):
        pass

    def run(self, game_date: str, league_id: str = "00") -> Dict[str, Any]:
        """
        Return scoreboard details as a dictionary. 
        Typically you can find 'GAME_ID' in the 'GameHeader' dataset.
        """
        try:
            sb = scoreboardv2.ScoreboardV2(
                game_date=game_date,
                league_id=league_id,
            )
            data_dict = sb.get_normalized_dict()  # or .get_dict() if you prefer raw structure
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_list_todays_games = StructuredTool(
    name="nba_list_todays_games",
    description=(
        "BASKETBALL NBA: "
        "Returns scoreboard data from stats.nba.com for a given date (YYYY-MM-DD), "
        "including the game IDs, matchups, status, etc."
    ),
    func=NBATodayGamesTool().run,
    args_schema=TodayGamesInput
)



# -------------------------------------------------------------------
# 10) TeamGameLogsTool: Fetch a Team's Game Logs
# -------------------------------------------------------------------
from nba_api.stats.endpoints import teamgamelogs

# 1) Define Input Schema
class TeamGameLogsInput(BaseModel):
    """
    Tool input for fetching a team's game logs (and thus their game IDs).
    """
    team_id: str = Field(
        ...,
        description=(
            "The NBA Team ID (e.g. '1610612739' for Cleveland Cavaliers). "
            "Use other search tools or static data to find this ID."
        )
    )
    season: str = Field(
        default="2022-23",
        description=(
            "Season in 'YYYY-YY' format (e.g. '2022-23')."
        )
    )
    season_type: str = Field(
        default="Regular Season",
        description=(
            "One of 'Regular Season', 'Pre Season', 'Playoffs', or 'All Star'. "
            "Typically 'Regular Season'."
        )
    )

# 2) Define the Tool Class
class TeamGameLogsTool:
    """
    Fetches all game logs for a specific team in a certain season 
    using the `teamgamelogs.TeamGameLogs` endpoint from stats.nba.com.
    """
    def __init__(self):
        pass

    def run(self, team_id: str, season: str, season_type: str) -> List[Dict[str, Any]]:
        """
        Calls teamgamelogs.TeamGameLogs(...) and returns a simplified list 
        of dictionaries containing at least the 'GAME_ID' and other fields 
        like MATCHUP, GAME_DATE, W/L, etc.
        """
        try:
            # Use the TeamGameLogs endpoint
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable=season_type
            )
            # get_data_frames() returns a list of DataFrames. The main one is index=0
            df = logs.get_data_frames()[0]  # the primary DataFrame with all logs

            # Convert to dict. We'll select a few columns that matter for game identification
            # Feel free to keep or drop whichever columns you want.
            selected_columns = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
            partial_df = df[selected_columns]

            # Convert to list of dict
            results = partial_df.to_dict("records")
            return results
        except Exception as e:
            # Return a list with an error
            return [{"error": str(e)}]

# 3) Create the LangChain StructuredTool
nba_team_game_logs = StructuredTool(
    name="nba_team_game_logs",
    description=(
        "BASKETBALL NBA: "
        "Fetch a list of all games (including game IDs, date, matchup, result) "
        "for a given Team ID in a specified season and season type. "
        "Useful to find all the game_ids a team played, from which you can pick a certain matchup."
    ),
    func=TeamGameLogsTool().run,
    args_schema=TeamGameLogsInput
)

# -------------------------------------------------------------------
# 11) team_game_logs_by_name_tool: Fetch a Team's Game Logs by Name
# -------------------------------------------------------------------

from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelogs

# 1) Define Input Schema
class TeamGameLogsByNameInput(BaseModel):
    """
    User provides:
    - team_name: partial or full name for an NBA team (e.g. "Warriors", "Golden State Warriors")
    - season: e.g. "2022-23"
    - season_type: "Regular Season", "Playoffs", "Pre Season", or "All Star"
    """
    team_name: str = Field(
        ...,
        description="Partial or full NBA team name (e.g. 'Warriors', 'Cavaliers')."
    )
    season: str = Field(
        default="2022-23",
        description="Season in YYYY-YY format (e.g. '2022-23')."
    )
    season_type: str = Field(
        default="Regular Season",
        description="One of 'Regular Season', 'Playoffs', 'Pre Season', or 'All Star'."
    )

# 2) Define the Tool Class
class TeamGameLogsByNameTool:
    """
    Single tool that:
      1. Finds the best match for the given team name.
      2. Retrieves that team's ID.
      3. Calls 'teamgamelogs.TeamGameLogs' to fetch the logs (GAME_ID, MATCHUP, etc.).
    """
    def __init__(self):
        pass

    def run(self, team_name: str, season: str, season_type: str) -> List[Dict[str, Any]]:
        try:
            # A) Search teams by name
            found = teams.find_teams_by_full_name(team_name)

            if not found:
                return [{
                    "error": f"No NBA team found matching name '{team_name}'."
                }]
            elif len(found) > 1:
                # If you want to handle multiple matches differently, do so here.
                # Example: pick the first
                best_match = found[0]
            else:
                best_match = found[0]

            # B) Extract the team_id from best_match
            team_id = best_match["id"]  # e.g. 1610612744 for Golden State

            # C) Get the game logs from teamgamelogs
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=str(team_id),
                season_nullable=season,
                season_type_nullable=season_type
            )

            df = logs.get_data_frames()[0]
            # We'll pick out some columns for clarity
            columns_we_want = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
            partial_df = df[columns_we_want]
            results = partial_df.to_dict("records")

            return results
        except Exception as e:
            return [{"error": str(e)}]

# 3) Create the LangChain StructuredTool
nba_team_game_logs_by_name = StructuredTool(
    name="nba_team_game_logs_by_name",
    description=(
        "BASKETBALL NBA: "
        "Fetch a team's game logs (and thus game_ids) by providing the team name, "
        "without needing the numeric team_id directly. Returns a list of dictionaries "
        "with 'GAME_ID', 'GAME_DATE', 'MATCHUP', and 'WL'."
    ),
    func=TeamGameLogsByNameTool().run,
    args_schema=TeamGameLogsByNameInput
)

# ---------------------------------------------------------------- Here------------------------------------------
# --------------------------------------------------
# 12) nba_fetch_game_results: Fetch Game Results for a Team
# --------------------------------------------------
# ========== 1) Define Input Schema ==========
class GameResultsInput(BaseModel):
    """
    Schema for fetching game results for a given team and date range.
    """
    team_id: str = Field(
        ...,
        description="A valid NBA team ID (e.g., '1610612740')."
    )
    dates: List[str] = Field(
        ...,
        description="A list of one or more dates in the format 'YYYY-MM-DD' (e.g., ['2023-01-01', '2023-01-02']).",
        min_items=1
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchGameResultsTool:
    """
    Fetches game results for a given team and date range.
    """
    def __init__(self):
        pass

    def run(self, team_id: str, dates: List[str]) -> List[Dict[str, Any]]:
        """
        Return the game results as a list of dictionaries.
        """
        try:
            # Convert dates to datetime objects
            date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

            # Find games for the given team and date range
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_type_nullable=SeasonType.regular,
                date_from_nullable=min(date_objects).strftime('%m/%d/%Y'),
                date_to_nullable=max(date_objects).strftime('%m/%d/%Y')
            )

            games = gamefinder.get_data_frames()[0]

            # Filter games by the given dates
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            # 1. Find the start and end dates
            start_date = min(date_objects)  # The earliest date
            end_date = max(date_objects)    # The latest date

            # 2. Generate the list of dates
            all_dates = []
            current_date = start_date
            while current_date <= end_date:
                all_dates.append(current_date)
                current_date += timedelta(days=1)  # Increment by one day

            # 3.  Correctly create a list of dates to filter on
            games = games[games['GAME_DATE'].dt.date.isin([d.date() for d in all_dates])]

            # Return game results as a list of dictionaries
            return games.to_dict('records')
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
nba_fetch_game_results = StructuredTool(
    name="nba_fetch_game_results",
    description=(
        "BASKETBALL NBA: "
        "Fetch game results for a given NBA team ID and date range. "
        "Provides game stats and results."
    ),
    func=NBAFetchGameResultsTool().run,
    args_schema=GameResultsInput
)


# -------------------------------------------------------------------------
# nba_team_standings: Retrieve NBA Team Standings
# -------------------------------------------------------------------------
class LeagueStandingsInput(BaseModel):
    season: str = Field(
        default=SeasonYear.default,
        description="The NBA season (e.g., '2023-24'). Defaults to the current season."
    )
    season_type: str = Field(
        default="Regular Season",
        description="The season type (e.g., 'Regular Season', 'Playoffs', 'Pre Season', 'All Star'). Defaults to 'Regular Season'."
    )

# ========== 2) Define the Tool Class ==========
class NBATeamStandingsTool:
    """
    Fetches NBA team standings from stats.nba.com.
    """
    def __init__(self):
        pass

    def run(self, season: str = SeasonYear.default, season_type: str = "Regular Season") -> List[Dict[str, Any]]:
        """
        Returns the NBA team standings as a list of dictionaries.
        """
        try:
            # Fetch standings data
            standings = leaguestandingsv3.LeagueStandingsV3(
                season=season,
                season_type=season_type
            )
            standings_data = standings.get_data_frames()[0]

            # Convert the DataFrame to a list of dictionaries
            return standings_data.to_dict('records')

        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
nba_team_standings = StructuredTool(
    name="nba_team_standings",
    description=(
        "BASKETBALL NBA: "
        "Fetch the NBA team standings for a given season and season type. "
        "Returns a list of teams with their standings and basic stats."
    ),
    func=NBATeamStandingsTool().run,
    args_schema=LeagueStandingsInput # Use the defined input schema
)


# -------------------------------------------------------------------------
# nba_team_stats_by_name: Retrieve NBA Team Stats by Team Name
# -------------------------------------------------------------------------
class TeamStatsInput(BaseModel):
    team_name: str = Field(
        ...,
        description="The NBA team name (e.g., 'Cleveland Cavaliers')."
    )
    season_type: str = Field(
        default="Regular Season",
        description="The season type (e.g., 'Regular Season', 'Playoffs', 'Pre Season', 'All Star'). Defaults to 'Regular Season'."
    )
    per_mode: str = Field(
        default="PerGame",
        description="Options are Totals, PerGame, Per48, Per40, PerMinute, PerPossession, MinutesPer, Rank"
    )

    @field_validator("team_name")
    def validate_team_name(cls, value):
        # Basic validation: check if team name exists
        found_teams = teams.find_teams_by_full_name(value)
        if not found_teams:
            raise ValueError(f"No NBA team found with the name '{value}'.")
        return value

# ========== 2) Define the Tool Class ==========
class NBATeamStatsByNameTool:
    """
    Fetches NBA team statistics from stats.nba.com using the team name.
    """
    def __init__(self):
        pass

    def run(self, team_name: str, season_type: str = "Regular Season", per_mode: str = "PerGame") -> List[Dict[str, Any]]:  # Corrected: Use string defaults
        """
        Returns the NBA team statistics as a list of dictionaries.
        """
        try:
            # 1. Find the team's ID based on the name
            found_teams = teams.find_teams_by_full_name(team_name)
            if not found_teams:
                return [{"error": f"No NBA team found with the name '{team_name}'."}]

            # Handle multiple matches (though unlikely with full names). Take the first.
            team_id = found_teams[0]['id']

            # 2. Fetch team stats data using the team ID
            # Corrected: Pass parameters individually, not as a dictionary
            team_stats = teamyearbyyearstats.TeamYearByYearStats(
                team_id=team_id,
                per_mode_simple=per_mode,
                season_type_all_star=season_type,
            )


            team_stats_data = team_stats.get_data_frames()[0]

            # 3. Check if the DataFrame is empty
            if team_stats_data.empty:
                return [{"error": f"No stats found for {team_name},  season_type {season_type}."}]

            # 4. Convert the DataFrame to a list of dictionaries
            return team_stats_data.to_dict('records')

        except Exception as e:
            return [{"error": str(e)}]


# ========== 3) Create the LangChain StructuredTool ==========
nba_team_stats_by_name = StructuredTool(
    name="nba_team_stats_by_name",
    description=(
        "BASKETBALL NBA: "
        "Fetch the NBA team statistics for a given team name, season type, and per mode."
        " Returns a list of statistics for that team."
    ),
    func=NBATeamStatsByNameTool().run,
    args_schema=TeamStatsInput  # Use the defined input schema
)


# -------------------------------------------------------------------------
# nba_all_teams_stats: Retrieve NBA Team Stats for All Teams
# -------------------------------------------------------------------------
from nba_api.stats.endpoints import leaguestandingsv3

class AllTeamsStatsInput(BaseModel):
    years: List[str] = Field(
        default=["2023"],
        description="A list of NBA season years (e.g., ['2022', '2023']). Defaults to the current season."
    )
    season_type: str = Field(
        default="Regular Season",
        description="The season type (e.g., 'Regular Season', 'Playoffs', 'Pre Season', 'All Star'). Defaults to 'Regular Season'."
    )

    @field_validator("years")
    def validate_years(cls, value):
        for year in value:
            if not year.isdigit() or len(year) != 4:
                raise ValueError("Each year must be a 4-digit string (e.g., '2023')")
        return value

# ========== 2) Define the Tool Class ==========
class NBAAllTeamsStatsTool:
    """
    Fetches NBA statistics for all teams for multiple seasons from stats.nba.com.
    """
    def __init__(self):
        pass

    def run(self, years: List[str] = ["2023"], season_type: str = "Regular Season") -> List[Dict[str, Any]]:
        """
        Returns the NBA team statistics as a list of dictionaries, one for each season.
        """
        all_seasons_stats = []
        try:
            for year in years:
                # Fetch team stats data using the team ID
                team_stats = leaguestandingsv3.LeagueStandingsV3(
                    season=year,  # Pass the year
                    season_type=season_type,
                    league_id='00',  # NBA league ID
                )

                team_stats_data = team_stats.get_data_frames()[0]

                # Check if the DataFrame is empty
                if team_stats_data.empty:
                    all_seasons_stats.append({"error": f"No stats found for season {year}, season_type {season_type}."})
                    continue  # Skip to the next year

                # Convert relevant columns and handle potential errors
                for col in ['PlayoffRank', 'ConferenceRank', 'DivisionRank', 'WINS', 'LOSSES', 'ConferenceGamesBack', 'DivisionGamesBack']:
                    if col in team_stats_data.columns:
                        try:
                            team_stats_data[col] = pd.to_numeric(team_stats_data[col], errors='coerce')
                        except (ValueError, TypeError):
                            pass

                # Add a 'Season' column to distinguish the results
                team_stats_data['Season'] = year
                all_seasons_stats.extend(team_stats_data.to_dict('records'))

            return all_seasons_stats

        except Exception as e:
            return [{"error": str(e)}]



# ========== 3) Create the LangChain StructuredTool ==========
nba_all_teams_stats = StructuredTool(
    name="nba_all_teams_stats",
    description=(
        "BASKETBALL NBA: "
        "Fetch the NBA team statistics for all teams for a given list of season years and a season type."
        " Returns a list of statistics for all teams for each season."
    ),
    func=NBAAllTeamsStatsTool().run,
    args_schema=AllTeamsStatsInput  # Use the defined input schema
)

# -------------------------------------------------------------------------
# nba_player_game_logs: Retrieve NBA Player Game Logs and stats
# -------------------------------------------------------------------------
# ========== 1) Define Input Schema ==========
class PlayerGameLogsInput(BaseModel):
    """
    Input schema for retrieving a player's game logs within a specified date range.
    """
    player_id: str = Field(
        ...,
        description="NBA player ID (e.g., '2544' for LeBron James)."
    )
    date_range: List[str] = Field(
        ...,
        description="A list of two dates representing the start and end of the range, formatted as 'YYYY-MM-DD' (e.g., ['2022-12-01', '2022-12-31']).",
        min_items=2,
        max_items=2
    )
    season_type: str = Field(
        default="Regular Season",
        description="Season type. One of 'Regular Season', 'Playoffs', 'Pre Season', 'All Star'."
    )

    @field_validator('date_range')
    def validate_date_range(cls, v):
        try:
            start_date = datetime.strptime(v[0], '%Y-%m-%d')
            end_date = datetime.strptime(v[1], '%Y-%m-%d')
            if start_date > end_date:
                raise ValueError("Start date must be before end date.")
        except ValueError:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD'.")
        return v


# ========== 2) Define the Tool Class ==========
class NBAPlayerGameLogsTool:
    """
    Pull game logs for an NBA player from stats.nba.com for each date within a specified date range.
    """
    def __init__(self):
        pass

    def run(self, player_id: str, date_range: List[str], season_type: str = "Regular Season") -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries, each representing a game log within the specified date range.
        If no game was played on a particular date, that date is skipped in the output.

        Args:
            player_id (str): The ID of the player to retrieve game logs for.
            date_range (List[str]): A list of two dates [start_date, end_date] in YYYY-MM-DD format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a game log, or an error message.
        """
        try:
            start_date_str, end_date_str = date_range
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            # Find games for the given player and date range
            gamefinder = leaguegamefinder.LeagueGameFinder(
                player_id_nullable=player_id,
                season_type_nullable=season_type,
                date_from_nullable=start_date.strftime('%m/%d/%Y'),
                date_to_nullable=end_date.strftime('%m/%d/%Y')
            )

            games = gamefinder.get_data_frames()[0]

            # Convert GAME_DATE to datetime objects for comparison
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

            # Generate all dates in the range
            all_dates = []
            current_date = start_date
            while current_date <= end_date:
                all_dates.append(current_date)
                current_date += timedelta(days=1)
            
            # Filter games by the generated dates
            games = games[games['GAME_DATE'].dt.date.isin([d.date() for d in all_dates])]


            # Return game results as a list of dictionaries
            return games.to_dict('records')

        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
from langchain.tools import StructuredTool

nba_player_game_logs = StructuredTool(
    name="nba_player_game_logs",
    description=(
        "BASKETBALL NBA: "
        "Obtain an NBA player's game statistics for dates within a specified date range "
        "from the stats.nba.com endpoints. Requires a valid player_id and a date_range "
        "as a list: ['YYYY-MM-DD', 'YYYY-MM-DD']. Returns game stats for each date where a game was played."
    ),
    func=NBAPlayerGameLogsTool().run,
    args_schema=PlayerGameLogsInput
)


# ---------------------------------------------------- SOCCER ------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------


###############################################################################
# 1) get_league_id_by_name_tool: Retrieve the League ID by Name
###############################################################################

class GetLeagueIdByNameInput(BaseModel):
    """
    Input schema for retrieving the league ID based on the league name.
    """
    league_name: str = Field(
        ...,
        description="Name of the league (e.g. 'Premier League', 'La Liga')."
    )

class GetLeagueIdByNameTool:
    """
    1. Search for the league ID via /leagues?search=league_name.
    2. Return the league ID for the specified league name.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_league_id(self, league_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        try:
            # Step 1: Get league ID by searching for league name
            leagues_url = f"{self.base_url}/leagues"
            leagues_params = {"search": league_name}  # Search the league by name
            resp = requests.get(leagues_url, headers=headers, params=leagues_params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("response"):
                return {"error": f"No leagues found matching '{league_name}'."}
            
            # Grab the first league from the response (assuming there's only one match)
            league_id = data["response"][0]["league"]["id"]
            return {"league_id": league_id}

        except Exception as e:
            return {"error": str(e)}

# Define the tool to retrieve the league ID
get_league_id_by_name = StructuredTool(
    name="get_league_id_by_name",
    description="SOCCER: Retrieve the league ID for a given league name (e.g. 'Premier League', 'La Liga').",
    func=GetLeagueIdByNameTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_league_id,
    args_schema=GetLeagueIdByNameInput
)


###############################################################################
# 1) get_all_leagues_id: Retrieve All Football Leagues with IDs
###############################################################################

class GetAllLeaguesInput(BaseModel):
    """
    Input schema for retrieving all football leagues with an optional filter for multiple countries.
    """
    country: Optional[List[str]] = Field(
        default=None,
        description="List of countries to filter by (e.g., ['England', 'Spain']). Use ['all'] to retrieve leagues from all countries."
    )

class GetAllLeaguesTool:
    """
    Retrieves a list of all football leagues with their corresponding league IDs,
    optionally filtered by one or more countries.
    Endpoint: GET /leagues
    Docs: https://www.api-football.com/documentation-v3#operation/get-leagues
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_all_leagues(self, country: Optional[List[str]] = None) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key
        }

        try:
            # Fetch all leagues
            leagues_url = f"{self.base_url}/leagues"
            response = requests.get(leagues_url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Extract league names and IDs
            leagues = {}
            for league_info in data.get("response", []):
                league_name = league_info["league"]["name"]
                league_id = league_info["league"]["id"]
                league_country = league_info["country"]["name"]

                # Apply filters
                if country and "all" not in country:
                    if league_country.lower() not in [c.lower() for c in country]:
                        continue

                leagues[league_name] = {
                    "league_id": league_id,
                    "country": league_country
                }

            return {"leagues": leagues}

        except Exception as e:
            return {"error": str(e)}

# Define the tool
get_all_leagues_id = StructuredTool(
    name="get_all_leagues_id",
    description="SOCCER: Retrieve a list of all football leagues with IDs, and an optional filter for one or multiple countries.",
    func=GetAllLeaguesTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_all_leagues,
    args_schema=GetAllLeaguesInput
)

###############################################################################
# 1) GetStandingsTool: Retrieve League/Team Standings
###############################################################################

class GetStandingsToolInput(BaseModel):
    """
    Input schema for retrieving league/team standings.
    'season' is a list of years, and 'league_id' is now a list of league IDs.
    """
    league_id: Optional[List[int]] = Field(
        default=None,
        description="List of League IDs to retrieve standings for (e.g., [2, 39] for La Liga & Premier League)."
    )
    season: List[int] = Field(
        ..., 
        description="(REQUIRED) List of 4-digit seasons (e.g. [2021] or [2021, 2022, 2023])."
    )
    team: Optional[int] = Field(
        default=None,
        description="Optionally retrieve standings for a specific team ID within the leagues/seasons."
    )

class GetStandingsTool:
    """
    Retrieves standings for multiple leagues or a specific team in those leagues.
    Supports multiple seasons.
    Endpoint: GET /standings
    Docs: https://www.api-football.com/documentation-v3#operation/get-standings
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_standings(
        self,
        league_id: Optional[List[int]],
        season: List[int],
        team: Optional[int]
    ) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  
            "x-rapidapi-key": self.api_key  
        }

        results = {}
        leagues = league_id if league_id else []  # Handle None case

        for league in leagues:
            results[league] = {}  # Store results by league
            for year in season:
                url = f"{self.base_url}/standings"
                params = {"season": year, "league": league}

                if team is not None:
                    params["team"] = team

                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    response.raise_for_status()
                    results[league][year] = response.json()  # Store results per league & season
                except Exception as e:
                    results[league][year] = {"error": str(e)}

        return results  # Dictionary with league_id as keys and nested seasons

# Structured tool integration
get_standings = StructuredTool(
    name="get_standings",
    description=(
        "SOCCER: Retrieve the standings table for multiple leagues and multiple seasons, "
        "optionally filtered by a team ID."
    ),
    func=GetStandingsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_standings,
    args_schema=GetStandingsToolInput
)


###############################################################################
# get_player_id_tool: Retrieve Player IDs by Name
###############################################################################
class GetPlayerIdInput(BaseModel):
    player_name: str = Field(
        ...,
        description="The first *or* last name of the player to search for (e.g., 'Lionel' OR 'Messi').  Do NOT enter both first and last name.",
    )

    # @validator("player_name")
    # def check_single_name(cls, value):
    #     if " " in value.strip():
    #         raise ValueError(
    #             "Please enter only the first *or* last name, not both.  "
    #             "The API treats the input as either a first name OR a last name."
    #         )
    #     if len(value.strip()) <3:
    #         raise ValueError("The name must be at least 3 characters long.")

    #     return value.strip()

class GetPlayerIdTool:
    """
    Retrieves a list of players matching a given name, returning key identifying information
    to help select the correct player ID.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_player_ids(self, player_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/players/profiles"  # Use the /players endpoint
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        params = {
            "search": player_name,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("response"):
                return {"error": f"No players found matching '{player_name}'."}

            player_list = []
            for item in data["response"]:
                player = item.get("player", {})
                # Extract relevant identifying information
                player_info = {
                    "player_id": player.get("id"),
                    "firstname": player.get("firstname"),
                    "lastname": player.get("lastname"),
                    "age": player.get("age"),
                    "nationality": player.get("nationality"),
                    "birth_date": player.get("birth", {}).get("date"),  # Include birth date
                    "birth_place": player.get("birth", {}).get("place"), # Include place of birth
                    "birth_country": player.get("birth", {}).get("country"), # Include country of birth
                    "height": player.get("height"),
                    "weight" : player.get("weight")
                }
                player_list.append(player_info)


            return {"players": player_list}  # Return a list of player info dictionaries

        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}


get_player_id = StructuredTool.from_function(
    func=GetPlayerIdTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_player_ids,
    name="get_player_id",
    description=(
        "SOCCER: "
        "Retrieve a list of player IDs and identifying information (name, age, nationality, birth date, birth place, height, weight) "
        "for players matching a given name.  Use this to find the ID of a specific player."
    ),
    args_schema=GetPlayerIdInput,
)

###############################################################################
# GetPlayerProfileTool: Fetch a Player's Profile
###############################################################################

class GetPlayerProfileInput(BaseModel):
    """
    Input schema for retrieving a player's profile by last name.
    """
    player_name: str = Field(
        ...,
        description="The last name of the player to look up. Must be >= 3 characters."
    )

class GetPlayerProfileTool:
    """
    Retrieves a player's profile and basic info by searching for their last name.
    Internally calls /players/profiles with a 'search' parameter.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_player_profile(self, player_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/players/profiles"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        params = {
            "search": player_name,
            "page": 1  # We fetch only the first page for simplicity
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_player_profile = StructuredTool(
    name="get_player_profile",
    description=(
        "SOCCER: "
        "Use this tool to retrieve a single player's profile info by their last name. "
        "Example usage: Provide 'Messi' or 'Ronaldo' to look up that player's details."
    ),
    func=GetPlayerProfileTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_player_profile,
    args_schema=GetPlayerProfileInput
)


###############################################################################
# get_player_statistics_tool: Retrieve Detailed Player Statistics
###############################################################################

class GetPlayerStatisticsInput(BaseModel):
    player_id: int = Field(..., description="The ID of the player.")
    seasons: List[int] = Field(
        ...,
        description="A list of seasons to get statistics for (4-digit years, e.g., [2021, 2022, 2023] or [2022] for a single season).",
    )
    league_name: Optional[str] = Field(
        None,
        description="Optional. The name of the league (e.g., 'Premier League').",
    )

    @field_validator("seasons", mode='before')
    def convert_single_season_to_list(cls, value):
        if isinstance(value, int):
            return [value]  # Convert single integer to a list
        return value

    @field_validator("league_name")
    def check_league_name(cls, value):
        if value is not None and len(value.strip()) < 3:
            raise ValueError("League name must be at least 3 characters long.")
        return value


class GetPlayerStatisticsTool:
    """
    Retrieves detailed player statistics, including advanced stats, for a given player ID.
    Filters by a list of seasons and an optional league name.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def _get_league_id(self, league_name: str, season: int) -> Optional[int]:
        """Helper function to get the league ID from the league name."""
        url = f"{self.base_url}/leagues"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        params = {"name": league_name, "season": season}  # Use season for accuracy, use 'name' instead of 'search'
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("response"):
                return None  # No league found

            for league_data in data["response"]:
                # Check for name match (case-insensitive)
                if league_data["league"]["name"].lower() == league_name.lower():
                    # Check if the specified season is available for this league
                  for league_season in league_data["seasons"]:
                    if league_season["year"] == season:
                        return league_data["league"]["id"]
            return None # Return after looping through

        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None

    def get_player_statistics(
        self,
        player_id: int,
        seasons: List[int],
        league_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/players"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        all_stats = []

        # Make API requests for each season
        for current_season in seasons:
            league_id = None  # Initialize league_id
            if league_name:
                league_id = self._get_league_id(league_name, current_season)
                if league_id is None:
                    all_stats.append({
                        "error": f"Could not find league ID for '{league_name}' in season {current_season}."
                    })
                    continue  # Skip to the next season

            params: Dict[str, Any] = {"id": player_id, "season": current_season}
            if league_id:
                params["league"] = league_id

            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get("response"):
                    # No stats found for this particular season/league
                    continue

                # Extract and format relevant statistics
                for entry in data["response"]:
                    player_info = entry.get("player", {})
                    for stats in entry.get("statistics", []):
                        extracted_stats: Dict[str, Any] = {
                            "player": {
                                "id": player_info.get("id"),
                                "name": player_info.get("name"),
                                "photo": player_info.get("photo"),
                            },
                            "team": {
                                "id": stats.get("team", {}).get("id"),
                                "name": stats.get("team", {}).get("name"),
                                "logo": stats.get("team", {}).get("logo"),
                            },
                            "league": {
                                "id": stats.get("league", {}).get("id"),
                                "name": stats.get("league", {}).get("name"),
                                "season": stats.get("league", {}).get("season"),
                                "country": stats.get("league", {}).get("country"),
                                "flag": stats.get("league", {}).get("flag"),
                            },
                            "games": {
                                "appearances": stats.get("games", {}).get("appearences"),
                                "lineups": stats.get("games", {}).get("lineups"),
                                "minutes": stats.get("games", {}).get("minutes"),
                                "position": stats.get("games", {}).get("position"),
                                "rating": stats.get("games", {}).get("rating"),
                            },
                            "substitutes": {
                                "in": stats.get("substitutes", {}).get("in"),
                                "out": stats.get("substitutes", {}).get("out"),
                                "bench": stats.get("substitutes", {}).get("bench"),
                            },
                            "shots": {
                                "total": stats.get("shots", {}).get("total"),
                                "on": stats.get("shots", {}).get("on"),
                            },
                            "goals": {
                                "total": stats.get("goals", {}).get("total"),
                                "conceded": stats.get("goals", {}).get("conceded"),
                                "assists": stats.get("goals", {}).get("assists"),
                                "saves": stats.get("goals", {}).get("saves"),
                            },
                            "passes": {
                                "total": stats.get("passes", {}).get("total"),
                                "key": stats.get("passes", {}).get("key"),
                                "accuracy": stats.get("passes", {}).get("accuracy"),
                            },
                            "tackles": {
                                "total": stats.get("tackles", {}).get("total"),
                                "blocks": stats.get("tackles", {}).get("blocks"),
                                "interceptions": stats.get("tackles", {}).get("interceptions"),
                            },
                            "duels": {
                                "total": stats.get("duels", {}).get("total"),
                                "won": stats.get("duels", {}).get("won"),
                            },
                            "dribbles": {
                                "attempts": stats.get("dribbles", {}).get("attempts"),
                                "success": stats.get("dribbles", {}).get("success"),
                            },
                            "fouls": {
                                "drawn": stats.get("fouls", {}).get("drawn"),
                                "committed": stats.get("fouls", {}).get("committed"),
                            },
                            "cards": {
                                "yellow": stats.get("cards", {}).get("yellow"),
                                "red": stats.get("cards", {}).get("red"),
                            },
                            "penalty": {
                                "won": stats.get("penalty", {}).get("won"),
                                "committed": stats.get("penalty", {}).get("committed"),
                                "scored": stats.get("penalty", {}).get("scored"),
                                "missed": stats.get("penalty", {}).get("missed"),
                                "saved": stats.get("penalty", {}).get("saved"),
                            },
                        }
                        all_stats.append(extracted_stats)

            except requests.exceptions.RequestException as e:
                all_stats.append({"error": f"Request failed for season {current_season}: {e}"})
            except Exception as e:
                all_stats.append({"error": f"An unexpected error occurred for season {current_season}: {e}"})

        if not all_stats:
            return {
                "error": f"No statistics found for player ID {player_id} for the specified seasons/league."
            }

        return {"player_statistics": all_stats}


get_player_statistics = StructuredTool.from_function(
    func=GetPlayerStatisticsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_player_statistics,
    name="get_player_statistics",
    description=(
        "SOCCER: "
        "Retrieve detailed player statistics for a given player ID.  "
        "Filter by a list of seasons and an optional league name.  Includes advanced stats."
    ),
    args_schema=GetPlayerStatisticsInput,
)


###############################################################################
# get_player_statistics_tool_2: Retrieve Detailed Player Statistics
###############################################################################

class GetPlayerStatisticsInput_2(BaseModel):
    player_id: int = Field(..., description="The ID of the player.")
    seasons: List[int] = Field(
        ...,
        description="A list of seasons to get statistics for (4-digit years, e.g., [2021, 2022, 2023] or [2022] for a single season).",
    )
    league_id: Optional[int] = Field(
        None,
        description="Optional. The ID of the league.  Requires 'seasons' to be set.",
    )

    @field_validator("seasons", mode='before')
    def convert_single_season_to_list(cls, value):
        if isinstance(value, int):
            return [value]  # Convert single integer to a list
        return value


class GetPlayerStatisticsTool_2:
    """
    Retrieves detailed player statistics, including advanced stats, for a given player ID.
    Filters by a list of seasons and an optional league ID.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_player_statistics(
        self,
        player_id: int,
        seasons: List[int],
        league_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/players"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        all_stats = []

        # Make API requests for each season
        for current_season in seasons:
            params: Dict[str, Any] = {"id": player_id, "season": current_season}
            if league_id:
                params["league"] = league_id

            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get("response"):
                    # No stats found for this particular season, continue to the next
                    continue

                # Extract and format relevant statistics for this season
                for entry in data["response"]:
                    player_info = entry.get("player", {})
                    for stats in entry.get("statistics", []):
                        extracted_stats: Dict[str, Any] = {
                            "player": {
                                "id": player_info.get("id"),
                                "name": player_info.get("name"),
                                "photo": player_info.get("photo"),
                            },
                            "team": {
                                "id": stats.get("team", {}).get("id"),
                                "name": stats.get("team", {}).get("name"),
                                "logo": stats.get("team", {}).get("logo"),
                            },
                            "league": {
                                "id": stats.get("league", {}).get("id"),
                                "name": stats.get("league", {}).get("name"),
                                "season": stats.get("league", {}).get("season"),
                                "country": stats.get("league", {}).get("country"),
                                "flag": stats.get("league", {}).get("flag")
                            },
                            "games": {
                                "appearances": stats.get("games", {}).get("appearences"),
                                "lineups": stats.get("games", {}).get("lineups"),
                                "minutes": stats.get("games", {}).get("minutes"),
                                "position": stats.get("games", {}).get("position"),
                                "rating": stats.get("games", {}).get("rating"),
                            },
                            "substitutes": {
                                "in": stats.get("substitutes", {}).get("in"),
                                "out": stats.get("substitutes", {}).get("out"),
                                "bench": stats.get("substitutes", {}).get("bench"),
                            },
                            "shots": {
                                "total": stats.get("shots", {}).get("total"),
                                "on": stats.get("shots", {}).get("on"),
                            },
                            "goals": {
                                "total": stats.get("goals", {}).get("total"),
                                "conceded": stats.get("goals", {}).get("conceded"),
                                "assists": stats.get("goals", {}).get("assists"),
                                "saves": stats.get("goals", {}).get("saves"),
                            },
                            "passes": {
                                "total": stats.get("passes", {}).get("total"),
                                "key": stats.get("passes", {}).get("key"),
                                "accuracy": stats.get("passes", {}).get("accuracy"),
                            },
                            "tackles": {
                                "total": stats.get("tackles", {}).get("total"),
                                "blocks": stats.get("tackles", {}).get("blocks"),
                                "interceptions": stats.get("tackles", {}).get("interceptions"),
                            },
                            "duels": {
                                "total": stats.get("duels", {}).get("total"),
                                "won": stats.get("duels", {}).get("won"),
                            },
                            "dribbles": {
                                "attempts": stats.get("dribbles", {}).get("attempts"),
                                "success": stats.get("dribbles", {}).get("success"),
                            },
                            "fouls": {
                                "drawn": stats.get("fouls", {}).get("drawn"),
                                "committed": stats.get("fouls", {}).get("committed"),
                            },
                            "cards": {
                                "yellow": stats.get("cards", {}).get("yellow"),
                                "red": stats.get("cards", {}).get("red"),
                            },
                            "penalty": {
                                "won": stats.get("penalty", {}).get("won"),
                                "committed": stats.get("penalty", {}).get("committed"),
                                "scored": stats.get("penalty", {}).get("scored"),
                                "missed": stats.get("penalty", {}).get("missed"),
                                "saved": stats.get("penalty", {}).get("saved"),
                            },
                        }
                        all_stats.append(extracted_stats)

            except requests.exceptions.RequestException as e:
                return {"error": f"Request failed for season {current_season}: {e}"}
            except Exception as e:
                return {"error": f"An unexpected error occurred for season {current_season}: {e}"}

        if not all_stats:
            return {
                "error": f"No statistics found for player ID {player_id} for the specified seasons/league."
            }

        return {"player_statistics": all_stats}


get_player_statistics_2 = StructuredTool.from_function(
    func=GetPlayerStatisticsTool_2(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_player_statistics,
    name="get_player_statistics_2",
    description=(
        "SOCCER: "
        "Retrieve detailed player statistics for a given player ID.  "
        "Filter by a list of seasons and an optional league ID.  Includes advanced stats."
    ),
    args_schema=GetPlayerStatisticsInput_2,
)

# -------------------------------------------------------------------
#  GetTeamFixturesTool: Fetch a Team's Fixtures
# -------------------------------------------------------------------
class GetTeamFixturesInput(BaseModel):
    """
    Input for retrieving a team's fixtures by name.
    """
    team_name: str = Field(
        ...,
        description="The team's name to search for. Must be >= 3 characters for accurate search."
    )
    type: str = Field(
        default="upcoming",
        description="Either 'past' or 'upcoming' fixtures."
    )
    limit: int = Field(
        default=5,
        description="How many fixtures to retrieve: e.g. last=5 or next=5. Default=5."
    )

class GetTeamFixturesTool:
    """
    Given a team name, finds the team's ID, then fetches either the last N or next N fixtures.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_team_fixtures(self, team_name: str, type: str, limit: int) -> Dict[str, Any]:
        """
        1) Look up team ID from /teams?search={team_name}.
        2) Depending on 'type':
            - if 'past': use /fixtures?team=ID&last={limit}
            - if 'upcoming': use /fixtures?team=ID&next={limit}
        3) Return the resulting fixtures or an error if not found.
        """
        # Step 1: Find the Team ID
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        search_url = f"{self.base_url}/teams"
        search_params = {"search": team_name}

        try:
            search_resp = requests.get(search_url, headers=headers, params=search_params, timeout=15)
            search_resp.raise_for_status()
            teams_data = search_resp.json()

            if not teams_data.get("response"):
                return {"error": f"No teams found matching '{team_name}'."}

            # Just pick the first matching team for simplicity
            first_team = teams_data["response"][0]
            team_id = first_team["team"]["id"]

            # Step 2: Fetch fixtures
            fixtures_url = f"{self.base_url}/fixtures"
            fixtures_params = {"team": team_id}

            if type.lower() == "past":
                fixtures_params["last"] = limit
            else:
                # Default is 'upcoming'
                fixtures_params["next"] = limit

            fixtures_resp = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
            fixtures_resp.raise_for_status()
            return fixtures_resp.json()

        except Exception as e:
            return {"error": str(e)}

get_team_fixtures = StructuredTool(
    name="get_team_fixtures",
    description=(
        "SOCCER: "
        "Given a team name, returns either the last N or the next N fixtures for that team. "
        "Useful for quickly seeing a team's recent or upcoming matches."
    ),
    func=GetTeamFixturesTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_team_fixtures,
    args_schema=GetTeamFixturesInput
)



# -------------------------------------------------------------------
# GetFixtureStatisticsTool: Fetch Detailed Fixture Stats
# -------------------------------------------------------------------
class GetFixtureStatisticsInput(BaseModel):
    """
    Input schema for retrieving a single fixture's detailed stats.
    """
    fixture_id: int = Field(
        ...,
        description="The numeric ID of the fixture/game. Example: 215662."
    )

class GetFixtureStatisticsTool:
    """
    Given a fixture (game) ID, retrieves stats like shots on goal, possession, corners, etc.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_fixture_stats(self, fixture_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/fixtures/statistics"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        params = {"fixture": fixture_id}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_fixture_statistics = StructuredTool(
    name="get_fixture_statistics",
    description=(
        "SOCCER: "
        "Use this tool to retrieve box-score style statistics for a given fixture. "
        "You must already know the fixture ID, e.g. 215662."
    ),
    func=GetFixtureStatisticsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_fixture_stats,
    args_schema=GetFixtureStatisticsInput
)



# -------------------------------------------------------------------
# GetTeamFixturesByDateRangeTool
# -------------------------------------------------------------------
class GetTeamFixturesByDateRangeInput(BaseModel):
    team_name: str = Field(
        ...,
        description="Team name to search for (e.g. 'Arsenal', 'Barcelona')."
    )
    season: str = Field(
        default="2024",
        description="Season in YYYY format (e.g. '2024')."
    )
    from_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format (e.g. '2023-08-01')."
    )
    to_date: str = Field(
        ...,
        description="End date in YYYY-MM-DD format (e.g. '2023-08-31')."
    )

class GetTeamFixturesByDateRangeTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # RapidAPI base URL
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_team_fixtures_by_date_range(self, team_name: str, from_date: str, to_date: str, season: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        # Step 1: find team ID
        teams_url = f"{self.base_url}/teams"
        teams_params = {"search": team_name}
        resp = requests.get(teams_url, headers=headers, params=teams_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # print(data)
        if not data.get("response"):
            return {"error": f"No team found matching '{team_name}'."}
        team_id = data["response"][0]["team"]["id"]

        # Step 2: fetch fixtures in date range
        fixtures_url = f"{self.base_url}/fixtures"
        fixtures_params = {
            "team": team_id,
            "from": from_date,
            "to": to_date,
            "season": season  # or some 4-digit year
        }
        resp_fixtures = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
        resp_fixtures.raise_for_status()
        return resp_fixtures.json()


get_team_fixtures_by_date_range = StructuredTool(
    name="get_team_fixtures_by_date_range",
    description=(
        "SOCCER: "
        "Retrieve all fixtures for a given team within a date range. "
        "Input: team name, from_date (YYYY-MM-DD), to_date (YYYY-MM-DD)."
    ),
    func=GetTeamFixturesByDateRangeTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_team_fixtures_by_date_range,
    args_schema=GetTeamFixturesByDateRangeInput
)



# -------------------------------------------------------------------
# GetFixtureEventsTool
# -------------------------------------------------------------------
class GetFixtureEventsInput(BaseModel):
    fixture_id: int = Field(
        ...,
        description="Numeric ID of the fixture whose events you want (e.g. 215662)."
    )

class GetFixtureEventsTool:
    """
    Given a fixture ID, returns the events that occurred (goals, substitutions, cards, etc.).
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_fixture_events(self, fixture_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/fixtures/events"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        params = {"fixture": fixture_id}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_fixture_events = StructuredTool(
    name="get_fixture_events",
    description=(
        "SOCCER: "
        "Retrieve all in-game events for a given fixture ID (e.g. goals, cards, subs). "
        "You must know the fixture ID beforehand."
    ),
    func=GetFixtureEventsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_fixture_events,
    args_schema=GetFixtureEventsInput
)


# -------------------------------------------------------------------
# GetMultipleFixturesStatsTool
# -------------------------------------------------------------------
class GetMultipleFixturesStatsInput(BaseModel):
    fixture_ids: list[int] = Field(
        ...,
        description="A list of numeric fixture IDs to get stats for, e.g. [215662, 215663] or [215663] for one fixture IDs."
    )

class GetMultipleFixturesStatsTool:
    """
    Given multiple fixture IDs, calls /fixtures/statistics for each ID one by one
    and aggregates the results in a list.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_multiple_fixtures_stats(self, fixture_ids: list[int]) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        combined_results = []

        for f_id in fixture_ids:
            try:
                url = f"{self.base_url}/fixtures/statistics"
                params = {"fixture": f_id}
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                combined_results.append({f_id: data})
            except Exception as e:
                combined_results.append({f_id: {"error": str(e)}})

        return {"fixtures_statistics": combined_results}

get_multiple_fixtures_stats = StructuredTool(
    name="get_multiple_fixtures_stats",
    description=(
        "SOCCER: "
        "Retrieve stats (shots, possession, etc.) for multiple fixtures at once. "
        "Input a list of fixture IDs, e.g. [215662, 215663]."
    ),
    func=GetMultipleFixturesStatsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_multiple_fixtures_stats,
    args_schema=GetMultipleFixturesStatsInput
)


# -------------------------------------------------------------------
# GetLeagueScheduleByDateTool
# -------------------------------------------------------------------
from langchain.tools.base import StructuredTool
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import requests

class GetLeagueScheduleByDateInput(BaseModel):
    league_name: str = Field(
        ..., 
        description="Name of the league (e.g. 'Premier League', 'La Liga')."
    )
    date: List[str] = Field(
        ..., 
        description="List of dates in YYYY-MM-DD format (e.g. ['2025-03-09'] or ['2025-03-09', '2025-03-10'])."
    )
    season: str = Field(
        default="2024", 
        description="Season in YYYY format (e.g. '2024')."
    )

class GetLeagueScheduleByDateTool:
    """
    1. Search for the league ID via /leagues?search=league_name
    2. Use the found ID to call /fixtures?league={id}&date={YYYY-MM-DD}&season={season}
    3. Return JSON of the fixtures (the schedule for those days), supporting multiple dates.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_league_schedule(self, league_name: str, date: List[str], season: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  
            "x-rapidapi-key": self.api_key  
        }

        # Step 1: Get league ID by searching name
        try:
            leagues_url = f"{self.base_url}/leagues"
            leagues_params = {"search": league_name}
            resp = requests.get(leagues_url, headers=headers, params=leagues_params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("response"):
                return {"error": f"No leagues found matching '{league_name}'."}

            # We'll just grab the first result
            league_id = data["response"][0]["league"]["id"]
            
            results = {}
            for match_date in date:
                # Step 2: Get fixtures for that league & date
                fixtures_url = f"{self.base_url}/fixtures"
                fixtures_params = {
                    "league": league_id,
                    "date": match_date,  
                    "season": season  
                }
                
                resp_fixtures = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
                resp_fixtures.raise_for_status()
                
                results[match_date] = resp_fixtures.json()  # Store results per date

            return results  # Return structured results with dates as keys

        except Exception as e:
            return {"error": str(e)}

# Define the tool
get_league_schedule_by_date = StructuredTool(
    name="get_league_schedule_by_date",
    description=(
        "SOCCER: "
        "Retrieve the schedule (fixtures) for a given league on one or multiple specified dates. "
        "Supports a single season."
    ),
    func=GetLeagueScheduleByDateTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_league_schedule,
    args_schema=GetLeagueScheduleByDateInput
)

# -------------------------------------------------------------------
# GetLiveMatchForTeamTool: 
# -------------------------------------------------------------------
from langchain.tools.base import StructuredTool
from pydantic import BaseModel, Field
from typing import Any, Dict
import requests

class GetLiveMatchForTeamInput(BaseModel):
    """
    Minimal input: just the team's name.
    """
    team_name: str = Field(
        ...,
        description="The team's name. Example: 'Arsenal', 'Barcelona'. Must be >= 3 chars for accurate searching."
    )

class GetLiveMatchForTeamTool:
    """
    1) Resolve the team name to team ID (via /teams?search).
    2) Check /fixtures?team=TEAM_ID&live=all to find any in-progress match.
    3) Return the fixture data if live, else error message.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_live_match_for_team(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        # Step 1: find team ID
        try:
            teams_resp = requests.get(
                f"{self.base_url}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_data = teams_resp.json()

            if not teams_data.get("response"):
                return {"error": f"No team found matching '{team_name}'."}

            team_id = teams_data["response"][0]["team"]["id"]

            # Step 2: look for live matches
            fixtures_resp = requests.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "live": "all"},
                timeout=15
            )
            fixtures_resp.raise_for_status()
            fixtures_data = fixtures_resp.json()

            live_fixtures = fixtures_data.get("response", [])

            if not live_fixtures:
                return {"message": f"No live match found for '{team_name}' right now."}

            # Typically only 1, but if multiple, just return the first
            return {"live_fixture": live_fixtures[0]}

        except Exception as e:
            return {"error": str(e)}

get_live_match_for_team = StructuredTool(
    name="get_live_match_for_team",
    description=(
        "SOCCER: "
        "Check if a given team is currently playing live. Input the team name. "
        "Returns the live match fixture info if found, else returns a message that no live match is found."
    ),
    func=GetLiveMatchForTeamTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_live_match_for_team,
    args_schema=GetLiveMatchForTeamInput
)

# -------------------------------------------------------------------
# GetLiveStatsForTeamTool
# -------------------------------------------------------------------
class GetLiveStatsForTeamInput(BaseModel):
    team_name: str = Field(
        ...,
        description="Team name to get live stats for. e.g. 'Arsenal', 'Barcelona'."
    )

class GetLiveStatsForTeamTool:
    """
    1. Find team ID by name.
    2. Find current live fixture for that team.
    3. If found, call /fixtures/statistics?fixture=FIXTURE_ID to get live stats.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_live_stats_for_team(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        try:
            # Step 1: get team ID
            teams_resp = requests.get(
                f"{self.base_url}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_data = teams_resp.json()
            if not teams_data.get("response"):
                return {"error": f"No team found matching '{team_name}'."}
            team_id = teams_data["response"][0]["team"]["id"]

            # Step 2: check for live fixtures
            fixtures_resp = requests.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "live": "all"},
                timeout=15
            )
            fixtures_resp.raise_for_status()
            fixtures_data = fixtures_resp.json()
            live_fixtures = fixtures_data.get("response", [])
            if not live_fixtures:
                return {"message": f"No live match for '{team_name}' right now."}

            fixture_id = live_fixtures[0]["fixture"]["id"]

            # Step 3: get stats for that fixture
            stats_resp = requests.get(
                f"{self.base_url}/fixtures/statistics",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=15
            )
            stats_resp.raise_for_status()
            stats_data = stats_resp.json()

            return {"fixture_id": fixture_id, "live_stats": stats_data}

        except Exception as e:
            return {"error": str(e)}

get_live_stats_for_team = StructuredTool(
    name="get_live_stats_for_team",
    description=(
        "SOCCER: "
        "Retrieve live in-game stats (shots on goal, possession, etc.) for a team currently in a match. "
        "Input the team name. If no live match is found, returns a message."
    ),
    func=GetLiveStatsForTeamTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_live_stats_for_team,
    args_schema=GetLiveStatsForTeamInput
)

# -------------------------------------------------------------------
# GetLiveMatchTimelineTool
# -------------------------------------------------------------------
class GetLiveMatchTimelineInput(BaseModel):
    team_name: str = Field(
        ...,
        description="Team name to retrieve live timeline of the current match if playing. E.g. 'Arsenal'."
    )

class GetLiveMatchTimelineTool:
    """
    1. Find the team ID by name
    2. Check if there's a live fixture for that team
    3. If found, call /fixtures/events?fixture=... to get timeline events
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_live_match_timeline(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        try:
            # Step 1: team ID
            teams_resp = requests.get(
                f"{self.base_url}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_data = teams_resp.json()
            if not teams_data.get("response"):
                return {"error": f"No team found matching '{team_name}'."}
            team_id = teams_data["response"][0]["team"]["id"]

            # Step 2: check live fixtures
            fixtures_resp = requests.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "live": "all"},
                timeout=15
            )
            fixtures_resp.raise_for_status()
            fixtures_data = fixtures_resp.json()
            live_fixtures = fixtures_data.get("response", [])
            if not live_fixtures:
                return {"message": f"No live match for '{team_name}' right now."}

            fixture_id = live_fixtures[0]["fixture"]["id"]

            # Step 3: get events timeline
            events_resp = requests.get(
                f"{self.base_url}/fixtures/events",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=15
            )
            events_resp.raise_for_status()
            events_data = events_resp.json()

            return {"fixture_id": fixture_id, "timeline_events": events_data}

        except Exception as e:
            return {"error": str(e)}

get_live_match_timeline = StructuredTool(
    name="get_live_match_timeline",
    description=(
        "SOCCER: "
        "Retrieve the real-time timeline of a currently live match for a given team. "
        "Input the team name. Returns events like goals, substitutions, and cards."
    ),
    func=GetLiveMatchTimelineTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_live_match_timeline,
    args_schema=GetLiveMatchTimelineInput
)

# -------------------------------------------------------------------
# LeagueInformationTool
# -------------------------------------------------------------------
class GetLeagueInfoInput(BaseModel):
    league_name: str = Field(..., description="Name of the league (e.g., 'Champions League')")

class GetLeagueInfoTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
    
    def get_league_info(self, league_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key
        }

        # Fetch league information
        league_url = f"{self.base_url}/leagues"
        params = {"search": league_name}
        resp = requests.get(league_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data

# Define the tool
get_league_info = StructuredTool(
    name="get_league_info",
    description="SOCCER: Retrieve information about a specific football league (teams, season, fixtures, etc.)",
    func=GetLeagueInfoTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_league_info,
    args_schema=GetLeagueInfoInput
)

# -------------------------------------------------------------------
# TeamInformationTool
# -------------------------------------------------------------------
class GetTeamInfoInput(BaseModel):
    team_name: str = Field(..., description="Name of the team (e.g., 'Manchester United')")

class GetTeamInfoTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
    
    def get_team_info(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key
        }

        # Fetch team information
        teams_url = f"{self.base_url}/teams"
        teams_params = {"search": team_name}
        resp = requests.get(teams_url, headers=headers, params=teams_params)
        resp.raise_for_status()
        data = resp.json()
        return data


# Define the tool
get_team_info = StructuredTool(
    name="get_team_info",
    description="SOCCER: Retrieve basic information about a specific football team (players, history, etc.)",
    func=GetTeamInfoTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_team_info,
    args_schema=GetTeamInfoInput
)

# -------------------------------------------------------------------
# PlayerStatisticsTool
# -------------------------------------------------------------------


base_tools = [get_player_statistics_2, get_team_fixtures, get_fixture_statistics, get_team_fixtures_by_date_range,
              get_fixture_events, get_multiple_fixtures_stats, get_live_match_for_team, get_live_stats_for_team,
              get_live_match_timeline, get_league_info, get_team_info]