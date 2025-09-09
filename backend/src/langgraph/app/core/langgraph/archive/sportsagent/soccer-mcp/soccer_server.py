from mcp.server.fastmcp import FastMCP
import time
import signal
import sys
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import os
import requests


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
    name="soccer_server",
    # host="127.0.0.1",
    # port=5000,
    # Add this to make the server more resilient
    timeout=30  # Increase timeout to 30 seconds
)

@mcp.tool()
def get_league_fixtures(league_id: int, season: int) -> Dict[str, Any]:
    """Retrieves all fixtures for a given league and season.

    Args:
        league_id (int): The ID of the league.
        season (int): The year of the season (e.g., 2023 for the 2023-2024 season).

    Returns:
        Dict[str, Any]: A dictionary containing fixture data or an error message. Key fields:
            * "response" (List[Dict[str, Any]]): A list of fixture dictionaries, as returned by the API.
            * "error" (str): An error message if the request failed.

    Example:
        ```python
        get_league_fixtures(league_id=39, season=2023)
        ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    fixtures_url = f"{base_url}/fixtures"
    fixtures_params = {"league": league_id, "season": season}

    try:
        response = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=30)  # Increased timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_league_id_by_name(league_name: str) -> Dict[str, Any]:
    """Retrieve the league ID for a given league name.

    This tool searches for a league by its name and returns its ID.  It uses the
    `/leagues` endpoint of the API-Football API.

    **Args:**

        league_name (str): The name of the league (e.g., "Premier League", "La Liga").

    **Returns:**

        Dict[str, Any]: A dictionary containing the league ID, or an error message.  Key fields:

            *   "league_id" (int): The ID of the league, if found.
            *   "error" (str): An error message if the league is not found or an error occurs.

    **Example:**
        ```
        get_league_id_by_name(league_name="Premier League")
        # Expected output (may vary):  {"league_id": 39}
        ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    try:
        leagues_url = f"{base_url}/leagues"
        leagues_params = {"search": league_name}
        resp = requests.get(leagues_url, headers=headers, params=leagues_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("response"):
            return {"error": f"No leagues found matching '{league_name}'."}

        league_id = data["response"][0]["league"]["id"]
        return {"league_id": league_id}

    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
def get_all_leagues_id(country: Optional[List[str]] = None) -> Dict[str, Any]:
    """Retrieve a list of all football leagues with IDs, optionally filtered by country.

    This tool retrieves a list of football leagues and their IDs. It can be filtered
    by providing a list of country names.  Uses the `/leagues` endpoint.

    **Args:**

        country (Optional[List[str]]): A list of country names to filter the leagues.
            Use ["all"] to retrieve leagues from all countries.  If None (default),
            no filtering is applied (though this is the same behavior as ["all"]).

    **Returns:**

        Dict[str, Any]:  A dictionary containing league information, or an error message. Key fields:

            *   "leagues" (Dict[str, Dict[str, Any]]):  A dictionary where keys are league names
                and values are dictionaries containing "league_id" and "country".
            *   "error" (str): An error message if the request fails.
        
        **Example:**
          ```python
          get_all_leagues_id(country = ["England", "Spain"])
          # Expected sample Output (will have many more entries):
          # {
          #     "leagues": {
          #         "Premier League": {"league_id": 39, "country": "England"},
          #         "La Liga": {"league_id": 140, "country": "Spain"},
          #          ...
          #     }
          # }
          ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    try:
        leagues_url = f"{base_url}/leagues"
        response = requests.get(leagues_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        leagues: Dict[str, Dict[str, Any]] = {}
        for league_info in data.get("response", []):
            league_name = league_info["league"]["name"]
            league_id = league_info["league"]["id"]
            league_country = league_info["country"]["name"]

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



@mcp.tool()
def get_standings(league_id: Optional[List[int]], season: List[int], team: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve league standings for multiple leagues and seasons, optionally filtered by team.

    This tool retrieves the standings table for one or more leagues, across multiple
    seasons. It can optionally filter the results to show standings for a specific team.
    Uses the `/standings` endpoint.

    **Args:**

        league_id (Optional[List[int]]): A list of league IDs to retrieve standings for.
        season (List[int]): A list of 4-digit season years (e.g., [2021, 2022]).
        team (Optional[int]):  A specific team ID to filter the standings.

    **Returns:**

        Dict[str, Any]: A dictionary containing the standings, or an error message.  The structure is:

            *   `{league_id: {season: standings_data}}`

            `standings_data` is the raw JSON response from the API for the given league and season.  If an error occurs
            for a specific league/season, the `standings_data` will be `{"error": "error message"}`.

        **Example:**

          ```python
            get_standings(league_id=[39, 140], season=[2022, 2023], team=None)
          ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    results: Dict[int, Dict[int, Any]] = {}
    leagues = league_id if league_id else []

    for league in leagues:
        results[league] = {}
        for year in season:
            url = f"{base_url}/standings"
            params = {"season": year, "league": league}

            if team is not None:
                params["team"] = team

            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                results[league][year] = response.json()
            except Exception as e:
                results[league][year] = {"error": str(e)}

    return results

@mcp.tool()
def get_player_id(player_name: str) -> Dict[str, Any]:
    """Retrieve a list of player IDs and identifying information for players matching a given name.

    This tool searches for players by either their first *or* last name and returns a list of
    potential matches.  It includes identifying information to help disambiguate players.
    Uses the `/players/profiles` endpoint.

    **Args:**

        player_name (str): The first *or* last name of the player (e.g., "Lionel" or "Messi").
                           Do *not* provide both first and last names.  The name must be at least
                           3 characters long.

    **Returns:**

        Dict[str, Any]: A dictionary containing a list of players or an error message. Key fields:
            * "players" (List[Dict[str, Any]]): A list of dictionaries, each representing a player.
              Each player dictionary includes:
                * "player_id" (int): The player's ID.
                * "firstname" (str): The player's first name.
                * "lastname" (str): The player's last name.
                * "age" (int): The player's age.
                * "nationality" (str): The player's nationality.
                * "birth_date" (str): The player's birth date (YYYY-MM-DD).
                * "birth_place" (str): The player's birth place.
                *  "birth_country" (str)
                * "height" (str): The player's height (e.g., "170 cm").
                * "weight" (str): The player's weight (e.g., "68 kg").
            * "error" (str): An error message if no players are found or an error occurs.

    **Example:**
        ```
        get_player_id(player_name="Messi")
        ```

    """
    if " " in player_name.strip():
        return {"error": "Please enter only the first *or* last name, not both."}
    if len(player_name.strip()) < 3:
         return {"error": "The name must be at least 3 characters long."}


    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    url = f"{base_url}/players/profiles"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key,
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
            player_info = {
                "player_id": player.get("id"),
                "firstname": player.get("firstname"),
                "lastname": player.get("lastname"),
                "age": player.get("age"),
                "nationality": player.get("nationality"),
                "birth_date": player.get("birth", {}).get("date"),
                "birth_place": player.get("birth", {}).get("place"),
                "birth_country": player.get("birth", {}).get("country"),
                "height": player.get("height"),
                "weight": player.get("weight")
            }
            player_list.append(player_info)

        return {"players": player_list}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_player_profile(player_name: str) -> Dict[str, Any]:
    """Retrieve a single player's profile information by their last name.

    This tool retrieves a player's profile by searching for their last name.  It uses
    the `/players/profiles` endpoint.

    **Args:**

        player_name (str): The last name of the player to look up. Must be >= 3 characters.

    **Returns:**

        Dict[str, Any]: The raw JSON response from the API, or a dictionary with an "error" key
        if the request fails.

    **Example:**
    ```python
    get_player_profile(player_name = "Messi")
    ```
    """
    if len(player_name.strip()) < 3:
         return {"error": "The name must be at least 3 characters long."}

    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}


    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    url = f"{base_url}/players/profiles"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    params = {
        "search": player_name,
        "page": 1  # Fetch only the first page
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
def get_player_statistics(player_id: int, seasons: List[int], league_name: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve detailed player statistics for given seasons and optional league name.

    This tool retrieves detailed player statistics, including advanced stats, for a
    specified player ID.  It filters the results by a list of seasons and, optionally,
    by a league name. It uses the /players endpoint.

    **Args:**

        player_id (int): The ID of the player.
        seasons (List[int]): A list of seasons to get statistics for (4-digit years,
            e.g., [2021, 2022] or [2023]).
        league_name (Optional[str]): The name of the league (e.g., "Premier League").
            If provided, statistics will be retrieved only for this league.  If the
            league name cannot be found for a given season, an error will be included
            in the results for that season.

    **Returns:**

        Dict[str, Any]: A dictionary containing the player statistics or error messages. Key fields:

            *   "player_statistics" (List[Dict[str, Any]]): A list of dictionaries, each
                representing player statistics for a specific season (and league, if
                specified).
            *   "error" (str):  An error message may be present *within* the
                `player_statistics` list if there was a problem fetching data for a specific
                season, or at the top level if no statistics at all could be retrieved.

            Each dictionary in "player_statistics" contains detailed statistics, grouped
            by category ("player", "team", "league", "games", "substitutes", "shots",
            "goals", "passes", "tackles", "duels", "dribbles", "fouls", "cards", "penalty").
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if isinstance(seasons, int):
        seasons = [seasons]
    if league_name is not None and len(league_name.strip()) < 3:
        return {"error": "League name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    url = f"{base_url}/players"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key,
    }
    all_stats = []

    def _get_league_id(league_name: str, season: int) -> Optional[int]:
        """Helper function to get the league ID from the league name."""
        url = f"{base_url}/leagues"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": api_key,
        }
        params = {"name": league_name, "season": season}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("response"):
                return None

            for league_data in data["response"]:
                if league_data["league"]["name"].lower() == league_name.lower():
                    for league_season in league_data["seasons"]:
                        if league_season["year"] == season:
                            return league_data["league"]["id"]
            return None

        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None
    # End of helper function

    for current_season in seasons:
        league_id = None
        if league_name:
            league_id = _get_league_id(league_name, current_season)
            if league_id is None:
                all_stats.append({
                    "error": f"Could not find league ID for '{league_name}' in season {current_season}."
                })
                continue

        params: Dict[str, Any] = {"id": player_id, "season": current_season}
        if league_id:
            params["league"] = league_id

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("response"):
                continue

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


@mcp.tool()
def get_player_statistics_2(player_id: int, seasons: List[int], league_id: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve detailed player statistics for given seasons and optional league ID.

    This tool retrieves detailed player statistics, including advanced stats, for a
    specified player ID. It filters the results by a list of seasons and, optionally,
    by a league ID. It uses the /players endpoint.

    **Args:**

        player_id (int): The ID of the player.
        seasons (List[int]): A list of seasons to get statistics for (4-digit years,
            e.g., [2021, 2022] or [2023]).
        league_id (Optional[int]): The ID of the league.

    **Returns:**
        Dict[str, Any]: A dictionary containing the player statistics or error messages.  Key fields:

            * "player_statistics" (List[Dict[str, Any]]):  A list of dictionaries where each
              dictionary contains statistics for a single season.
            * "error" (str): An error is returned if the API key is missing, a season
              is invalid, or if no statistics are found.

            Each dictionary in "player_statistics" contains detailed statistics, grouped
            by category ("player", "team", "league", "games", "substitutes", "shots",
            "goals", "passes", "tackles", "duels", "dribbles", "fouls", "cards", "penalty").
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    if isinstance(seasons, int):
        seasons = [seasons]

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    url = f"{base_url}/players"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key,
    }
    all_stats = []

    for current_season in seasons:
        params: Dict[str, Any] = {"id": player_id, "season": current_season}
        if league_id:
            params["league"] = league_id

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("response"):
                continue

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
            return {"error": f"Request failed for season {current_season}: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred for season {current_season}: {e}"}


    if not all_stats:
        return {
            "error": f"No statistics found for player ID {player_id} for the specified seasons/league."
        }

    return {"player_statistics": all_stats}


@mcp.tool()
def get_team_fixtures(team_name: str, type: str = "upcoming", limit: int = 5) -> Dict[str, Any]:
    """Given a team name, returns either the last N or the next N fixtures for that team.

    **Args:**

        team_name (str): The team's name to search for. Must be >= 3 characters.
        type (str, optional): Either 'past' or 'upcoming' fixtures. Defaults to 'upcoming'.
        limit (int, optional): How many fixtures to retrieve.  Defaults to 5.

    **Returns:**

        Dict[str, Any]: A dictionary containing the fixture data or an error message. Key fields:
            * "response" (List[Dict[str,Any]]): List of fixtures, if found.
            * "error" (str): Error message if the request failed, or the team wasn't found.

        The structure of each fixture in `response` is the raw JSON response from the API.

    **Example:**

        ```python
        get_team_fixtures(team_name="Manchester United", type="past", limit=3)
        ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(team_name.strip()) < 3:
         return {"error": "The team name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    # Step 1: Find the Team ID
    search_url = f"{base_url}/teams"
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
        fixtures_url = f"{base_url}/fixtures"
        fixtures_params = {"team": team_id}

        if type.lower() == "past":
            fixtures_params["last"] = limit
        elif type.lower() == "upcoming":
            fixtures_params["next"] = limit
        else:
             return {"error": "The 'type' parameter must be either 'past' or 'upcoming'."}

        fixtures_resp = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
        fixtures_resp.raise_for_status()
        return fixtures_resp.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_fixture_statistics(fixture_id: int) -> Dict[str, Any]:
    """Retrieves detailed statistics for a specific fixture (game).

    **Args:**

        fixture_id (int): The numeric ID of the fixture/game.

    **Returns:**

        Dict[str, Any]:  A dictionary containing fixture statistics or an error message.  Key fields:
            * "response" (List[Dict[str, Any]]): List of team statistics, if found.
            * "error" (str): Error message, if any occurred.

        The structure of the data within `response` is the raw API response.

    **Example:**

    ```python
    get_fixture_statistics(fixture_id=867946)
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    url = f"{base_url}/fixtures/statistics"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    params = {"fixture": fixture_id}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_team_fixtures_by_date_range(team_name: str, from_date: str, to_date: str, season: str) -> Dict[str, Any]:
    """Retrieve all fixtures for a given team within a date range.

    **Args:**

        team_name (str): Team name to search for (e.g. 'Arsenal', 'Barcelona').
        from_date (str): Start date in YYYY-MM-DD format (e.g. '2023-08-01').
        to_date (str): End date in YYYY-MM-DD format (e.g. '2023-08-31').
        season (str):  Season in YYYY format.

    **Returns:**
        Dict[str, Any]: A dictionary containing the fixture data or an error message. Key fields:
            * "response" (List[Dict[str, Any]]):  A list of fixture dictionaries.
            * "error" (str): An error message.

        The structure of each dictionary in `response` is the raw API response.

    **Example:**

        ```python
        get_team_fixtures_by_date_range(
            team_name="Liverpool", from_date="2023-09-01", to_date="2023-09-30", season="2023"
        )
        ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(team_name.strip()) < 3:
        return {"error": "The team name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    # Step 1: find team ID
    teams_url = f"{base_url}/teams"
    teams_params = {"search": team_name}
    try:
        resp = requests.get(teams_url, headers=headers, params=teams_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("response"):
            return {"error": f"No team found matching '{team_name}'."}
        team_id = data["response"][0]["team"]["id"]

        # Step 2: fetch fixtures in date range
        fixtures_url = f"{base_url}/fixtures"
        fixtures_params = {
            "team": team_id,
            "from": from_date,
            "to": to_date,
            "season": season
        }
        resp_fixtures = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
        resp_fixtures.raise_for_status()
        return resp_fixtures.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
      return {"error":f"An unexpected error occurred: {e}"}
  

@mcp.tool()
def get_fixture_events(fixture_id: int) -> Dict[str, Any]:
    """Retrieves all in-game events for a given fixture ID (e.g. goals, cards, subs).

    **Args:**

        fixture_id (int): Numeric ID of the fixture whose events you want.

    **Returns:**

        Dict[str, Any]: A dictionary containing the fixture events or an error message. Key fields:
            * "response" (List[Dict[str, Any]]): List of events, if found.
            * "error" (str): Error message if the request failed.

        The structure of the data within `response` is the raw API response.

    **Example:**

    ```python
    get_fixture_events(fixture_id=867946)
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    url = f"{base_url}/fixtures/events"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    params = {"fixture": fixture_id}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_multiple_fixtures_stats(fixture_ids: List[int]) -> Dict[str, Any]:
    """Retrieves stats (shots, possession, etc.) for multiple fixtures at once.

    **Args:**
      fixture_ids (List[int]): A list of numeric fixture IDs.

    **Returns:**
        Dict[str, Any]: A dictionary containing the statistics for each fixture, or error messages. Key fields:
            * "fixtures_statistics" (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary contains the stats for a fixture (keyed by fixture ID) or an error for that fixture.

    **Example:**

        ```python
        get_multiple_fixtures_stats(fixture_ids=[867946, 867947, 867948])
        ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }
    combined_results = []

    for f_id in fixture_ids:
        try:
            url = f"{base_url}/fixtures/statistics"
            params = {"fixture": f_id}
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            combined_results.append({f_id: data})
        except requests.exceptions.RequestException as e:
            combined_results.append({f_id: {"error": f"Request failed: {e}"}})
        except Exception as e:
            combined_results.append({f_id: {"error": f"An unexpected error occurred: {e}"}})

    return {"fixtures_statistics": combined_results}

@mcp.tool()
def get_league_schedule_by_date(league_name: str, date: List[str], season: str) -> Dict[str, Any]:
    """Retrieves the schedule (fixtures) for a given league on one or multiple specified dates.

    **Args:**

        league_name (str): Name of the league (e.g., 'Premier League', 'La Liga').
        date (List[str]): List of dates in YYYY-MM-DD format (e.g., ['2024-03-08', '2024-03-09']).
        season (str): Season in YYYY format (e.g., '2023').

    **Returns:**

        Dict[str, Any]: A dictionary where each key is a date from the input `date` list,
            and the value is the API response for that date, or an error message.

    **Example:**

    ```python
    get_league_schedule_by_date(
        league_name="Premier League", date=["2024-03-08", "2024-03-09"], season="2023"
    )
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(league_name.strip()) < 3:
        return {"error": "The league name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    # Step 1: Get league ID by searching name
    try:
        leagues_url = f"{base_url}/leagues"
        leagues_params = {"search": league_name, "season": season}  # Include season in league search
        resp = requests.get(leagues_url, headers=headers, params=leagues_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("response"):
            return {"error": f"No leagues found matching '{league_name}' for season {season}."}

        # Find the correct league and season
        league_id = None
        for league_data in data["response"]:
             if league_data["league"]["name"].lower() == league_name.lower():
                for league_season in league_data["seasons"]:
                    if str(league_season["year"]) == season:
                        league_id = league_data["league"]["id"]
                        break
                if league_id:
                    break
        if not league_id:
            return {"error": f"Could not find {league_name} for season {season}."}


        results = {}
        for match_date in date:
            # Step 2: Get fixtures for that league & date
            fixtures_url = f"{base_url}/fixtures"
            fixtures_params = {
                "league": league_id,
                "date": match_date,
                "season": season
            }

            resp_fixtures = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
            resp_fixtures.raise_for_status()

            results[match_date] = resp_fixtures.json()  # Store results per date

        return results  # Return structured results with dates as keys

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_live_match_for_team(team_name: str) -> Dict[str, Any]:
    """Checks if a given team is currently playing live.

    **Args:**

        team_name (str): The team's name. Example: 'Arsenal'. Must be >= 3 chars.

    **Returns:**

        Dict[str, Any]:  If a live match is found, returns a dictionary with a "live_fixture"
            key containing the fixture data. If no live match is found, returns a dictionary
            with a "message" key. If an error occurs, returns a dictionary with an "error" key.

    **Example:**

    ```python
    get_live_match_for_team(team_name="Chelsea")
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(team_name.strip()) < 3:
        return {"error": "The team name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    # Step 1: find team ID
    try:
        teams_resp = requests.get(
            f"{base_url}/teams",
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
            f"{base_url}/fixtures",
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

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_live_stats_for_team(team_name: str) -> Dict[str, Any]:
    """Retrieves live in-game stats for a team currently in a match.

    **Args:**
        team_name (str): Team name to get live stats for. e.g., 'Arsenal'.

    **Returns:**

        Dict[str, Any]:  If the team is playing live, returns a dictionary containing
            the `fixture_id` and `live_stats`.  If no live match is found, returns
            a dictionary with a "message" key.  If an error occurs, returns a dictionary
            with an "error" key.

    **Example:**

    ```python
    get_live_stats_for_team(team_name="Liverpool")
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(team_name.strip()) < 3:
        return {"error": "The team name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    try:
        # Step 1: get team ID
        teams_resp = requests.get(
            f"{base_url}/teams",
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
            f"{base_url}/fixtures",
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
            f"{base_url}/fixtures/statistics",
            headers=headers,
            params={"fixture": fixture_id},
            timeout=15
        )
        stats_resp.raise_for_status()
        stats_data = stats_resp.json()

        return {"fixture_id": fixture_id, "live_stats": stats_data}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_live_match_timeline(team_name: str) -> Dict[str, Any]:
    """Retrieves the real-time timeline of events for a team's current live match.

    **Args:**

        team_name (str): Team name.

    **Returns:**

        Dict[str, Any]: If the team is playing live, returns a dictionary containing
            `fixture_id` and `timeline_events`. If not, returns a dictionary with a "message" key.
            If an error occurs, it returns a dictionary with an "error" key.

    **Example:**

    ```python
    get_live_match_timeline(team_name="Manchester City")
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(team_name.strip()) < 3:
        return {"error": "The team name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    try:
        # Step 1: team ID
        teams_resp = requests.get(
            f"{base_url}/teams",
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
            f"{base_url}/fixtures",
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
            f"{base_url}/fixtures/events",
            headers=headers,
            params={"fixture": fixture_id},
            timeout=15
        )
        events_resp.raise_for_status()
        events_data = events_resp.json()

        return {"fixture_id": fixture_id, "timeline_events": events_data}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_league_info(league_name: str) -> Dict[str, Any]:
    """Retrieve information about a specific football league.

    **Args:**

        league_name (str): Name of the league (e.g., 'Champions League').

    **Returns:**

        Dict[str, Any]:  A dictionary containing league information or an error message.  Key fields:
            *  "response" (List[Dict[str,Any]]): A list of leagues that match the search, if found.
            *  "error" (str): An error message if the request fails or no leagues are found.

        The structure of data in "response" is the raw API response.

    **Example:**

    ```python
    get_league_info(league_name="Premier League")
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(league_name.strip()) < 3:
        return {"error": "The league name must be at least 3 characters long."}


    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    # Fetch league information
    league_url = f"{base_url}/leagues"
    params = {"search": league_name}
    try:
        resp = requests.get(league_url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("response"):
          return {"error": f"No leagues found matching '{league_name}'."}
        return data
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_team_info(team_name: str) -> Dict[str, Any]:
    """Retrieve basic information about a specific football team.

    **Args:**

        team_name (str): Name of the team (e.g., 'Manchester United').

    **Returns:**
        Dict[str, Any]: A dictionary containing team information or an error message. Key fields:
          * "response" (List[Dict[str,Any]]): List of teams that match the search name.
          * "error" (str): If the API request failed or the team is not found.

        The structure of data in "response" is the raw API response.
    **Example:**

    ```python
    get_team_info(team_name="Real Madrid")
    ```
    """
    api_key = os.getenv("RAPID_API_KEY_FOOTBALL")
    if not api_key:
        return {"error": "RAPID_API_KEY_FOOTBALL environment variable not set."}
    if len(team_name.strip()) < 3:
      return {"error": "The team name must be at least 3 characters long."}

    base_url = "https://api-football-v1.p.rapidapi.com/v3"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    # Fetch team information
    teams_url = f"{base_url}/teams"
    teams_params = {"search": team_name}
    try:
        resp = requests.get(teams_url, headers=headers, params=teams_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("response"):
            return {"error": f"No team found matching '{team_name}'."}
        return data
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except Exception as e:
      return {"error": f"An unexpected error occurred: {e}"}
  

if __name__ == "__main__":
    try:
        print("Starting MCP server 'soccer_server' on 127.0.0.1:5000")
        # Use this approach to keep the server running
        mcp.run()
    except Exception as e:
        print(f"Error: {e}")
        # Sleep before exiting to give time for error logs
        time.sleep(5)