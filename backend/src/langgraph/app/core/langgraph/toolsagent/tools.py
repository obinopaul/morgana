"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast, Dict, Literal, Union
import base64
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfWriter, PdfReader
from geopy import Nominatim
import math 
import httpx
from dateutil.parser import parse as parse_datetime
# Assuming fast_flights module is correctly installed and accessible
from fast_flights import FlightData, Passengers, Result, get_flights

import os
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Iterator
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field, validator

# LangChain Community
from langchain_community.document_loaders import NeedleLoader
from langchain_community.retrievers import NeedleRetriever
from datetime import datetime
import pyairbnb

# from crewai_tools import BaseTool
from typing import Optional
from os import environ
from langchain.tools import BaseTool, Tool
import requests
from pydantic import Field, BaseModel  
import logging
import aiohttp
import asyncio
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()
      

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------------------------------------------
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

            # Perform the search (synchronously)
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

tavily_search_tool = Tool(
    name="Tavily Search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)

#----------------------------------------------------------------------------------------------------------------------------

# Define Input Schema
class WeatherSearchInput(BaseModel):
    location: str = Field(..., description="The location to get weather information for (e.g., 'New York').")
    date: Optional[str] = Field(..., description="The date for the weather forecast in YYYY-MM-DD format.")

# Define the WeatherSearchTool class
class WeatherSearchTool:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is missing.")

    def get_weather(self, input: WeatherSearchInput) -> str:
        try:
            if input.date:
                # Use forecast endpoint for future dates
                forecast_url = f"{self.base_url}/forecast"
                forecast_params = {
                    "q": input.location,
                    "appid": self.api_key,
                    "units": "metric"
                }
                response = requests.get(forecast_url, params=forecast_params)
                response.raise_for_status()
                forecast_data = response.json()

                target_date = datetime.strptime(input.date, "%Y-%m-%d").date()
                for forecast in forecast_data.get('list', []):
                    forecast_date = datetime.fromtimestamp(forecast['dt']).date()
                    if forecast_date == target_date:
                        weather = forecast['weather'][0]['description']
                        temp_min = forecast['main']['temp_min']
                        temp_max = forecast['main']['temp_max']
                        humidity = forecast['main']['humidity']
                        return (
                            f"Weather in {input.location} on {input.date}:\n"
                            f"Description: {weather}\n"
                            f"Temperature: {temp_min}°C to {temp_max}°C\n"
                            f"Humidity: {humidity}%"
                        )
                return f"No weather data found for {input.location} on {input.date}."
            else:
                # Use weather endpoint for current weather
                weather_url = f"{self.base_url}/weather"
                weather_params = {
                    "q": input.location,
                    "appid": self.api_key,
                    "units": "metric"
                }
                response = requests.get(weather_url, params=weather_params)
                response.raise_for_status()
                weather_data = response.json()

                weather = weather_data['weather'][0]['description']
                temp = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                return (
                    f"Current weather in {input.location}:\n"
                    f"Description: {weather}\n"
                    f"Temperature: {temp}°C\n"
                    f"Humidity: {humidity}%"
                )
        except Exception as e:
            return f"An error occurred: {str(e)}"


# Define LangChain Tool without coroutine (since it's now synchronous)
weather_tool = Tool(
    name="Weather Search",
    func=WeatherSearchTool().get_weather,
    description="Provides weather information for a given location and date using the OpenWeatherMap API.",
    args_schema=WeatherSearchInput
)



# ---------------------------------------------------------------------------- #
#                      WHATSAPP CONNECTION AND FUNCTIONS                       #
# ---------------------------------------------------------------------------- #
# This section contains the core logic for connecting to and interacting with  #
# the WhatsApp API, as provided in the prompt.                                 #
# ---------------------------------------------------------------------------- #

class WhatsAppConnection:
    """Handles connection and authentication with the WPPConnect server."""
    def __init__(self):
        self.base_url = os.getenv("WPPCONNECT_BASE_URL")
        self.session = os.getenv("WPPCONNECT_SESSION_NAME")
        self.secret_key = os.getenv("WPPCONNECT_SECRET_KEY")
        self.token = os.getenv("WPPCONNECT_TOKEN")

        if not all([self.base_url, self.session, self.secret_key, self.token]):
            raise ValueError(
                "One or more WhatsApp environment variables are not set. "
                "Please set WPPCONNECT_BASE_URL, WPPCONNECT_SESSION_NAME, "
                "WPPCONNECT_SECRET_KEY, and WPPCONNECT_TOKEN."
            )
        self.base_url = self.base_url.rstrip("/")


    def __enter__(self):
        # The connection logic doesn't require a special setup for each call,
        # but the context manager pattern is kept for consistency.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No special teardown needed.
        pass


def send_message(message: str, phone_number: str) -> Dict:
    """Sends a WhatsApp text message to a specified phone number."""
    if not phone_number:
        raise ValueError("Missing 'phone_number'. This field is required.")

    try:
        with WhatsAppConnection() as conn:
            url = f"{conn.base_url}/api/{conn.session}/send-message"
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {conn.token}",
            }
            data = {"phone": phone_number, "message": message, "isGroup": False}
            
            logger.info(f"Sending message to {phone_number}...")
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            logger.info("Message sent successfully.")
            return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"status": "error", "message": str(e)}


def send_voice(audio_path: str, phone_number: str) -> Dict:
    """Sends a WhatsApp voice message from an audio file to a specified phone number."""
    if not phone_number:
        raise ValueError("Missing 'phone_number'. This field is required.")
    if not audio_path:
        raise ValueError("Missing 'audio_path'. This field is required.")

    # Convert audio file to base64
    try:
        with open(audio_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    except FileNotFoundError:
        error_msg = f"Audio file not found at path: {audio_path}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"Error reading audio file: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

    try:
        with WhatsAppConnection() as conn:
            url = f"{conn.base_url}/api/{conn.session}/send-voice-base64"
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {conn.token}",
            }
            data = {
                "phone": phone_number,
                "isGroup": False,
                "base64Ptt": f"data:audio/mpeg;base64,{base64_audio}",
            }

            logger.info(f"Sending voice message from {audio_path} to {phone_number}...")
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            logger.info("Voice message sent successfully.")
            return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending WhatsApp voice message: {e}")
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------- #
#                        LANGCHAIN STRUCTURED TOOL CREATION                    #
# ---------------------------------------------------------------------------- #
# This section defines the Pydantic models for input validation and creates   #
# the structured LangChain tools that an agent can use.                        #
# ---------------------------------------------------------------------------- #

## Send Text Message Tool
class SendMessageInput(BaseModel):
    """Input schema for the Send WhatsApp Message tool."""
    message: str = Field(..., description="The text message content to be sent.")
    phone_number: str = Field(..., description="The recipient's phone number in international format (e.g., 15551234567).")

send_whatsapp_message_tool = Tool(
    name="send_whatsapp_message",
    func=send_message,
    description="Use this tool to send a WhatsApp text message to a specific phone number. It requires the message content and the recipient's phone number.",
    args_schema=SendMessageInput
)

## Send Voice Message Tool
class SendVoiceMessageInput(BaseModel):
    """Input schema for the Send WhatsApp Voice Message tool."""
    audio_path: str = Field(..., description="The local file path to the audio file (e.g., /path/to/voice_note.mp3) to be sent as a voice message.")
    phone_number: str = Field(..., description="The recipient's phone number in international format (e.g., 15551234567).")

send_whatsapp_voice_tool = Tool(
    name="send_whatsapp_voice_message",
    func=send_voice,
    description="Use this tool to send a WhatsApp voice message. It requires the local file path of the audio and the recipient's phone number.",
    args_schema=SendVoiceMessageInput
)



#----------------------------------------------------------------------------------------------------------------------------

import googlemaps
from googlemaps.convert import decode_polyline


#----------------------------------------------------------------------------------------------------------------------------

import os
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Define Input Schema
# Define Input Schema
class FlightSearchInput(BaseModel):
    departure_id: str = Field(..., description="The departure airport code or location kgmid.")
    arrival_id: str = Field(..., description="The arrival airport code or location kgmid.")
    outbound_date: str = Field(..., description="The outbound date in YYYY-MM-DD format.")
    return_date: str = Field(..., description="The return date in YYYY-MM-DD format (optional for one-way flights).")
    currency: str = Field(default="USD", description="The currency for the flight prices.")
    hl: str = Field(default="en", description="The language for the search results.")
    adults: int = Field(default=1, description="The number of adult passengers.")
    children: int = Field(default=0, description="The number of child passengers.")
    infants_in_seat: int = Field(default=0, description="The number of infants in seat.")
    infants_on_lap: int = Field(default=0, description="The number of infants on lap.")
    travel_class: int = Field(default=1, description="The travel class (1: Economy, 2: Premium Economy, 3: Business, 4: First).")
    sort_by: int = Field(default=1, description="The sorting order of the results (1: Top flights, 2: Price, etc.).")
    deep_search: bool = Field(default=False, description="Enable deep search for more precise results.")

# Define the Tool
class GoogleFlightsSearchTool:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SerpApi API key is missing. Please set the SERPAPI_API_KEY environment variable.")
        self.base_url = "https://serpapi.com/search.json"

    def _extract_flight_details(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "airlines": [leg["airline"] for leg in flight.get("flights", [])],
            "price": flight.get("price"),
            "departure_airport": flight.get("flights", [{}])[0].get("departure_airport", {}).get("name"),
            "arrival_airport": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("name"),
            "departure_time": flight.get("flights", [{}])[0].get("departure_airport", {}).get("time"),
            "arrival_time": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("time"),
            "total_duration": flight.get("total_duration"),
            "layovers": [
                {
                    "duration": layover.get("duration"),
                    "airport": layover.get("name"),
                    "overnight": layover.get("overnight", False),
                }
                for layover in flight.get("layovers", [])
            ],
            "travel_class": flight.get("flights", [{}])[0].get("travel_class"),
            "carbon_emissions": flight.get("carbon_emissions", {}).get("this_flight"),
            "booking_token": flight.get("booking_token"),
            "departure_token": flight.get("departure_token"),
        }

    async def search_flights(self, input: FlightSearchInput) -> Dict[str, Any]:
        params = {
            "engine": "google_flights",
            "departure_id": input.departure_id,
            "arrival_id": input.arrival_id,
            "outbound_date": input.outbound_date,
            "currency": input.currency,
            "hl": input.hl,
            "adults": input.adults,
            "children": input.children,
            "infants_in_seat": input.infants_in_seat,
            "infants_on_lap": input.infants_on_lap,
            "travel_class": input.travel_class,
            "sort_by": input.sort_by,
            "deep_search": "true" if input.deep_search else "false",
            "api_key": self.api_key,
        }
        
        if input.return_date:
            params["return_date"] = input.return_date

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    results = await response.json()

                    best_flights = [self._extract_flight_details(flight) for flight in results.get("best_flights", [])]
                    other_flights = [self._extract_flight_details(flight) for flight in results.get("other_flights", [])]

                    return {
                        "best_flights": best_flights,
                        "other_flights": other_flights,
                        "search_metadata": results.get("search_metadata", {}),
                        "search_parameters": results.get("search_parameters", {}),
                    }
            except Exception as e:
                return {"error": f"An error occurred: {str(e)}"}

flight_tool_instance = GoogleFlightsSearchTool()

google_flight_tool = Tool(
    name="google_flight_tool",
    func=flight_tool_instance.search_flights,
    coroutine=flight_tool_instance.search_flights,
    description="Provides flight information between two locations, including airlines, prices, departure/arrival times, and more.",
    args_schema=FlightSearchInput
)
        
#-------------------------------------------------------------- Google Flights Scraper --------------------------------------------------------------

class FlightSearchInput_2(BaseModel):
    """
    Input schema for searching flights.
    """
    departure_airport: str = Field(..., description="The departure airport code (e.g., LAX).")
    arrival_airport: str = Field(..., description="The arrival airport code (e.g., NYC).")
    departure_date: str = Field(..., description="The departure date in YYYY-MM-DD format.")
    return_date: Optional[str] = Field(..., description="The return date in YYYY-MM-DD format (optional for one-way flights).")
    adults: int = Field(default=1, description="The number of adults.")
    children: int = Field(default=0, description="The number of children.")
    travel_class: str = Field(default="all", description="The travel class (economy, business, first, or all).")
    sort_by: str = Field(default="price", description="Sort results by (price, duration, departure, arrival).")


class GoogleFlightsTool:
    def __init__(self):
        pass

    def _sort_flights(self, flights: List[Dict], sort_by: str) -> List[Dict]:
        """
        Sort flights by price, duration, departure, or arrival.
        """
        if sort_by == "price":
            # Remove currency symbols and commas, then convert to float for sorting
            return sorted(
                flights, 
                key=lambda x: float(x.get("price", "0").replace('$', '').replace(',', ''))
            )
        elif sort_by == "duration":
            # Assuming duration is in format "X hr Y min"
            def duration_in_minutes(flight):
                parts = flight.get("duration", "0 hr 0 min").split()
                hours = int(parts[0]) if len(parts) > 0 else 0
                minutes = int(parts[2]) if len(parts) > 2 else 0
                return hours * 60 + minutes
            return sorted(flights, key=duration_in_minutes)
        elif sort_by == "departure":
            return sorted(
                flights, 
                key=lambda x: datetime.strptime(x.get("departure_time", ""), "%I:%M %p on %a, %b %d, %Y")
            )
        elif sort_by == "arrival":
            return sorted(
                flights, 
                key=lambda x: datetime.strptime(x.get("arrival_time", ""), "%I:%M %p on %a, %b %d, %Y")
            )
        return flights

    def _structure_flight_data(self, result: Result, from_airport: str, to_airport: str, year: int, travel_class: str) -> List[Dict]:
        """
        Structure flight data into a list of dictionaries, including airport codes and year in time strings.
        """
        flights = []
        for flight in result.flights:
            flight_dict = {
                "airline": flight.name,
                "departure_time": f"{flight.departure}, {year}",
                "arrival_time": f"{flight.arrival}, {year}",
                "departure_airport": from_airport,
                "arrival_airport": to_airport,
                "duration": flight.duration,
                "stops": flight.stops,
                "price": flight.price,
                "travel_class": travel_class,
            }
            flights.append(flight_dict)
        return flights

    async def search_flights(self, input: FlightSearchInput_2) -> List[Dict]:
        """
        Search for flights and return a list of dictionaries.
        """
        try:
            structured_output = []

            # Prepare Passengers object
            passengers_obj = Passengers(
                adults=input.adults,
                children=input.children,
                infants_in_seat=0,
                infants_on_lap=0
            )

            # Extract year from departure_date
            departure_year = int(input.departure_date.split('-')[0])

            # --- Departure Flights ---
            departure_flight_data = [
                FlightData(
                    date=input.departure_date,
                    from_airport=input.departure_airport,
                    to_airport=input.arrival_airport
                )
            ]

            departure_result: Result = get_flights(
                flight_data=departure_flight_data,
                trip="one-way",
                seat=input.travel_class,
                passengers=passengers_obj,
                fetch_mode="fallback",
            )

            departure_flights = self._structure_flight_data(
                departure_result, 
                from_airport=input.departure_airport, 
                to_airport=input.arrival_airport,
                year=departure_year,
                travel_class=input.travel_class
            )
            departure_flights = self._sort_flights(departure_flights, input.sort_by)

            structured_output.append({"departure flights": departure_flights})

            # --- Return Flights (if return_date is provided) ---
            if input.return_date:
                arrival_year = int(input.return_date.split('-')[0])
                arrival_flight_data = [
                    FlightData(
                        date=input.return_date,
                        from_airport=input.arrival_airport,
                        to_airport=input.departure_airport
                    )
                ]

                arrival_result: Result = get_flights(
                    flight_data=arrival_flight_data,
                    trip="one-way",
                    seat=input.travel_class,
                    passengers=passengers_obj,
                    fetch_mode="fallback",
                )

                arrival_flights = self._structure_flight_data(
                    arrival_result, 
                    from_airport=input.arrival_airport, 
                    to_airport=input.departure_airport,
                    year=arrival_year,
                    travel_class=input.travel_class
                )
                arrival_flights = self._sort_flights(arrival_flights, input.sort_by)

                structured_output.append({"arrival flights": arrival_flights})

            return structured_output

        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]

# Initialize the tool instance
google_flight_instance = GoogleFlightsTool()

# Create the tool with both sync and async capabilities
google_flight_search = Tool(
    name="google_flight_search",
    func=google_flight_instance.search_flights,
    coroutine=google_flight_instance.search_flights,
    description="Provides flight information between two locations, including airlines, prices, departure/arrival times, and more.",
    args_schema=FlightSearchInput_2
)


#----------------------------------------------------------------------------------------------------------------------------------
# Input schema definition
class GoogleFlightsToolSync:
    def __init__(self):
        pass

    def _sort_flights(self, flights: List[Dict], sort_by: str) -> List[Dict]:
        """
        Sort flights by price, duration, departure, or arrival.
        """
        if sort_by == "price":
            # Remove currency symbols and commas, then convert to float for sorting
            return sorted(
                flights, 
                key=lambda x: float(x.get("price", "0").replace('$', '').replace(',', ''))
            )
        elif sort_by == "duration":
            # Assuming duration is in format "X hr Y min"
            def duration_in_minutes(flight):
                parts = flight.get("duration", "0 hr 0 min").split()
                hours = int(parts[0]) if len(parts) > 0 else 0
                minutes = int(parts[2]) if len(parts) > 2 else 0
                return hours * 60 + minutes
            return sorted(flights, key=duration_in_minutes)
        elif sort_by == "departure":
            return sorted(
                flights, 
                key=lambda x: datetime.strptime(x.get("departure_time", ""), "%I:%M %p on %a, %b %d, %Y")
            )
        elif sort_by == "arrival":
            return sorted(
                flights, 
                key=lambda x: datetime.strptime(x.get("arrival_time", ""), "%I:%M %p on %a, %b %d, %Y")
            )
        return flights

    def _structure_flight_data(self, result: Result, from_airport: str, to_airport: str, year: int, travel_class: str) -> List[Dict]:
        """
        Structure flight data into a list of dictionaries, including airport codes and year in time strings.
        """
        flights = []
        for flight in result.flights:
            flight_dict = {
                "airline": flight.name,
                "departure_time": f"{flight.departure}, {year}",
                "arrival_time": f"{flight.arrival}, {year}",
                "departure_airport": from_airport,
                "arrival_airport": to_airport,
                "duration": flight.duration,
                "stops": flight.stops,
                "price": flight.price,
                "travel_class": travel_class,
            }
            flights.append(flight_dict)
        return flights

    def search_flights(self, input: FlightSearchInput_2) -> List[Dict]:
        """
        Search for flights and return a list of dictionaries.
        """
        try:
            structured_output = []

            # Prepare Passengers object
            passengers_obj = Passengers(
                adults=input.adults,
                children=input.children,
                infants_in_seat=0,
                infants_on_lap=0
            )

            # Extract year from departure_date
            departure_year = int(input.departure_date.split('-')[0])

            # --- Departure Flights ---
            departure_flight_data = [
                FlightData(
                    date=input.departure_date,
                    from_airport=input.departure_airport,
                    to_airport=input.arrival_airport
                )
            ]

            # Use travel class directly from input, defaults to economy if not specified
            departure_result: Result = get_flights(
                flight_data=departure_flight_data,
                trip="one-way",
                seat=input.travel_class,
                passengers=passengers_obj,
                fetch_mode="fallback",
            )

            departure_flights = self._structure_flight_data(
                departure_result, 
                from_airport=input.departure_airport, 
                to_airport=input.arrival_airport,
                year=departure_year,
                travel_class=input.travel_class
            )
            departure_flights = self._sort_flights(departure_flights, input.sort_by)

            structured_output.append({"departure flights": departure_flights})

            # --- Return Flights (if return_date is provided) ---
            if input.return_date:
                arrival_year = int(input.return_date.split('-')[0])
                arrival_flight_data = [
                    FlightData(
                        date=input.return_date,
                        from_airport=input.arrival_airport,
                        to_airport=input.departure_airport
                    )
                ]

                arrival_result: Result = get_flights(
                    flight_data=arrival_flight_data,
                    trip="one-way",
                    seat=input.travel_class,
                    passengers=passengers_obj,
                    fetch_mode="fallback",
                )

                arrival_flights = self._structure_flight_data(
                    arrival_result, 
                    from_airport=input.arrival_airport, 
                    to_airport=input.departure_airport,
                    year=arrival_year,
                    travel_class=input.travel_class
                )
                arrival_flights = self._sort_flights(arrival_flights, input.sort_by)

                structured_output.append({"arrival flights": arrival_flights})

            return structured_output

        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]

# Initialize the tool
google_flights_tool_sync = Tool(
    name="google_flights_tool_sync",
    func=GoogleFlightsToolSync().search_flights,
    description="Provides flight information between two locations, including airlines, prices, departure/arrival times, and more.",
    args_schema=FlightSearchInput_2
)

#----------------------------------------------------------------------------------------------------------------------------
# Define Input Schema
class BookingSearchInput(BaseModel):
    location: str = Field(..., description="The destination city or location (e.g., 'London').")
    checkin_date: str = Field(..., description="The check-in date in YYYY-MM-DD format.")
    checkout_date: str = Field(..., description="The check-out date in YYYY-MM-DD format.")
    adults: int = Field(default=2, description="The number of adult guests.")
    rooms: int = Field(default=1, description="The number of rooms.")
    currency: str = Field(default="USD", description="The currency for the prices.")


class BookingScraperTool:
    def __init__(self):
        self.base_url = "https://www.booking.com/searchresults.html"

    async def search(self, input: BookingSearchInput) -> List[Dict]:
        """
        Scrape hotel data from Booking.com based on the provided input parameters asynchronously.
        """
        params = {
            'ss': input.location,
            'dest_type': 'city',
            'checkin': input.checkin_date,
            'checkout': input.checkout_date,
            'group_adults': input.adults,
            'no_rooms': input.rooms,
            'selected_currency': input.currency
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params, headers=headers) as response:
                if response.status != 200:
                    return [{"error": f"Failed to fetch data, status code {response.status}"}]
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')

        results = []
        for card in soup.find_all('div', {'data-testid': 'property-card'}):
            try:
                name = card.find('div', {'data-testid': 'title'}).text.strip()

                # Handle multiple possible price selectors
                price_elem = None
                selectors = [
                    {'class': 'prco-valign-middle-helper'},
                    {'data-testid': 'price-and-discounted-price'},
                    {'data-id': 'price-box'}
                ]
                for selector in selectors:
                    price_elem = card.find(['span', 'div'], selector)
                    if price_elem:
                        break

                price = price_elem.text.strip() if price_elem else 'N/A'

                rating_elem = card.find('div', {'data-testid': 'review-score'})
                rating = rating_elem.text.strip() if rating_elem else 'N/A'

                link_element = card.find('a', {'data-testid': 'title-link'})
                link = link_element['href'] if link_element else 'N/A'
                if link != 'N/A' and not link.startswith('http'):
                    link = f"https://www.booking.com{link}"

                results.append({
                    'name': name,
                    'price': price,
                    'rating': rating,
                    'link': link
                })
            except Exception as e:
                print(f'Error parsing hotel card: {str(e)}')
                continue

        return results if results else [{"error": "No hotels found."}]


# ✅ Instantiate the scraper tool first
booking_scraper_instance = BookingScraperTool()

# ✅ Define the LangChain tool correctly
booking_tool = Tool(
    name="booking_tool",
    func=booking_scraper_instance.search,  # Pass instance method
    coroutine=booking_scraper_instance.search,  # Explicitly define the coroutine
    description="Scrapes hotel data from Booking.com based on destination, check-in/check-out dates, and other parameters.",
    args_schema=BookingSearchInput
)

#------------------------------------------------------------
# List of available tools
TOOLS: List[Callable[..., Any]] = [tavily_search_tool, booking_tool]
#------------------------------------------------------------

#---------------------------------------------------------- Places Tool----------------------------------------------------------

import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GoogleMapsPlacesInput(BaseModel):
    """
    Input schema for the Google Maps Places API tool (search, nearby, details, etc.).
    
    Note: This is a broad input; in practice, you might split this into specialized 
          tools for find_place, text_search, nearby_search, place_details, etc.
    """
    query: Optional[str] = Field(
        None,
        description="Text query to search for, e.g. 'pizza in New York'."
    )
    location: Optional[str] = Field(
        None,
        description="Lat/lng or 'place_id:...' for nearby search or find_place bias."
    )
    radius: Optional[int] = Field(
        None,
        description="Radius in meters for nearby or text search."
    )
    type: Optional[List[str]] = Field(  # Changed to List[str]
        None,
        description="List of types of place, e.g., ['restaurant', 'museum']."
    )
    language: Optional[str] = Field(
        None,
        description="Language code for the response."
    )
    min_price: Optional[int] = Field(
        0,
        description="Minimum price range (0 to 4)."
    )
    max_price: Optional[int] = Field(
        4,
        description="Maximum price range (0 to 4)."
    )
    open_now: Optional[bool] = Field(
        False,
        description="Whether to show only places open now."
    )
    rank_by: Optional[str] = Field(
        None,
        description="For nearby search: 'prominence' or 'distance'."
    )
    name: Optional[str] = Field(
        None,
        description="A term to be matched against place names."
    )
    page_token: Optional[str] = Field(
        None,
        description="Token for pagination of results."
    )
    # Additional: for place details
    place_id: Optional[str] = Field(
        None,
        description="Place ID for retrieving details."
    )
    fields: Optional[List[str]] = Field(
        None,
        description="List of place detail fields to return."
    )


class GoogleMapsPlacesTool:
    """
    A tool to call various Google Places methods via googlemaps.Client:
      - find_place(...)
      - places(...)
      - places_nearby(...)
      - place(...)
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.base_url = "https://maps.googleapis.com/maps/api/place"

    async def run_places_search(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        radius: Optional[int] = None,
        type: Optional[List[str]] = None,
        language: Optional[str] = None,
        min_price: Optional[int] = 0,
        max_price: Optional[int] = 4,
        open_now: Optional[bool] = False,
        rank_by: Optional[str] = None,
        name: Optional[str] = None,
        page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Example: text search (places) or nearby search if 'location' is set.
        """
        try:
            params = {
                "key": self.api_key,
                "query": query,
                "location": location,
                "radius": radius,
                "type": ",".join(type) if type else None,
                "language": language,
                "minprice": min_price,
                "maxprice": max_price,
                "opennow": open_now,
                "rankby": rank_by,
                "name": name,
                "pagetoken": page_token
            }
            params = {k: v for k, v in params.items() if v is not None}

            if location and rank_by == "distance":
                url = f"{self.base_url}/nearbysearch/json"
            elif location and radius:
                url = f"{self.base_url}/nearbysearch/json"
            else:
                url = f"{self.base_url}/textsearch/json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return {"places_search_result": response.json()}
        except Exception as e:
            return {"error": str(e)}

    async def run_find_place(
        self,
        query: str,
        input_type: str = "textquery",
        fields: Optional[List[str]] = None,
        location_bias: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wraps googlemaps.Client.find_place(...)
        """
        try:
            params = {
                "key": self.api_key,
                "input": query,
                "inputtype": input_type,
                "fields": ",".join(fields) if fields else None,
                "locationbias": location_bias,
                "language": language
            }
            params = {k: v for k, v in params.items() if v is not None}

            url = f"{self.base_url}/findplacefromtext/json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return {"find_place_result": response.json()}
        except Exception as e:
            return {"error": str(e)}

    async def run_place_details(
        self,
        place_id: str,
        fields: Optional[List[str]] = None,
        language: Optional[str] = None,
        reviews_no_translations: Optional[bool] = False,
        reviews_sort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wraps googlemaps.Client.place(...)
        """
        try:
            params = {
                "key": self.api_key,
                "place_id": place_id,
                "fields": ",".join(fields) if fields else None,
                "language": language,
                "reviews_no_translations": reviews_no_translations,
                "reviews_sort": reviews_sort
            }
            params = {k: v for k, v in params.items() if v is not None}

            url = f"{self.base_url}/details/json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return {"place_details_result": response.json()}
        except Exception as e:
            return {"error": str(e)}


# ✅ Instantiate the tool once to reuse the same instance
google_maps_tool_instance = GoogleMapsPlacesTool()

# ✅ Correctly define the tools for async execution
google_places_tool = Tool(
    name="google_places_tool",
    func=google_maps_tool_instance.run_places_search,
    coroutine=google_maps_tool_instance.run_places_search,
    description="Calls the Google Maps Places API for text search and nearby search.",
    args_schema=GoogleMapsPlacesInput
)

google_find_place_tool = Tool(
    name="google_find_place_tool",
    func=google_maps_tool_instance.run_find_place,
    coroutine=google_maps_tool_instance.run_find_place,
    description="Calls the Google Maps Find Place API to find places by text query.",
    args_schema=GoogleMapsPlacesInput
)

google_place_details_tool = Tool(
    name="google_place_details_tool",
    func=google_maps_tool_instance.run_place_details,
    coroutine=google_maps_tool_instance.run_place_details,
    description="Calls the Google Maps Place Details API to get detailed information about a place.",
    args_schema=GoogleMapsPlacesInput
)


#---------------------------------------------------------- TicketMaster ----------------------------------------------------------
class TicketmasterEventSearchInput(BaseModel):
    keyword: Optional[str] = Field(default=None, description="Keyword to search for events (e.g., artist, event name).")
    city: Optional[str] = Field(default=None, description="Filter events by city.")
    country_code: Optional[str] = Field(default=None, description="Filter events by country code (ISO Alpha-2 Code).")
    classification_name: Optional[str] = Field(default=None, description="Filter by classification (e.g., 'Music').")
    start_date_time: Optional[str] = Field(default=None, description="Start date filter in ISO8601 format (YYYY-MM-DDTHH:mm:ssZ).")
    end_date_time: Optional[str] = Field(default=None, description="End date filter in ISO8601 format (YYYY-MM-DDTHH:mm:ssZ).")
    size: int = Field(default=10, description="Number of events to return per page.")
    page: int = Field(default=0, description="Page number to retrieve.")
    sort: Optional[str] = Field(default="relevance,desc", description="Sorting order of the search results.")

    @field_validator('start_date_time', 'end_date_time')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """
        Validates that the provided datetime string conforms to the expected ISO8601 format.
        """
        # The validator only runs if the value is not None
        if v and "T" not in v:
            raise ValueError("Datetime must be in ISO8601 format (e.g., 'YYYY-MM-DDTHH:mm:ssZ').")
        return v

class TicketmasterAPITool:
    """
    Async Ticketmaster API tool to fetch events and event details.
    """

    BASE_URL = "https://app.ticketmaster.com/discovery/v2"

    def __init__(self):
        self.api_key = os.getenv("TICKETMASTER_API_KEY")
        if not self.api_key:
            raise ValueError("Ticketmaster API key is missing. Please set TICKETMASTER_API_KEY environment variable.")

    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Helper method to make async API requests to Ticketmaster.
        """
        params = params or {}
        params["apikey"] = self.api_key  # Add API key to request parameters
        url = f"{self.BASE_URL}/{endpoint}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                return {"error": f"HTTP Error {e.status}: {e.message}"}
            except aiohttp.ClientError as e:
                return {"error": f"Client error: {str(e)}"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}

    async def search_events(self, input: TicketmasterEventSearchInput) -> List[Dict[str, Any]]:
        """
        Asynchronously search for events using the Ticketmaster API.
        """
        params = {
            "keyword": input.keyword,
            "city": input.city,
            "countryCode": input.country_code,
            "classificationName": input.classification_name,
            "startDateTime": input.start_date_time,
            "endDateTime": input.end_date_time,
            "size": input.size,
            "page": input.page,
            "sort": input.sort,
        }

        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}

        # Fetch results
        data = await self._make_request("events.json", params=params)

        # Extract event results
        events = data.get("_embedded", {}).get("events", [])
        results = []
        for event in events:
            results.append({
                "Event": event.get("name"),
                "Date": event.get("dates", {}).get("start", {}).get("localDate"),
                "Time": event.get("dates", {}).get("start", {}).get("localTime"),
                "Venue": event["_embedded"]["venues"][0].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                "City": event["_embedded"]["venues"][0]["city"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                "Country": event["_embedded"]["venues"][0]["country"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                "Url": event.get("url"),
            })

        return results

    async def get_event_details(self, event_id: str) -> Dict[str, Any]:
        """
        Retrieve details for a specific event by its ID.
        """
        data = await self._make_request(f"events/{event_id}.json")

        # Extract event details
        event = data
        return {
            "Event": event.get("name"),
            "Date": event.get("dates", {}).get("start", {}).get("localDate"),
            "Time": event.get("dates", {}).get("start", {}).get("localTime"),
            "Venue": event["_embedded"]["venues"][0].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
            "City": event["_embedded"]["venues"][0]["city"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
            "Country": event["_embedded"]["venues"][0]["country"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
            "Url": event.get("url"),
        }

ticketmaster_tool = Tool(
    name="ticketmaster_tool",
    func=TicketmasterAPITool().search_events,  # Sync-compatible version
    coroutine=TicketmasterAPITool().search_events,  # Explicit async coroutine
    description="Searches for events using the Ticketmaster API.",
    args_schema=TicketmasterEventSearchInput,
)

#---------------------------------------------------------- AirBnB Tools ----------------------------------------------------------
class AirbnbSearchInput(BaseModel):
    location: str = Field(..., description="The destination city or area (e.g., 'Brooklyn' or 'New York City').")
    checkin_date: str = Field(..., description="The check-in date in YYYY-MM-DD format.")
    checkout_date: str = Field(..., description="The check-out date in YYYY-MM-DD format.")
    currency: str = Field(default="USD", description="The currency for the prices.")
    margin_km: float = Field(default=5.0, description="Size (in km) of bounding box margin.")


# Define Airbnb Scraper Tool
class AirbnbScraperTool:
    def __init__(self):
        """
        Initialize the Airbnb scraper.
        """
        self.geolocator = Nominatim(user_agent="airbnb_search")

    async def _get_dynamic_bbox(self, location_name: str, margin_km: float):
        """
        Asynchronously geocode to get lat/long, then build a bounding box around the center.
        """
        loop = asyncio.get_running_loop()
        geocode_result = await loop.run_in_executor(None, self.geolocator.geocode, location_name)

        if not geocode_result:
            raise ValueError(f"Could not geocode location: {location_name}")

        center_lat = geocode_result.latitude
        center_lng = geocode_result.longitude

        lat_margin_deg = margin_km / 111.0
        lng_margin_deg = margin_km / (111.0 * abs(math.cos(math.radians(center_lat))) + 1e-9)

        ne_lat = center_lat + lat_margin_deg
        ne_lng = center_lng + lng_margin_deg
        sw_lat = center_lat - lat_margin_deg
        sw_lng = center_lng - lng_margin_deg
        zoom_value = 10  # Adjust as needed

        return ne_lat, ne_lng, sw_lat, sw_lng, zoom_value

    async def search(self, input_data: AirbnbSearchInput) -> List[Dict]:
        """
        Asynchronously perform an Airbnb search by dynamically constructing a bounding box.
        """
        # Get bounding box + zoom for the location
        ne_lat, ne_lng, sw_lat, sw_lng, zoom_val = await self._get_dynamic_bbox(
            input_data.location, input_data.margin_km
        )

        # Call pyairbnb.search_all asynchronously
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            pyairbnb.search_all,
            input_data.checkin_date,
            input_data.checkout_date,
            ne_lat,
            ne_lng,
            sw_lat,
            sw_lng,
            zoom_val,
            input_data.currency,
            ""
        )

        output = []
        for item in results:
            property_name = item.get("name", "N/A")

            # Extract nightly price
            price_info = item.get("price", {})
            unit_price = price_info.get("unit", {})
            currency_symbol = unit_price.get("currency_symbol", "")
            nightly_amount = unit_price.get("amount", "N/A")

            # Basic rating
            rating_info = item.get("rating", {})
            rating_value = rating_info.get("value", "N/A")

            # Construct a link from the room_id
            room_id = item.get("room_id", "")
            link = f"https://www.airbnb.com/rooms/{room_id}" if room_id else "N/A"

            output.append({
                "name": property_name,
                "price_per_night": f"{currency_symbol}{nightly_amount}",
                "rating": rating_value,
                "link": link
            })

        return output


# Define LangChain Tool
airbnb_tool = Tool(
    name="airbnb_tool",
    func=AirbnbScraperTool().search,
    coroutine=AirbnbScraperTool().search,  # Explicit async support
    description="Scrapes Airbnb listings based on location and check-in/check-out dates.",
    args_schema=AirbnbSearchInput
)






# A list of all local tools for easy import
base_tools = [
tavily_search_tool, weather_tool, send_whatsapp_voice_tool, send_whatsapp_message_tool, 
google_flight_tool, google_flight_search, booking_tool, google_places_tool, google_find_place_tool,
google_place_details_tool, ticketmaster_tool, airbnb_tool
]
