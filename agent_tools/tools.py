from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient
import os
import requests

from rich import print


# Creating the weather tool
@tool
def get_weather(city: str) -> dict[str, str] | None:
    """
    Get the current weather information for a city.

    Args:
        city: Name of the city.

    Returns:
        Current temperature, feels-like temperature,
        humidity, wind speed and weather description.
    """

    api_key = os.getenv(key="OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        return {
            "error": f"Could not fetch weather for {city}"
        }

    data = response.json()

    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "weather": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }

# Creating tavily news tool to get latest news
tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY"),
    )

@tool
def get_news(city : str) -> str:
    """
    Search for recent news about a city.

    Args:
        city: Name of the city.

    Returns:
        A list of recent news articles.
    """

    response = tavily_client.search(
        query=f"latest news of {city}",
        search_depth="basic",
        topic="news",
        max_results=3
    )

    news = []

    for result in response["results"]:
        news.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content")
        })

    return news


news_result = get_news.invoke("Mumbai")
print(news_result)