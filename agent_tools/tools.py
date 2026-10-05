from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, ToolMessage
import os, requests
from rich import print


# ***************** Step 1: Creating tool *****************

# Creating the get_weather tool
@tool
def get_weather(city: str) -> str:
    """
        Get the current weather information for a city.

        Args:
            city: Name of the city.

        Returns:
            Current temperature, feels-like temperature,
            humidity, wind speed and weather description.
        """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        return f"error: Could not fetch weather for {city}"

    data = response.json()
    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return (
        f"Weather in {city}: "
        f"{data['weather'][0]['description']}, "
        f"Temperature: {data['main']['temp']}°C, "
        f"Feels like: {data['main']['feels_like']}°C, "
        f"Humidity: {data['main']['humidity']}%, "
        f"Wind speed: {data['wind']['speed']} m/s"
    )


# print(get_weather.invoke("Mumbai"))

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# Creating a get_news tool
@tool
def get_news(city: str) -> str:
    """
    Search for recent news about a city.

    Args:
        city: Name of the city.

    Returns:
        A list of recent news articles.
    """

    response = tavily_client.search(
        query=f"latest news in {city} today",
        search_depth="basic",
        topic="news",
        max_results=5
    )

    results = response.get("results", [])

    if not results:
        return f"No recent news found for {city}"

    news = []

    for result in results:
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")

        news.append(
            f"- {title}\n 🔗{url}\n {content[:200]}..."
        )

    return f"Latest news in {city}: \n\n {'\n\n'.join(news)}"
