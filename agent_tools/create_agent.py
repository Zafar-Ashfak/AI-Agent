from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_core.tools import tool
from tavily import TavilyClient
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
    return f"Weather in {city}: {desc}, {temp}°C"


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
        query=city,
        search_depth="basic",
        topic="news",
        max_results=3
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
            f"- {title}\n 🔗{url}\n {content}..."
        )

    return f"Latest news in {city}: \n\n {'\n\n'.join(news)}"


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)


model = get_llm()

SYSTEM_PROMPT = """
You are a helpful City Assistant.

- Use tools for current weather and recent city news.
- Never use tables or invent information.
- Keep weather concise and include important details.
- Put news under a separate "Latest News" section.
- Summarize each news item in 3-5 easy-to-read sentences.
- Include the exact source URL provided by the news tool.
- Keep Weather and Latest News clearly separated.
"""

agent = create_agent(
    model=model,
    tools=[get_weather, get_news],
    system_prompt=SYSTEM_PROMPT
)

print("City assistant agent")
print("type exit or quit to close the chat")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    result = agent.invoke({
        "messages": [
            {"role": "user",
             "content": user_input
             }
        ]
    })

    print(result['messages'][-1].content)
