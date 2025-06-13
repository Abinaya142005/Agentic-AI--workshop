import os
import streamlit as st
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# ===== API KEYS =====
os.environ["GOOGLE_API_KEY"] = "AIzaSyC8LU7hQObdyIo3UL0i8bP3h6oD1iFEGvQ"
os.environ["WEATHER_API_KEY"] = "c617b4405baf4eee9e172112251306"
os.environ["TAVILY_API_KEY"] = "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"

# ===== LLM Setup =====
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ===== Weather Tool =====
@tool
def get_weather(city: str) -> str:
    """Returns current weather for a city using WeatherAPI."""
    key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={key}&q={city}"
    try:
        res = requests.get(url)
        data = res.json()
        return (
            f"The weather in {city} is {data['current']['condition']['text']} "
            f"with {data['current']['temp_c']}°C, humidity {data['current']['humidity']}%, "
            f"and wind speed {data['current']['wind_kph']} kph."
        )
    except:
        return "Weather info not available."

# ===== Attractions Tool =====
@tool
def get_top_attractions(city: str) -> str:
    """Returns top tourist attractions in a city using Tavily."""
    try:
        search = TavilySearchResults(k=5)
        results = search.run(f"Top tourist attractions in {city}")
        return "Here are top attractions:\n\n" + "\n\n".join([r['content'] for r in results])
    except:
        return "Attraction info not available."

# ===== Tools & Agent =====
tools = [get_weather, get_top_attractions]

# 💡 agent_scratchpad is required!
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Use tools to help users with city travel plans."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ===== Streamlit App UI =====
st.title("🧳 Travel Assistant AI")
st.write("Get current weather and top attractions for your travel destination.")

city = st.text_input("Enter a city name")

if st.button("Get Info") and city:
    with st.spinner("Fetching information..."):
        result = agent_executor.invoke({
            "input": f"I want to travel to {city}. Tell me the weather and attractions."
        })
        st.subheader("Travel Info")
        st.write(result["output"])
