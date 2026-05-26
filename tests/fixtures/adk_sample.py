"""Fixture: ADK FunctionTool() wrappers around plain functions."""

from google.adk.tools import FunctionTool


def get_weather(city: str) -> dict:
    """Look up the current weather."""
    import requests
    return requests.get(f"https://api.example.com/weather/{city}").json()


def run():
    return "ok"


weather_tool = FunctionTool(get_weather)
run_tool = FunctionTool(run)
