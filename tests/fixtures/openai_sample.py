"""Fixture: a few @function_tool decorated functions for scanner tests."""

from agents import function_tool


@function_tool
def create_payment(amount: int, currency: str) -> dict:
    """Charge the user; returns the new payment ID."""
    import requests
    return requests.post("https://api.example.com/pay", json={"amount": amount}).json()


@function_tool(strict_mode=False)
def process(data):
    return data


def not_a_tool(x: int) -> int:
    return x + 1
