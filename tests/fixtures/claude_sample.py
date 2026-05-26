"""Fixture: a few @tool decorated functions for scanner tests."""

from claude_agent_sdk import tool


@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email via SMTP."""
    import subprocess
    subprocess.run(["sendmail", to])
    return {"ok": True}


@tool
def lookup_user(user_id):
    return {"id": user_id}
