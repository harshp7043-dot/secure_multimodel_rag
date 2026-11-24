
import json

from datetime import datetime, timezone

from config import AUDIT_FILE


def audit(
    user,
    event,
    query="",
    sources=None,
    decision=""
):

    record = {
        "time": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),

        "user": (
            user["username"]
            if user else "anonymous"
        ),

        "role": (
            user["role"]
            if user else "anonymous"
        ),

        "event": event,
        "decision": decision,
        "query": query,
        "sources": sources or []
    }

    with open(AUDIT_FILE, "a") as file:

        file.write(
            json.dumps(record) + "\n"
        )
