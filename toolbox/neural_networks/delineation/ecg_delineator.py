import json
import os
from pathlib import Path
import socket

from ...leads import DEFAULT_MEASUREMENT_LEAD


def ecg_delineator(ecg_path: str, lead: str = DEFAULT_MEASUREMENT_LEAD) -> dict:
    """Return P, QRS, and T as inclusive [onset, offset] sample indices for one lead."""
    endpoint = os.environ["EKO_DELINEATOR"]
    request = {"ecg_path": str(Path(ecg_path).expanduser().resolve()), "lead": lead}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(float(os.environ.get("EKO_PYTHON_TIMEOUT", "30")))
        connection.connect(endpoint)
        connection.sendall((json.dumps(request) + "\n").encode())
        with connection.makefile("rb") as reader:
            response = json.loads(reader.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]
