from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.recognition_service import recognize_faces


def recognize_face() -> None:
    recognize_faces()


if __name__ == "__main__":
    recognize_face()
