"""Command-line entry point for running the tracker from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from object_detection.app import main


if __name__ == "__main__":
    main()
