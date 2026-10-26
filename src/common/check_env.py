"""Environment check for the Emotion Detecting Robot project.

Reports Python/OS details and whether key packages are installed.
This script only reads information; it installs nothing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys

# (import name, pip package name)
PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("tensorflow", "tensorflow"),
    ("sklearn", "scikit-learn"),
    ("librosa", "librosa"),
    ("sounddevice", "sounddevice"),
    ("matplotlib", "matplotlib"),
]


def check_package(import_name: str, pip_name: str) -> str:
    """Return a status string for one package without hiding the real error."""
    try:
        importlib.import_module(import_name)
    except ModuleNotFoundError as exc:
        if exc.name == import_name:
            return "NOT INSTALLED"
        return f"IMPORT FAILED (missing dependency '{exc.name}'): {exc}"
    except Exception as exc:  # installed but broken; report it, do not hide it
        return f"IMPORT FAILED: {type(exc).__name__}: {exc}"
    try:
        version = importlib.metadata.version(pip_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"OK (version {version})"

def tool_version(command: list[str]) -> str:
    """Return the first line of a tool's version output, or a not-found message."""
    if shutil.which(command[0]) is None:
        return "NOT FOUND"
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else "no output"


def main() -> None:
    print("=== Emotion Detecting Robot: Environment Check ===")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Python path    : {sys.executable}")
    print(f"OS             : {platform.system()} {platform.release()}")
    print(f"Architecture   : {platform.machine()}")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Virtual env    : {'YES' if in_venv else 'NO'}")
    print(f"pip            : {tool_version([sys.executable, '-m', 'pip', '--version'])}")
    print(f"git            : {tool_version(['git', '--version'])}")
    print("\n--- Packages ---")
    for import_name, pip_name in PACKAGES:
        print(f"{pip_name:<15}: {check_package(import_name, pip_name)}")


if __name__ == "__main__":
    main()