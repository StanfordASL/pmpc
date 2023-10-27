import sys
from pathlib import Path

project_root = Path(__file__).absolute().parents[1]
sys.argv = [str(project_root / "setup.py"), "install"]

if __name__ == "__main__":
    sys.path.insert(0, str(project_root))

    from setup import install_dynamic, install_static

    install_static()
