"""Полный воспроизводимый запуск IMOEX + панель акций.

Из корня проекта: python financeML/run.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

for script in (
    "download.py",
    "vol_pilot.py",
    "checks_part1.py",
    "panel.py",
    "research_addons.py",
    "calendar_panel.py",
    "publication_figures.py",
):
    subprocess.run([sys.executable, str(ROOT / script)], check=True)
