import time
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
SCRIPTS_DIR = ROOT_DIR / "scripts"

_this_year = time.strftime("%Y")
__version__ = "0.1.0"
__author__ = "Maurice Günder"
__author_email__ = "mguender@uni-bonn.de"
__license__ = ""
__copyright__ = f"Copyright (c) 2024-{_this_year}, {__author__}."
__homepage__ = ""
__docs__ = "disease severity prediction"
