from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union


class ModelTrainer(ABC):
    def __init__(self, config: Dict, dev_run: bool = False):
        self.config = config
        self.dev_run = dev_run

    @abstractmethod
    def run(self, chkpt: Optional[Union[Path, str]] = None):
        pass
