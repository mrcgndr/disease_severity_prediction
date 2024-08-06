import os
from typing import Dict, Optional

import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class CopyConfigFile(Callback):
    def __init__(self, config: Dict, dirpath: Optional[str] = None):
        self.config = config
        self.dirpath = dirpath
        super().__init__()

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.dirpath:
            dirpath = self.dirpath
        else:
            dirpath = trainer.default_root_dir
        if trainer.global_rank == 0:
            os.makedirs(dirpath, exist_ok=True)

            with open(os.path.join(dirpath, "config.yml"), "w") as config_file:
                yaml.dump(self.config, config_file, yaml.Dumper)

            print(f"Config file saved to {dirpath}")
