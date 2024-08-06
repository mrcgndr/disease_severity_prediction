import os
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.profilers as pl_profilers

from ..datasets import norm
from ..datasets.augmentation import ImageAugmentationPipeline
from ..models.backbones import VGG
from ..models.multi_dldl import MultiDLDL
from ..types.callbacks import CopyConfigFile
from ..types.trainer import ModelTrainer


class MultiDLDLTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, chkpt: Optional[Union[Path, str]] = None) -> None:
        if isinstance(self.config["data"]["paths"], list):
            h5paths = sorted(self.config["data"]["paths"])
        else:
            h5glob = Path(self.config["data"]["paths"])
            h5paths = sorted(Path("/").glob(str(h5glob.relative_to("/"))))

        self.data_config = {
            "h5paths": h5paths,
            "labels": self.config["data"]["labels"],
            "image_channels": self.config["data"]["image_channels"],
            "balanced_split_label": self.config["data"]["balanced_split_label"],
            "balanced_train_label": self.config["data"]["balanced_train_label"],
            "val_size": self.config["data"]["val_size"],
            "num_workers": self.config["training"]["n_workers"],
            "pin_memory": self.config["training"]["pin_memory"],
            "persistent_workers": self.config["training"]["persistent_workers"],
            "distributed": len(self.config["training"]["gpus"]) > 1,
        }
        self.data_config["batch_size"] = self.config["training"]["batch_size_per_gpu"]
        if "paths_test" in self.config["data"].keys():
            if isinstance(self.config["data"]["paths_test"], list):
                h5paths_test = sorted(self.config["data"]["paths_test"])
            else:
                h5glob_test = Path(self.config["data"]["paths_test"])
                h5paths_test = sorted(Path("/").glob(str(h5glob_test.relative_to("/"))))
            self.data_config["test_h5paths"] = h5paths_test

        if not self.dev_run:
            from aim.pytorch_lightning import AimLogger

            aim_logger = AimLogger(repo=self.config["training"]["log_dir"], experiment=self.config["model"]["name"])

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                root_dir = str(
                    Path(self.config["training"]["chkpt_dir"])
                    / f"{self.config['model']['name']}_{self.config['model']['normalizer']['module']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                )
                os.environ["ROOT_DIR"] = root_dir
            else:
                root_dir = os.environ["ROOT_DIR"]

            callbacks = []
            callbacks.append(CopyConfigFile(config=self.config, dirpath=root_dir))
            callbacks.append(pl_callbacks.LearningRateMonitor(logging_interval="step"))
            for callback in self.config["training"]["callbacks"]:
                if callback["module"] == "ModelCheckpoint":
                    callbacks.append(pl_callbacks.ModelCheckpoint(dirpath=root_dir, **callback["args"]))
                else:
                    callbacks.append(getattr(pl_callbacks, callback["module"])(**callback["args"]))

            if self.config["training"]["profiler"] is not None:
                profiler = getattr(pl_profilers, self.config["training"]["profiler"]["module"])(
                    dirpath=root_dir, **self.config["training"]["profiler"]["args"]
                )
            else:
                profiler = None

        else:
            callbacks = None
            profiler = None
            root_dir = None

        if self.config["training"]["seed"] is not None:
            pl.seed_everything(self.config["training"]["seed"], workers=True)

        model = MultiDLDL(self.config["model"])

        normalizer = getattr(norm, self.config["model"]["normalizer"]["module"])(**self.config["model"]["normalizer"]["args"])
        if self.config["data"]["augmentations"] is not None:
            augmentations = ImageAugmentationPipeline(self.config["data"]["augmentations"])
        else:
            augmentations = None
        dataset_module = import_module(f"disease_pred.datasets.{self.config['data']['module'].split('.')[0]}")
        datamodule = getattr(dataset_module, self.config["data"]["module"].split(".")[1])(
            **self.data_config, normalizer=normalizer, augmentations=augmentations
        )

        trainer = pl.Trainer(
            logger=aim_logger if not self.dev_run else False,
            accelerator="auto",
            strategy="ddp_find_unused_parameters_true" if len(self.config["training"]["gpus"]) > 1 else "auto",
            use_distributed_sampler=False,
            devices=self.config["training"]["gpus"] if len(self.config["training"]["gpus"]) > 1 else "auto",
            deterministic=(
                (self.config["training"]["seed"] is not None)
                if (not isinstance(model.backbone, VGG) and not isinstance(normalizer, norm.HistogramEqualizer))
                else False
            ),
            callbacks=callbacks,
            profiler=profiler,
            fast_dev_run=False,  # self.dev_run,
            default_root_dir=root_dir,
            **self.config["training"]["trainer_args"],
        )

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=chkpt)

        if not self.dev_run:
            aim_logger.finalize(status="success")
