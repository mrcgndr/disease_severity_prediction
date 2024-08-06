from typing import Dict

import albumentations as al


class ImageAugmentationPipeline:

    pipeline = {"train": None, "validate": None, "test": None}

    def __init__(self, config: Dict) -> None:
        for stage in config.keys():
            if stage not in self.pipeline.keys():
                raise ValueError(f"stage {stage} not known. Choose one of {self.pipeline.keys()}.")
            else:
                if config.get(stage):
                    self.pipeline[stage] = al.from_dict({"transform": config.get(stage)})
                else:
                    self.pipeline[stage] = None

    def __getitem__(self, stage: str) -> al.BaseCompose:
        return self.pipeline[stage]
