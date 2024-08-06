from abc import ABC, abstractmethod

# from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from pytorch_lightning import LightningModule
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window

# from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

from ..models.multi_dldl import MultiDLDL
from ..utils import calc_m_per_px
from . import OrthoEvaluatorOutput, Prediction, PredictionDict


class Evaluator(ABC):
    """Abstract class for model evaluation."""

    def __init__(self, model_type: Type[LightningModule], checkpoint_path: str, device: torch.device, **kwargs):
        self.model = model_type.load_from_checkpoint(checkpoint_path, map_location=device, **kwargs)
        self.model.to(device)
        self.model.eval()
        self.model_device = device

    @abstractmethod
    def _evaluate(self, *args, **kwargs) -> Any:
        pass


class MultiDLDLEvaluator(Evaluator):
    """MultiDLDL model evaluator."""

    def __init__(self, checkpoint_path: str, device: torch.device, **kwargs):
        super().__init__(MultiDLDL, checkpoint_path, device, **kwargs)

    @torch.no_grad()
    def _evaluate_rotated(self, image_batch: torch.Tensor, flip: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        pred_dist, pred_exp_norm = [], []
        for label_idx, n in enumerate(self.model.n_steps):
            pred_dist_ = torch.zeros((4 * (1 + flip), len(image_batch), n))
            pred_exp_norm_ = torch.zeros((4 * (1 + flip), len(image_batch)))

            # 4 evaluations by 90 degree rotations
            for i in range(4):
                output = self.model(torch.rot90(image_batch, k=i, dims=(-2, -1)), label_idx)
                pred_dist_[i], pred_exp_norm_[i] = output["label_dist"], output["exp_norm"]

            # flip image and evaluate the 4 rotations again
            if flip:
                flipped_image_batch = torch.flip(image_batch, dims=[-1])
                for i in range(4, 8):
                    output = self.model(torch.rot90(flipped_image_batch, k=i, dims=(-2, -1)), label_idx)
                    pred_dist_[i], pred_exp_norm_[i] = output["label_dist"], output["exp_norm"]

            # average the 4 (8) evaluations for each image
            pred_dist.append(pred_dist_.mean(dim=0))
            pred_exp_norm.append(pred_exp_norm_.mean(dim=0))

        return {"label_dist": pred_dist, "exp_norm": pred_exp_norm}

    @torch.no_grad()
    def _evaluate(self, image_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_dist, pred_exp_norm = [], []
        for label_idx in range(len(self.model.n_steps)):
            output = self.model(image_batch, label_idx)

            pred_dist.append(output["label_dist"])
            pred_exp_norm.append(output["exp_norm"])

        return {"label_dist": pred_dist, "exp_norm": pred_exp_norm}

    @torch.no_grad()
    def get_prepared_targets(self, data_targets: Dict[str, Any], idx: Optional[int] = 0) -> Dict[str, Optional[torch.Tensor]]:
        return self.model.prepare_target(data_targets, idx, self.model_device)


class MultiDLDLDataLoaderEvaluator(MultiDLDLEvaluator):
    """MultiDLDL model evaluator for dataloaders."""

    def __init__(self, checkpoint_path: str, device: torch.device, **kwargs):
        super().__init__(checkpoint_path, device, **kwargs)

    @torch.no_grad()
    def evaluate(
        self, dataloader: torch.utils.data.DataLoader, rotate: Optional[bool] = False, flip: Optional[bool] = False
    ) -> PredictionDict:
        """Evaluate model for a DataLoader.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader object to evaluate.
            rotate (optional, bool): If True, the image batch is evaluated
                by repeated evaluation for 4 rotated image orientations. Defaults to False.
            flip (optional, bool): If True, the image batch is additionally flippd and evaluated for the 4 orientations.
                Only relevant if rotated is True. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with model predictions.
        """

        pred_dist_, pred_exp_norm_, true_ = [], [], []

        for images, targets in tqdm(dataloader, desc=f"Evaluate {dataloader.batch_size} tiles per batch"):
            images = images.to(self.model_device)

            if rotate:
                ev_out = self._evaluate_rotated(images, flip=flip)
            else:
                ev_out = self._evaluate(images)
            pred_dist_.append(ev_out["label_dist"])
            pred_exp_norm_.append(ev_out["exp_norm"])
            true_.append([labels for labels in targets["labels"]])

        preds = []
        for i in range(len(self.model.labels)):
            output = {
                "label_dist": torch.vstack([p[i] for p in pred_dist_]),
                "exp_norm": torch.hstack([p[i] for p in pred_exp_norm_]),
            }
            conf = self.model.calculate_confidence(output, i)
            preds.append(
                Prediction(
                    dists=output["label_dist"].cpu().numpy(),
                    vals=self.model.scale_from_norm(output["exp_norm"], i).cpu().numpy(),
                    true_vals=torch.hstack([t[i] for t in true_]).cpu().numpy(),
                    confs=conf["confidence"].cpu().numpy(),
                    stds=conf["sigma_pred"].cpu().numpy(),
                    reg_limits=self.model.reg_limits[i],
                    true_std=self.model.sigmas[i],
                    dist_steps=self.model.steps[i],
                )
            )

        return PredictionDict({label: preds[i] for i, label in enumerate(self.model.labels)})


class MultiDLDLOrthoEvaluator(MultiDLDLEvaluator):
    """MultiDLDL model evaluator for orthoimages."""

    def __init__(self, checkpoint_path: str, device: torch.device, **kwargs):
        super().__init__(checkpoint_path, device, **kwargs)

    def _read_ortho(self, image_path: str, src_channels: List[str], scaled: bool = True) -> Dict[str, Any]:
        read_channels = [np.argmax(np.asarray(src_channels) == c) + 1 for c in self.model.image_channels]
        with rio.open(image_path, "r") as raster:
            if scaled:
                raster_scale = calc_m_per_px(raster.meta) * 1e2  # cm/px
                scale_factor = raster_scale / self.model.image_resolution

            image = raster.read(
                read_channels,
                out_shape=(
                    (len(read_channels), int(raster.height * scale_factor), int(raster.width * scale_factor))
                    if scaled
                    else None
                ),
                resampling=Resampling.bilinear,
                fill_value=np.nan,
            )

            meta = {"scale_factor": scale_factor if scaled else 1.0, "transform": raster.transform}

        return {"image": image, "meta": meta}

    def _read_image_batch_from_ortho(
        self,
        image_path: str,
        src_channels: List[str],
        points: np.ndarray,
        coords_type: str = "xy",
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        image_batch = []
        read_channels = [np.argmax(np.asarray(src_channels) == c) + 1 for c in self.model.image_channels]
        with rio.open(image_path, "r") as raster:
            raster_scale = calc_m_per_px(raster.meta) * 1e2  # cm/px
            scale_factor = raster_scale / self.model.image_resolution

            if coords_type == "lonlat":
                points = np.asarray(rowcol(raster.transform, xs=points[:, 0], ys=points[:, 1], op=lambda x: x)).T
            elif coords_type == "xy":
                pass
            else:
                raise ValueError("coords_type argument not unterstood. Chosse 'xy' or 'lonlat'.")

            valid_points = []
            for p in tqdm(points, desc="Extract tiles from orthoimage", leave=False, position=0):
                w = Window(
                    p[1] - self.model.image_size / (2 * scale_factor),
                    p[0] - self.model.image_size / (2 * scale_factor),
                    self.model.image_size / scale_factor,
                    self.model.image_size / scale_factor,
                )

                try:
                    tile = raster.read(
                        read_channels,
                        out_shape=(len(read_channels), self.model.image_size, self.model.image_size),
                        resampling=Resampling.cubic,
                        window=w,
                        fill_value=np.nan,
                    )
                    image_batch.append(tile)
                    valid_points.append(p)
                except Exception:
                    pass

        meta = {"scale_factor": scale_factor, "transform": raster.transform, "points_xy": np.asarray(valid_points)}

        return {"batch": torch.as_tensor(np.asarray(image_batch), device=device), "meta": meta}

    def _get_grid_points(
        self, image_shape: Tuple[int, int], step: Tuple[int, int], output_shape: Tuple[int, int]
    ) -> Tuple[List, np.ndarray]:
        points = []

        hsteps = np.arange(0, image_shape[0], step[0])
        wsteps = np.arange(0, image_shape[1], step[1])

        for hs in hsteps:
            for ws in wsteps:
                points.append((hs + output_shape[0] // 2, ws + output_shape[1] // 2))

        return np.asarray(points)

    @torch.no_grad()
    def evaluate(
        self, images: torch.Tensor, batch_size: int, rotate: Optional[bool] = False, flip: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:
        """Evaluate model for an image batch.

        Args:
            image_batch (torch.Tensor): Image batch.
            rotate (Optional[bool], optional): If True, the image batch is evaluated
                by repeated evaluation for 4 rotated image orientations. Defaults to False.
            flip (Optional[bool], optional): If True, the image batch is additionally flippd and evaluated for the 4 orientations.
                Only relevant if rotated is True. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with model predictions.
        """
        pred_dist_, pred_exp_norm_ = [], []
        batches = torch.split(images, batch_size)
        for batch in tqdm(batches, desc=f"Evaluate {batch_size} tiles per batch", leave=False):
            batch = batch.to(self.model_device)
            if rotate:
                ev_out = self._evaluate_rotated(batch, flip=flip)
            else:
                ev_out = self._evaluate(batch)
            pred_dist_.append(ev_out["label_dist"])
            pred_exp_norm_.append(ev_out["exp_norm"])

        dists, vals = [], []
        for i in range(len(self.model.labels)):
            dists.append(torch.vstack([p[i] for p in pred_dist_]))
            vals.append(torch.hstack([p[i] for p in pred_exp_norm_]))

        return {"label_dist": dists, "exp_norm": vals}

    def evaluate_grid(
        self,
        image_path: str,
        src_channels: List[str],
        step: Tuple[int, int],
        batch_size: int,
        rotate: Optional[bool] = False,
        flip: Optional[bool] = False,
        save_raw_image: Optional[bool] = False,
    ) -> OrthoEvaluatorOutput:
        """Full grid-wise evaluation of an orthoimage.

        Args:
            image_path (str): Path to orthoimage.
            src_channels (List[str]): Channel names of source.
            step (Tuple[int, int]): (x, y) evaluation steps/strides in pixels of the original image.
            batch_size (int): Batch size for evaluation.
            rotate (Optional[bool], optional): If True, the image batch is evaluated
                by repeated evaluation for 4 rotated image orientations. Defaults to False.
            flip (Optional[bool], optional): If True, the image batch is additionally flippd and evaluated for the 4 orientations.
                Only relevant if rotated is True. Defaults to False.
            save_raw_image (Optional[bool], optional) = If True, the raw image is saved to the `ImageEvaluatorOutput` object resulting
                in large object sizes. Defaults to False.

        Returns:
            ImageEvaluatorOutput: Evaluated image with metadata and label predictions.
        """

        with rio.open(image_path, "r") as raster:
            image_shape = raster.shape

        points = self._get_grid_points(
            image_shape=image_shape, step=step, output_shape=(self.model.image_size, self.model.image_size)
        )

        points_batches = np.array_split(points, np.arange(batch_size, len(points), batch_size))

        pred_dist_, pred_exp_norm_, filtered_points = [], [], []

        for points_batch in tqdm(
            points_batches, desc=f"Evaluate {len(points_batches[0])} tiles per batch", total=len(points_batches)
        ):
            batch = self._read_image_batch_from_ortho(image_path, src_channels, points_batch, "xy")
            points_batch = batch["meta"]["points_xy"]

            nanmask = torch.sum(torch.isfinite(batch["batch"]), dim=(1, 2, 3)) >= 0.75 * (
                len(self.model.image_channels) * self.model.image_size**2
            )
            batch["batch"] = batch["batch"][nanmask]
            points_batch = points_batch[nanmask]

            if torch.sum(nanmask) > 0:
                ev_out = self.evaluate(batch["batch"], batch_size=len(batch["batch"], rotate=rotate, flip=flip))
                pred_dist_.append(ev_out["label_dist"])
                pred_exp_norm_.append(ev_out["exp_norm"])
                filtered_points.extend(points_batch)

        preds = []

        for i in range(len(self.model.n_steps)):
            output = {
                "label_dist": torch.vstack([p[i] for p in pred_dist_]),
                "exp_norm": torch.hstack([p[i] for p in pred_exp_norm_]),
            }
            conf = self.model.calculate_confidence(output, i)
            preds.append(
                Prediction(
                    dists=output["label_dist"].cpu().numpy(),
                    vals=self.model.scale_from_norm(output["exp_norm"], i).cpu().numpy(),
                    confs=conf["confidence"].cpu().numpy(),
                    stds=conf["sigma_pred"].cpu().numpy(),
                    reg_limits=self.model.reg_limits[i],
                    true_std=self.model.sigmas[i],
                    dist_steps=self.model.steps[i],
                )
            )

        pred_dict = PredictionDict({label: preds[i] for i, label in enumerate(self.model.labels)})

        return OrthoEvaluatorOutput(
            image=self._read_ortho(image_path, src_channels, scaled=False)["image"] if save_raw_image else None,
            scale_factor=batch["meta"]["scale_factor"],
            gps_transform=batch["meta"]["transform"],
            tile_size=self.model.image_size,
            points=np.asarray(filtered_points),
            preds=pred_dict,
        )

    def evaluate_plants_dataframe(
        self,
        plants_df_path: str,
        image_dir: str,
        src_channels: List[str],
        batch_size: int,
        rotate: Optional[bool] = True,
        flip: Optional[bool] = True,
    ) -> pd.DataFrame:
        """Evaluation of plants DataFrame from Cataloging Workflow with corresponding orthoimages.

        Args:
            plants_df_path (str): Path to plants DataFrame .pkl file.
            image_dir (str): Path to directory with orthoimages.
            src_channels (List[str]): Channel names of source.
            batch_size (int): Batch size for evaluation.
            rotate (Optional[bool], optional): If True, the image batch is evaluated
                by repeated evaluation for 4 rotated image orientations. Defaults to True.
            flip (Optional[bool], optional): If True, the image batch is additionally flippd and evaluated for the 4 orientations.
                Only relevant if rotated is True. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with model evaluations.
        """
        image_dir = Path(image_dir)
        plants_df = pd.read_pickle(Path(plants_df_path)).sort_values(["field_id", "date"])
        for label in self.model.labels:
            plants_df[f"{label}_dist"] = None
            plants_df[f"{label}_val"] = None

        dates = pd.unique(plants_df["date"])
        dates_str = np.datetime_as_string(dates, unit="D")
        field_ids = pd.unique(plants_df["field_id"])

        for f_id in tqdm(field_ids, "Field IDs"):
            for d, d_str in tqdm(zip(dates, dates_str), "Dates", leave=False, total=len(dates)):
                p = plants_df[(plants_df.date == d) & (plants_df.field_id == f_id)]

                image_path = next((image_dir / (f_id + "_" + d_str.replace("-", ""))).glob("*.tif"))

                centroids = np.stack(p.xy_px)

                tiles = self._read_image_batch_from_ortho(image_path, src_channels, centroids)

                nanmask = torch.sum(torch.isfinite(tiles["batch"]), dim=(1, 2, 3)) >= 0.75 * (
                    len(self.model.image_channels) * self.model.image_size**2
                )
                nanmask = nanmask.cpu().numpy()
                tiles["batch"] = tiles["batch"][nanmask]
                # print(f"Tiles with <75% data filtered ({np.sum(nanmask)/len(nanmask)*100:.2f}% remaining) ...")

                ev_out = self.evaluate(tiles["batch"], batch_size=batch_size, rotate=rotate, flip=flip)

                for i, (df_idx, row) in enumerate(p[nanmask].iterrows()):
                    for j, label in enumerate(self.model.labels):
                        plants_df.at[df_idx, f"{label}_dist"] = ev_out["label_dist"][j][i].cpu().numpy()
                        plants_df.at[df_idx, f"{label}_val"] = self.model.scale_from_norm(
                            ev_out["exp_norm"][j][i].cpu().numpy(), j
                        )

        return plants_df

    def evaluate_geo(
        self,
        geo_path: str,
        image_path: str,
        src_channels: List[str],
        batch_size: int,
        rotate: Optional[bool] = False,
        flip: Optional[bool] = False,
        out_dir: Optional[str] = None,
        save_raw_image: Optional[bool] = False,
    ) -> Optional[OrthoEvaluatorOutput]:
        """Evaluation of an orthoimage on positions given in a GeoPackage .gpkg or GeoJSON .json file.

        Args:
            geo_path (str): Path to GeoPackage .gpkg / GeoJSON .json file.
            image_path (str): Path to orthoimage.
            src_channels (List[str]): Channel names of source.
            batch_size (int): Batch size for evaluation.
            rotate (Optional[bool], optional): If True, the image batch is evaluated
                by repeated evaluation for 4 rotated image orientations. Defaults to False.
            flip (Optional[bool], optional): If True, the image batch is additionally flippd and evaluated for the 4 orientations.
                Only relevant if rotated is True. Defaults to False.
            out_dir (Optional[str], optional): Optional. Path to directory where gpkg with predictions is saved.
                If None, no gpkg is saved. Defaults to None.
            save_raw_image (Optional[bool], optional) = If True, the raw image is saved to the `ImageEvaluatorOutput` object resulting
                in large object sizes. Defaults to False.

        Returns:
            Optional[OrthoEvaluatorOutput]: Evaluated image with metadata and label predictions.
        """
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)

        source_df = gpd.read_file(geo_path).explode(ignore_index=True)
        source_df = source_df[~source_df.is_empty]

        lonlats = np.asarray([[p.xy[0][0], p.xy[1][0]] for p in source_df.geometry])
        print(f"Read {len(lonlats)} points from geo file ...")
        tiles = self._read_image_batch_from_ortho(image_path, src_channels, lonlats, coords_type="lonlat")

        if len(tiles["batch"]) > 0:
            nanmask = torch.sum(torch.isfinite(tiles["batch"]), dim=(1, 2, 3)) >= 0.75 * (
                len(self.model.image_channels) * self.model.image_size**2
            )
            nanmask = nanmask.cpu().numpy()
            tiles["batch"] = tiles["batch"][nanmask]
            lonlats = lonlats[nanmask]
            xys = tiles["meta"]["points_xy"][nanmask]
            print(f"Tiles with <75% data filtered ({np.sum(nanmask)/len(nanmask)*100:.2f}% remaining) ...")

            if np.sum(nanmask) > 0:
                ev_out = self.evaluate(tiles["batch"], batch_size=batch_size, rotate=rotate, flip=flip)

                preds = []

                if out_dir is not None:
                    out_path = out_dir / (image_path.stem + ".json")
                    print(f"Save predictions to {out_path} ...")
                    for i, l in enumerate(self.model.labels):
                        output = {"label_dist": ev_out["label_dist"][i], "exp_norm": ev_out["exp_norm"][i]}
                        source_df[f"{l}_prediction"] = np.nan
                        source_df.loc[nanmask, f"{l}_prediction"] = (
                            self.model.scale_from_norm(output["exp_norm"], i).cpu().numpy()
                        )
                        source_df[f"{l}_confidence"] = np.nan
                        source_df.loc[nanmask, f"{l}_confidence"] = (
                            self.model.calculate_confidence(output, i)["confidence"].cpu().numpy()
                        )
                    try:
                        source_df.to_file(out_path)
                    except Exception as e:
                        print(f"Failed to save file. Error: {e}")

                for i in range(len(self.model.n_steps)):
                    output = {"label_dist": ev_out["label_dist"][i], "exp_norm": ev_out["exp_norm"][i]}
                    conf = self.model.calculate_confidence(output, i)
                    preds.append(
                        Prediction(
                            dists=output["label_dist"].cpu().numpy(),
                            vals=self.model.scale_from_norm(output["exp_norm"], i).cpu().numpy(),
                            confs=conf["confidence"].cpu().numpy(),
                            stds=conf["sigma_pred"].cpu().numpy(),
                            reg_limits=self.model.reg_limits[i],
                            true_std=self.model.sigmas[i],
                            dist_steps=self.model.steps[i],
                        )
                    )

                pred_dict = PredictionDict({label: preds[i] for i, label in enumerate(self.model.labels)})

                return OrthoEvaluatorOutput(
                    image=self._read_ortho(image_path, src_channels, scaled=False)["image"] if save_raw_image else None,
                    scale_factor=tiles["meta"]["scale_factor"],
                    gps_transform=tiles["meta"]["transform"],
                    tile_size=self.model.image_size,
                    points=xys,
                    preds=pred_dict,
                )
        else:
            print("No position in the given geo file was found in the orthoimage.")
            return None
