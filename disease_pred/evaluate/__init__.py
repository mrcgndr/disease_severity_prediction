import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.colors import TwoSlopeNorm
from rasterio.transform import xy
from rasterio.windows import get_data_window
from scipy.interpolate import griddata

from ..types.errors import NoImageDataAvaliableError, ValueNotUnderstoodError
from ..visualize import make_ds_scale_cmap, rescale_rgb


@dataclass
class Prediction:
    """Dataclass for label prediction output."""

    dists: np.ndarray  # [n_tiles, n_steps]
    vals: np.array  # [n_tiles]
    confs: np.array  # [n_tiles]
    stds: np.array  # [n_tiles]
    reg_limits: Tuple[float, float]
    true_std: float
    dist_steps: np.array
    true_vals: Optional[np.array] = None  # [n_tiles]


@dataclass
class PredictionDict(dict):
    def __init__(self, dict_: Dict[str, Prediction]):
        for k, v in dict_.items():
            if (not isinstance(k, str)) or (not (isinstance(v, Prediction))):
                raise ValueError("Wrong type of prediction dict. Keys have to be of type 'str' and values of type 'Prediction'")
        super().__init__(dict_)

    def __repr__(self):
        info = "Prediction dict\n" f"labels: {list(self.keys())}"
        return info

    def __len__(self):
        return len(self.keys())

    def plot_vals_hist(
        self,
        bins: Union[int, np.array],
        fig_params: Optional[Dict] = {},
        plot_params: Optional[Dict] = {},
        ax_params: Optional[Dict] = {},
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots histogram with evaluation values.

        Args:
            bins (Union[int, np.array]): Number of bins.
            fig_params (Optional[Dict], optional): Further figure params. Defaults to None.
            plot_params (Optional[Dict], optional): Further plot params. Defaults to None.
            ax_params (Optional[Dict], optional): Further axis params. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
        """

        n_labels = len(self)
        fig, ax = plt.subplots(n_labels, 1, **fig_params)
        ax = np.atleast_1d(ax)
        for a, (labelname, pred) in zip(ax, self.items()):
            a.hist(pred.vals, bins=bins, range=pred.reg_limits, **plot_params)
            a.set(title=labelname, ylabel="counts", **ax_params)
            a.grid(True)

        return fig, ax

    def plot_stdhist(
        self,
        bins: Union[int, np.array],
        fig_kwargs: Optional[Dict] = {},
        ax_kwargs: Optional[Dict] = {},
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot histogram for predicted standard deviations.

        Args:
            bins (Union[int, np.array]): Number of bins.
            fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to {}.
            ax_kwargs (Optional[Dict], optional): Further axis params. Defaults to {}.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
        """
        fig, ax = plt.subplots(len(self), 1, figsize=(8, 3 * len(self)), tight_layout=True, **fig_kwargs)
        fig.suptitle("distributions of predicted standard deviations")
        if len(self) == 1:
            ax = [ax]
        for a, (labelname, pred) in zip(ax, self.items()):
            a.set(title=labelname, xlabel="std", ylabel="density", **ax_kwargs)
            a.hist(pred.stds, bins=bins, density=True)
            a.axvline(pred.true_std, c="r")
            a.grid(True)

        return fig, ax


@dataclass
class OrthoEvaluatorOutput:
    """Dataclass for orthophoto evaluation output."""

    scale_factor: float
    gps_transform: rio.Affine
    tile_size: int
    points: np.ndarray  # [n_tiles, 2]
    preds: PredictionDict
    image: Optional[np.ndarray] = None  # [C, H, W]

    def __repr__(self) -> str:
        info = (
            "EvaluatorOutput\n"
            "---------------\n"
            f"source image: {self.image.shape if self.image else 'not available'}\n"
            f"tile size: {self.tile_size}\n"
            f"predicted points: {len(self.points)}"
        )
        return info

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "OrthoEvaluatorOutput":
        """Loads existing OrthoEvaluatorOutput instance from disk as pickle file.

        Args:
            path (Union[str, Path]): Path to pickle file.

        Raises:
            TypeError: If file does not contain a valid OrthoEvaluatorOutput instance.
        """
        with open(path, "rb") as pfile:
            output = pickle.load(pfile)
        if isinstance(output, cls):
            return output
        else:
            raise TypeError("File does not contain a valid OrthoEvaluatorOutput type.")

    def to_pickle(self, path: Union[str, Path]) -> None:
        """Saves instance to disk as pickle file.

        Args:
            path (Union[str, Path]): Path to pickle file.
        """
        with open(path, "wb") as pfile:
            pickle.dump(self, pfile)

    def get_lonlat_points(self) -> np.array:
        """Transform pixel coordinates to lonlat coordinates.

        Returns:
            np.array: Lonlat coodinates.
        """
        return np.asarray(xy(self.gps_transform, rows=self.points[:, 0], cols=self.points[:, 1])).T

    def get_geopandas(self) -> gpd.GeoDataFrame:
        """Export to GeoPandas Dataframe.

        Returns:
            gpd.GeoDataFrame: GeoPandas Dataframe.
        """
        df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*self.get_lonlat_points().T))
        for labelname, pred in self.preds.items():
            df[f"{labelname}_prediction"] = pred.vals
            df[f"{labelname}_confidence"] = pred.confs
        return df

    def export_raster(self, filename: str, interp: str = "cubic") -> None:
        """Export to rastered image.

        Args:
            filename (str): Path of exported image.
            interp (str, optional): Interpolation method. Defaults to "cubic".
        """
        if self.image is None:
            raise NoImageDataAvaliableError()

        filename = Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)

        val_map = np.zeros((len(self.preds), *self.image.shape[1:])) * np.nan
        points = np.rint(self.points).astype(int)
        for i, (x, y) in enumerate(points):
            for j, p in enumerate(self.preds.values()):
                val_map[j, x, y] = p.vals[i]

        x = np.arange(0, val_map.shape[2])
        y = np.arange(0, val_map.shape[1])

        xx, yy = np.meshgrid(x, y)

        for i in range(len(val_map)):
            x1 = xx[np.isfinite(val_map[i])]
            y1 = yy[np.isfinite(val_map[i])]
            newarr = val_map[i][np.isfinite(val_map[i])]

            val_map[i] = griddata((x1, y1), newarr.ravel(), (xx, yy), method=interp)

        window = get_data_window(np.ma.masked_invalid(val_map))

        out_meta = {
            "dtype": "float32",
            "count": val_map.shape[0],
            "height": window.height,
            "width": window.width,
            "nodata": np.nan,
            "transform": rio.windows.transform(window, self.gps_transform),
        }

        with rio.open(filename, "w", **out_meta) as out:
            out.write(val_map)
            out.set_band_description = list(self.preds.keys())

    def plot(
        self,
        show_image: bool = False,
        fig_kwargs: Optional[Dict] = {},
        plot_kwargs: Optional[Dict] = {},
        ax_kwargs: Optional[Dict] = {},
        coords: Optional[str] = "xy",
        custom_labels: Optional[List[str]] = None,
        conf_plot: Optional[bool] = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot image with evalutations.

        Args:
            show_image (bool, optional): If image is shown. Defaults to False.
            fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to None.
            plot_kwargs (Optional[Dict], optional): Further plot params. Defaults to None.
            ax_kwargs (Optional[Dict], optional): Further axis params. Defaults to None.
            coords (Optional[str], optional): Coordinate system to plot the plant positions.
                Choose "lonlat" for GPS coordinates or "xy". Defaults to "xy".
            custom_labels (Optional[List[str]], optional): Custom labels for evaluation colorbar.
                Defaults to None.
            conf_plot (Optional[bool], optional): Whether to plot the confidences in a second axis.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
        """
        all_coords = ["lonlat", "xy"]
        if coords not in all_coords:
            raise ValueNotUnderstoodError("coords", coords, all_coords)
        ds_cmap = make_ds_scale_cmap()
        if coords == "lonlat":
            points = self.get_lonlat_points()
        else:
            points = self.points
        if custom_labels is None:
            custom_labels = [key for key in self.preds.keys()]

        if show_image:
            if self.image is None:
                raise NoImageDataAvaliableError()
            rgb = rescale_rgb(np.transpose(self.image[[2, 1, 0]], axes=(2, 1, 0)))
        n_labels = len(self.preds)
        fig, ax = plt.subplots(n_labels, int(conf_plot) + 1, **fig_kwargs)
        ax = np.atleast_2d(ax)
        for i in range(n_labels):
            val_plot = ax[i, 0].scatter(
                *points[:, [0, 1]].T, c=list(self.preds.values())[i].vals, cmap=ds_cmap, vmin=-0.5, vmax=9.5, **plot_kwargs
            )
            if show_image:
                ax[i, 0].imshow(rgb, alpha=0.7)
            # else:
            #     ax[i, 0].invert_yaxis()
            ax[i, 0].set(title=f"{custom_labels[i]} values", aspect="equal", **ax_kwargs)
            val_cbar = fig.colorbar(val_plot, ax=ax[0, i], extend="both")
            val_cbar.set_label(label=custom_labels[i])

            if conf_plot:
                conf_vals = list(self.preds.values())[i].confs
                conf_plot = ax[i, 1].scatter(
                    *points[:, [0, 1]].T,
                    c=conf_vals,
                    cmap="coolwarm_r",
                    norm=TwoSlopeNorm(vcenter=1),
                    **plot_kwargs,
                )
                if show_image:
                    ax[i, 1].imshow(rgb, alpha=0.7)
                # else:
                #     ax[i, 1].invert_yaxis()
                ax[i, 1].set(title=f"{custom_labels[i]} confidences", aspect="equal", **ax_kwargs)
                conf_cbar = fig.colorbar(conf_plot, ax=ax[i, 1])
                conf_cbar.ax.set_yscale("linear")
                conf_cbar.set_label(label=r"$\dfrac{\sigma_{true}}{\sigma_{pred}}$")

        return fig, ax
