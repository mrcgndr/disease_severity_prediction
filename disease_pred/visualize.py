from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.exposure import rescale_intensity
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.backbones import PretrainedViT, ViT
from .models.multi_dldl import MultiDLDL
from .models.utils import attention_rollout, pick_target_subset


def hex_to_rgb(value):
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    return [v / 256 for v in value]


def rescale_rgb(img: np.ndarray) -> np.ndarray:
    return rescale_intensity(img, in_range=tuple(np.nanpercentile(np.nanmean(img, axis=2), (0.5, 99.5))))


def get_continuous_cmap(
    hex_list: List[str], float_list: Optional[List[float]] = None, name: Optional[str] = "custom_cmap"
) -> mcolors.LinearSegmentedColormap:
    """Creates and returns a color map that can be used in heat map figures.

    Args:
        hex_list (List[str]): List of hex code strings
        float_list (Optional[List[float]], optional): List of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1. Defaults to None.
        name (Optional[str], optional): Name of color map. Defaults to "custom_cmap".

    Returns:
        mcolors.LinearSegmentedColormap: Color map object
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmap = mcolors.LinearSegmentedColormap(name, segmentdata=cdict, N=256)
    return cmap


def make_ds_scale_cmap() -> mcolors.LinearSegmentedColormap:
    """Make matplotlib color map object for disease severity scale.

    Returns:
        mcolors.LinearSegmentedColormap: Disease severity color map.
    """
    cmap = get_continuous_cmap(hex_list=["#00A933", "#FFFF00", "#FF0000"], name="ds_scale")
    cmap.set_under("#2A6099")
    cmap.set_over("#7B3F00")
    # TODO:
    return cmap


def plot_input(
    model: MultiDLDL,
    dataloader: DataLoader,
    n_samples: int,
    vmin: float = -3.0,
    vmax: float = 3.0,
    cmap: str = "RdBu",
    fig_kwargs: Optional[Dict] = {},
    ax_kwargs: Optional[Dict] = {},
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a sample of input images.

    Args:
        model (MultiDLDL): MultiDLDL model.
        dataloader (DataLoader): Dataloader with images.
        n_samples (int): Number of samples to be plotted.
        fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to {}.
        ax_kwargs (Optional[Dict], optional): Further axis params. Defaults to {}.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
    """
    custom_figsize = fig_kwargs.pop("figsize", None)

    channels = model.image_channels
    fig = plt.figure(
        figsize=custom_figsize if custom_figsize is not None else (n_samples * len(model.labels), 15),
        tight_layout=True,
        **fig_kwargs,
    )

    images, labels = next(iter(dataloader))

    choice_imgs = torch.stack(
        [images[i] for i in np.random.choice(np.arange(len(images)), size=n_samples, replace=False)], axis=0
    )
    choice_imgs = model.preprocessor(choice_imgs)

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(len(channels), n_samples),
        axes_pad=0.1,  # pad between axes in inch.
        cbar_mode="single",
        share_all=True,
    )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for i, ax in enumerate(grid):
        row, col = i // n_samples, i % n_samples
        im = ax.imshow(choice_imgs[col, row], vmin=vmin, vmax=vmax, cmap=cmap)
        # if row == 0:
        #    ax.set(title = l[col])
        if col == 0:
            ax.set(ylabel=f"{channels[row%len(channels)]}")
        if ax_kwargs:
            ax.set(**ax_kwargs)
    grid.cbar_axes[0].colorbar(im, extend="both")

    return fig, ax


def plot_examples(
    model: MultiDLDL,
    dataloader: DataLoader,
    n_samples: int,
    fig_kwargs: Optional[Dict] = {},
    transparency: Optional[bool] = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot model output by some samples.

    Args:
        model (MultiDLDL): MultiDLDL model.
        dataloader (DataLoader): Dataloader with images.
        n_samples (int): Number of samples to be plotted.
        fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to {}.
        transparency (Optional[bool], optional): Use transparency in plotting.
            Defaults to True.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
    """
    custom_figsize = fig_kwargs.pop("figsize", None)

    fig, ax = plt.subplots(
        n_samples,
        len(model.labels) + 1,
        figsize=custom_figsize if custom_figsize is not None else (3 + 5 * len(model.labels), 3 * n_samples),
        tight_layout=True,
        **fig_kwargs,
    )
    if n_samples == 1:
        ax = np.array([ax])
    model.eval()

    rgb_channels = [np.argmax(np.asarray(model.image_channels) == c) for c in ["R", "G", "B"]]

    images, labels = next(iter(dataloader))

    choice = np.random.choice(np.arange(len(images)), size=n_samples, replace=False)
    images = images[choice].to(model.device)
    labels = pick_target_subset(labels, choice)

    for i, obj_name in enumerate(model.labels):
        target = model.prepare_target(labels, i, device=torch.device("cpu"))

        for j, a in enumerate(ax[:, 0]):
            rgb = np.transpose(images[j].cpu().numpy()[rgb_channels, :, :], axes=(1, 2, 0))
            rgb = rescale_rgb(rgb)
            a.imshow(rgb)
        ax[0, 0].set(title="image (RGB repr.)")

        with torch.no_grad():
            output = model(images, i)
            output = {k: v.cpu() for k, v in output.items()}

            pred_std = torch.sqrt(
                torch.sum(
                    output["label_dist"] * torch.pow(model.norm_steps[i] - output["exp_norm"][:, None], 2),
                    dim=1,
                )
            )
            pred_std = pred_std * (model.reg_limits[i][1] - model.reg_limits[i][0])
            pred_exp = model.scale_from_norm(output["exp_norm"], i)

            true_std = (
                target["stds"] * (model.reg_limits[i][1] - model.reg_limits[i][0])
                if target["stds"]
                else torch.ones_like(target["exp_norm"]) * model.sigmas[i]
            )
            true_exp = model.scale_from_norm(target["exp_norm"], i)

            loss = model.calculate_loss(
                output,
                target,
                idx=i,
                reduction="none",
            )

            for j, a in enumerate(ax[:, i + 1]):
                a.plot(model.steps[i].cpu(), target["label_dist"][j], color="C2", label="true")
                a.plot(model.steps[i].cpu(), output["label_dist"][j], color="C0", label="pred")
                a.axvline(true_exp[j], c="C2", ls="--", label="true mean")
                a.axvline(pred_exp[j], c="C0", ls="--", label="pred mean")
                if transparency:
                    a.axvspan(true_exp[j] - true_std[j], true_exp[j] + true_std[j], color="C2", alpha=0.2, label="true std")
                    a.axvspan(pred_exp[j] - pred_std[j], pred_exp[j] + pred_std[j], color="C0", alpha=0.2, label="pred std")
                else:
                    a.axvline(true_exp[j] - true_std[j], c="C2", ls="--", linewidth=0.6, label="true std")
                    a.axvline(true_exp[j] + true_std[j], c="C2", ls="--", linewidth=0.6)
                    a.axvline(pred_exp[j] - pred_std[j], c="C0", ls="--", linewidth=0.6, label="pred std")
                    a.axvline(pred_exp[j] + pred_std[j], c="C0", ls="--", linewidth=0.6)
                # a.axvline(model.steps[i][np.argmax(output["label_dist"][j])], c="C0", ls="-.", label="pred max")
                title = f"{obj_name}, loss: "
                title += ", ".join([f"{loss_name}: {loss_value[j]:.2g}" for loss_name, loss_value in loss.items()])
                a.set(title=title)
                a.grid(True)

        ax[0, 1].legend()

    return fig, ax


def batch_evaluate_max_mean(model: MultiDLDL, dataloader: DataLoader) -> Dict[str, np.ndarray]:
    preds_max = []
    preds_mean = []
    true = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(model.device)
            temp_max, temp_mean, temp_true = [], [], []
            for i in range(len(model.sigmas)):
                output = model(images, i)
                output = {k: v.cpu() for k, v in output.items()}

                temp_max.append(model.steps[i][torch.argmax(output["label_dist"], dim=1)].numpy())
                temp_mean.append(model.scale_from_norm(output["exp_norm"], idx=i).numpy())
                temp_true.append(labels["labels"][i].numpy())

            preds_max.extend(np.asarray(temp_max).T)
            preds_mean.extend(np.asarray(temp_mean).T)
            true.extend(np.asarray(temp_true).T)

    return {"preds_max": np.asarray(preds_max), "preds_mean": np.asarray(preds_mean), "trues": np.asarray(true)}


def batch_evaluate_dist(model: MultiDLDL, dataloader: DataLoader) -> Dict:
    results = dict((label, dict()) for label in model.labels)
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(model.device)
            for i, label in enumerate(model.labels):
                output = model(images, i)
                output = {k: v.cpu() for k, v in output.items()}

                true = labels["labels"][i]

                for t, d in zip(true, output["label_dist"]):
                    if not t.item() in results[label].keys():
                        results[label][t.item()] = {"dist": d, "n_samples": 1}
                    else:
                        results[label][t.item()]["dist"] += d
                        results[label][t.item()]["n_samples"] += 1

    for label in model.labels:
        for k in results[label].keys():
            results[label][k]["dist"] = results[label][k]["dist"].numpy()
        results[label] = dict(sorted(results[label].items()))

    return results


def plot_violins(
    model: MultiDLDL, dataloader: DataLoader, fig_kwargs: Optional[Dict] = {}, ax_kwargs: Optional[Dict] = {}
) -> Tuple[plt.Figure, plt.Axes]:
    """Make violin plots for all given input image.

    Args:
        model (MultiDLDL): MultiDLDL model.
        dataloader (DataLoader): Dataloader with images.
        fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to {}.
        ax_kwargs (Optional[Dict], optional): Further axis params. Defaults to {}.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
    """

    model.eval()
    ev = batch_evaluate_max_mean(model, dataloader)

    def postprocess_violin(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
        violin["cbars"].set_visible(False)

    n = ev["preds_max"].shape[1]

    options = {"points": 1024, "showmeans": True, "showextrema": True}

    fig, ax = plt.subplots(n, 1, figsize=(10, 4 * n), tight_layout=True, **fig_kwargs)
    fig.suptitle("predictions")
    if n == 1:
        ax = [ax]
    for i, l in enumerate(model.labels):
        labels = []
        u = np.unique(ev["trues"][:, i])
        width = 0.9 * np.mean(np.diff(u))
        postprocess_violin(
            ax[i].violinplot(
                [ev["preds_max"][:, i][ev["trues"][:, i] == j] - j for j in u], positions=u, widths=width, **options
            ),
            "max",
        )
        postprocess_violin(
            ax[i].violinplot(
                [ev["preds_max"][:, i][ev["trues"][:, i] == j] - j for j in u], positions=u, widths=width, **options
            ),
            "mean",
        )
        ax[i].axhline(0, c="r")
        ax[i].set(title=l, xlim=model.reg_limits[i], xlabel="truth", ylabel="prediction-truth", **ax_kwargs)
        ax[i].grid(True)
        ax[i].legend(*zip(*labels), loc="upper left")

    return fig, ax


def plot_stdhist(
    model: MultiDLDL,
    dataloader: DataLoader,
    bins: Union[int, np.array],
    fig_kwargs: Optional[Dict] = {},
    ax_kwargs: Optional[Dict] = {},
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot histogram for predicted standard deviations.

    Args:
        model (MultiDLDL): MultiDLDL model.
        dataloader (DataLoader): Dataloader with images.
        bins (Union[int, np.array]): Number of bins.
        fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to {}.
        ax_kwargs (Optional[Dict], optional): Further axis params. Defaults to {}.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
    """
    n_sigmas = len(model.labels)
    sigmas = []
    for images, _ in tqdm(dataloader):
        images = images.to(model.device)
        temp = []
        for i in range(n_sigmas):
            output = model(images, i)
            pred_var_norm = torch.sum(
                output["label_dist"] * torch.pow(model.norm_steps[i].to(model.device) - output["exp_norm"][:, None], 2), dim=1
            )
            temp.append(pred_var_norm.detach().cpu().numpy() ** 0.5)
        sigmas.extend(np.asarray(temp).T)
    sigmas = np.stack(sigmas)

    fig, ax = plt.subplots(n_sigmas, 1, figsize=(8, 3 * n_sigmas), tight_layout=True, **fig_kwargs)
    fig.suptitle("distributions of predicted standard deviations")
    if n_sigmas == 1:
        ax = [ax]
    for i, a in enumerate(ax):
        a.set(title=model.labels[i], xlabel="std", ylabel="density", **ax_kwargs)
        a.hist(sigmas[:, i] * (model.reg_limits[i][1] - model.reg_limits[i][0]), bins=bins, density=True)
        a.axvline(model.sigmas[i], c="r")
        a.grid(True)

    return fig, ax


def plot_attention_maps(
    model: MultiDLDL,
    images: torch.Tensor,
    discard_ratio: float = 0.9,
    head_fusion: str = "max",
    fig_kwargs: Optional[Dict] = {},
    ax_kwargs: Optional[Dict] = {},
) -> Tuple[plt.Figure, plt.Axes]:
    """Make attention map plot.

    Args:
        model (MultiDLDL): MultiDLDL model.
        images (torch.Tensor): Test images.
        discard_ratio (float, optional): Discard ratio for attention rollout. Defaults to 0.9.
        head_fusion (str, optional): Head fusion method for attention rollout. Defaults to "max".
        fig_kwargs (Optional[Dict], optional): Further figure params. Defaults to {}.
        ax_kwargs (Optional[Dict], optional): Further axis params. Defaults to {}.

    Raises:
        NotImplementedError: If model is not attention-based.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Tuple with matplotlib figure and axis.
    """
    custom_figsize = fig_kwargs.pop("figsize", None)

    if not (isinstance(model.backbone, ViT) or isinstance(model.backbone, PretrainedViT)):
        raise NotImplementedError("Attention rollout only works for attention-based models!")
    else:
        plot_imgs, all_masks, all_layermasks = [], [], []
        for image in images:
            img_shape = image.shape[-2:]
            att = [a.detach() for a in model.backbone(image[None, :].to(model.device), output_attentions=True).attentions]
            mask, layermasks = attention_rollout(att, discard_ratio=discard_ratio, head_fusion=head_fusion)
            mask = cv2.resize(mask / mask.max(), img_shape)

            img = image[[2, 1, 0]].permute(1, 2, 0).numpy()
            img = rescale_rgb(img)
            plot_imgs.append(img)
            all_masks.append(mask)
            all_layermasks.append(layermasks)

        fig, ax = plt.subplots(
            len(images),
            2 + len(layermasks),
            figsize=custom_figsize if custom_figsize is not None else (2 * len(layermasks), 2 * len(images)),
            tight_layout=True,
            **fig_kwargs,
        )
        for a in ax.ravel():
            a.xaxis.set_ticks([])
            a.yaxis.set_ticks([])

        ax[-1, 0].set(xlabel="RGB", **ax_kwargs)
        ax[-1, 1].set(xlabel="joint", **ax_kwargs)

        for i, img in enumerate(plot_imgs):
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(img.mean(axis=2), alpha=1, cmap="gray")
            ax[i, 1].imshow(all_masks[i], alpha=0.75, cmap="jet")
            for j, lm in enumerate(all_layermasks[i]):
                ax[i, 2 + j].imshow(np.mean(img, axis=2), alpha=1, cmap="gray")
                lm_resized = cv2.resize(lm / lm.max(), img_shape)
                ax[i, 2 + j].imshow(lm_resized, alpha=0.75, cmap="jet")

        for j in range(len(all_layermasks[0])):
            ax[-1, 2 + j].set(xlabel=f"layer {j}", **ax_kwargs)

        return fig, ax
