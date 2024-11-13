# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent


@dataclass
class FlybySnapshot:
    host: pd.Series
    perturber: pd.Series
    particles: pd.DataFrame


# %%

df = pd.read_csv(ROOT_DIR / "data/simdata_ias15.csv")


# %%
def get_time_slice(df: pd.DataFrame, time_idx: int) -> Tuple[int, pd.DataFrame]:
    groups_time = df.groupby("time")
    time = list(groups_time.indices.keys())[time_idx]
    time_group = groups_time.get_group(time)
    time_group = time_group.drop(columns=["time", "vx", "vy", "vz", "source"])
    time_group.reset_index(drop=True, inplace=True)
    return time, time_group


def plot_flyby(
    host_pos: pd.Series,
    perturber_pos: pd.Series,
    particles: pd.DataFrame,
    title: str | None = None,
    xlim: Tuple[int, int] | None = None,
    ylim: Tuple[int, int] | None = None,
):
    plt.figure(figsize=(10, 10))
    plt.scatter(particles["x"], particles["y"], s=0.2)
    plt.scatter(host_pos["x"], host_pos["y"], s=10, c="red")
    plt.scatter(perturber_pos["x"], perturber_pos["y"], s=10, c="blue")
    plt.gca().set_aspect("equal")
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.show()


def snapshot(group: pd.DataFrame) -> FlybySnapshot:
    host = group[group["id"] == 0].copy()
    perturber = group[group["id"] == 1].copy()
    particles = group[group["id"] > 1].copy()
    # Center the host at the origin
    particles["x"] -= host["x"].iloc[0]
    particles["y"] -= host["y"].iloc[0]
    particles["z"] -= host["z"].iloc[0]
    perturber["x"] -= host["x"].iloc[0]
    perturber["y"] -= host["y"].iloc[0]
    perturber["z"] -= host["z"].iloc[0]
    host["x"] = 0
    host["y"] = 0
    host["z"] = 0
    return FlybySnapshot(host, perturber, particles)


def density_image(
    snapshot: FlybySnapshot,
    n_pixels: int,
    xlim: Tuple[int, int] | None = None,
    ylim: Tuple[int, int] | None = None,
    gaussian_sigma: int | None = None,
    n_colors: int | None = None,
    apply_log: int | None = None,
) -> np.ndarray:
    image, _, _ = np.histogram2d(
        snapshot.particles["y"],
        snapshot.particles["x"],
        bins=n_pixels,
        range=[ylim, xlim],
    )
    # Get pixels that are empty
    mask = image == 0
    print(mask.sum())

    # Apply log scale to histogram (multiple times to increase contrast)
    if apply_log is not None:
        for _ in range(apply_log):
            image = np.log1p(image)

    # Apply Gaussian filter to spread out particles
    if gaussian_sigma is not None:
        image = gaussian_filter(image, sigma=gaussian_sigma, mode="constant")
    image = image / image.max()

    # Apply binning to convert to n-color image
    if n_colors is not None:
        color_bins = np.geomspace(1, n_colors, n_colors)
        color_bins = color_bins / color_bins.max()
        image = np.digitize(image, color_bins)

    # Apply mask to empty pixels
    if gaussian_sigma is None and n_colors is None:
        image[mask] = 0

    return image


def apply_colormap(image: np.ndarray, cmap: str):
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    return mpl.colormaps[cmap](norm(image))


# %%


@dataclass
class PlotParameters:
    slice_idx: int
    xlim: Tuple[int, int]
    ylim: Tuple[int, int]
    n_pixels: int
    gaussian_sigma: int | None
    apply_log: int | None
    n_colors: int | None
    colormap: str


good_parameters = [
    PlotParameters(
        slice_idx=1,
        xlim=(-1e3, 1e3),
        ylim=(-1e3, 1e3),
        n_pixels=600,
        gaussian_sigma=None,
        apply_log=2,
        n_colors=None,
        colormap="Oranges",
    ),
    PlotParameters(
        slice_idx=1,
        xlim=(-3e3, 3e3),
        ylim=(-3e3, 3e3),
        n_pixels=600,
        gaussian_sigma=None,
        apply_log=1,
        n_colors=None,
        colormap="Oranges",
    ),
]


for param_idx, param in enumerate(good_parameters):
    time, group = get_time_slice(df, param.slice_idx)
    snap = snapshot(group)
    image = density_image(
        snap,
        n_pixels=param.n_pixels,
        xlim=param.xlim,
        ylim=param.ylim,
        gaussian_sigma=param.gaussian_sigma,
        n_colors=param.n_colors,
        apply_log=param.apply_log,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=param.colormap)
    # plt.show()
    image_map = apply_colormap(image, param.colormap)
    image_map[image == 0] = np.array([1, 1, 1, 0])
    pil_image = Image.fromarray((image_map * 255).astype(np.uint8))
    pil_image.save(ROOT_DIR / f"data/snap_{param_idx}.png")

# %%
