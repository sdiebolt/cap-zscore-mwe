#!/usr/bin/env python3
"""Minimal CAP z-scoring simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.cluster import kmeans_plusplus

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 0
NY = 50
NX = 50
NT = 1000
TR = 1.0
N_CAPS = 2
NOISE_SD = 0.15
ACTIVE_PROB = 0.6
MAX_ACTIVATION = 1.0

COLORS = {
    "amplitude": "#4C4C4C",
    "mean": "#E17C05",
    "clean": "#54A24B",
    "active": "#4C78A8",
    "cap": "#72B7B2",
    "anticap": "#B279A2",
    "agree": "#54A24B",
    "disagree": "#E45756",
}


def square_mask(y0: int, x0: int, size: int = 10) -> np.ndarray:
    mask = np.zeros((NY, NX), dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


def simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    roi = square_mask(20, 20)
    active = rng.random(NT) < ACTIVE_PROB
    amplitude = np.where(active, rng.uniform(0, MAX_ACTIVATION, NT), 0.0)
    data = rng.normal(scale=NOISE_SD, size=(NT, NY, NX))
    data[:, roi] += amplitude[:, None]
    return data, roi, amplitude, active


def zscore(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def samplewise_zscore_and_l2_normalize(data: np.ndarray) -> np.ndarray:
    flat = data.reshape(len(data), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    flat = flat / std
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return flat / norm


def spherical_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_centers = None
    best_score = -np.inf

    for _ in range(n_init):
        seed = int(rng.integers(0, 2**31 - 1))
        centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=seed)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        labels = None

        for _ in range(max_iter):
            similarity = X @ centers.T
            new_labels = similarity.argmax(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels

            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    new_centers[k] = X[int(rng.integers(0, len(X)))]
                    continue
                center = members.sum(axis=0)
                norm = np.linalg.norm(center)
                new_centers[k] = center / norm if norm > 0 else members[0]
            centers = new_centers

        score = float((X @ centers.T)[np.arange(len(X)), labels].sum())
        if score > best_score:
            best_score = score
            best_labels = labels.copy()
            best_centers = centers.copy()

    assert best_labels is not None and best_centers is not None
    return best_centers, best_labels


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(int), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts, stops, strict=True))


def shade(
    ax: plt.Axes,
    time: np.ndarray,
    mask: np.ndarray,
    color: str,
    label: str,
    ymin: float = 0.0,
    ymax: float = 1.0,
) -> None:
    for i, (start, stop) in enumerate(segments(mask)):
        ax.axvspan(
            time[start],
            time[min(stop, len(time) - 1)] + TR,
            ymin=ymin,
            ymax=ymax,
            color=color,
            alpha=0.35,
            label=label if i == 0 else None,
            lw=0,
        )


def save_nifti(data: np.ndarray, path: Path) -> None:
    nii = nib.Nifti1Image(np.moveaxis(data, 0, -1).astype(np.float32), affine=np.eye(4))
    nib.save(nii, path)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw, roi, amplitude, true_active = simulate()
    cleaned = zscore(raw)
    features = samplewise_zscore_and_l2_normalize(cleaned)

    centers_flat, labels = spherical_kmeans(
        features, n_clusters=N_CAPS, random_state=SEED
    )
    centers = centers_flat.reshape(N_CAPS, NY, NX)

    roi_means = centers[:, roi].mean(axis=1)
    positive_cap = int(np.argmax(roi_means))
    pred_positive = labels == positive_cap
    pred_negative = ~pred_positive
    above_mean = amplitude > amplitude.mean()
    agreement = pred_positive == above_mean
    disagree = ~agreement
    active_below_mean = true_active & ~above_mean
    disagree_pct = 100 * float(disagree.mean())
    active_below_mean_pct = 100 * float(active_below_mean.mean())

    reconstructed = centers[labels]
    save_nifti(raw[:, None, :, :], RESULTS_DIR / "simulated_recording_raw.nii.gz")
    save_nifti(
        cleaned[:, None, :, :], RESULTS_DIR / "simulated_recording_cleaned.nii.gz"
    )
    save_nifti(
        reconstructed[:, None, :, :],
        RESULTS_DIR / "simulated_cap_reconstruction.nii.gz",
    )

    time = np.arange(NT) * TR
    roi_ts_clean = cleaned[:, roi].mean(axis=1)
    mean_amp = amplitude.mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].imshow(roi, cmap="gray")
    axes[0].set_title("Ground-truth ROI")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].plot(time, amplitude, lw=1, color=COLORS["amplitude"])
    axes[1].axhline(
        mean_amp, ls="--", color=COLORS["mean"], label="mean injected amplitude"
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Injected activation amplitude in center ROI")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, roi_ts_clean, lw=0.8, color=COLORS["clean"])
    axes[2].axhline(0, ls="--", color="black")
    axes[2].set_ylabel("Z-score")
    axes[2].set_xlabel("Time")
    axes[2].set_title("ROI mean after z-scoring")
    fig.savefig(
        RESULTS_DIR / "simulation_roi_and_timeseries.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, ax in enumerate(axes):
        vmax = float(np.abs(centers[i]).max())
        im = ax.imshow(centers[i], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        label = "CAP" if i == positive_cap else "anti-CAP"
        ax.set_title(f"{label} (ROI mean {roi_means[i]:+.2f})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(RESULTS_DIR / "simulation_cap_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        3, 1, figsize=(20, 7), sharex=True, constrained_layout=True
    )
    axes[0].imshow(
        above_mean.astype(float)[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["white", COLORS["mean"]]),
        extent=[time[0], time[-1] + TR, 0, 0.5],
        origin="lower",
    )
    axes[0].imshow(
        true_active.astype(float)[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["white", COLORS["active"]]),
        extent=[time[0], time[-1] + TR, 0.5, 1.0],
        origin="lower",
    )
    axes[0].axhline(0.5, color="black", lw=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([0.25, 0.75])
    axes[0].set_yticklabels(["> mean", "active"])
    axes[0].set_ylabel("Truth")
    axes[0].set_title("Ground-truth states")
    axes[0].plot([], [], color=COLORS["active"], lw=6, label="top: activation present")
    axes[0].plot([], [], color=COLORS["mean"], lw=6, label="bottom: amplitude > mean")
    axes[0].legend(loc="upper right")

    axes[1].imshow(
        pred_negative.astype(float)[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["white", COLORS["anticap"]]),
        extent=[time[0], time[-1] + TR, 0, 0.5],
        origin="lower",
    )
    axes[1].imshow(
        pred_positive.astype(float)[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["white", COLORS["cap"]]),
        extent=[time[0], time[-1] + TR, 0.5, 1.0],
        origin="lower",
    )
    axes[1].axhline(0.5, color="black", lw=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([0.25, 0.75])
    axes[1].set_yticklabels(["anti-CAP", "CAP"])
    axes[1].set_ylabel("Pred")
    axes[1].set_title("Predicted CAP states")
    axes[1].plot([], [], color=COLORS["cap"], lw=6, label="top: CAP")
    axes[1].plot([], [], color=COLORS["anticap"], lw=6, label="bottom: anti-CAP")
    axes[1].legend(loc="upper right")

    shade(axes[2], time, agreement, COLORS["agree"], "CAP agrees with amplitude > mean")
    shade(
        axes[2],
        time,
        disagree,
        COLORS["disagree"],
        "CAP disagrees with amplitude > mean",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Match")
    axes[2].set_title(
        f"Agreement between CAP and amplitude > mean | disagrees: {disagree_pct:.1f}% | active but <= mean: {active_below_mean_pct:.1f}%"
    )
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Time")
    fig.savefig(
        RESULTS_DIR / "simulation_states_and_agreement.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"CAP vs true active: {np.mean(pred_positive == true_active):.3f}")
    print(f"CAP vs amplitude > mean: {np.mean(pred_positive == above_mean):.3f}")
    print(
        f"anti-CAP vs amplitude <= mean: {np.mean(pred_negative == (~above_mean)):.3f}"
    )
    print(f"disagrees: {disagree_pct:.1f}%")
    print(f"active but <= mean: {active_below_mean_pct:.1f}%")
    print(f"Saved outputs to {RESULTS_DIR}")

    assert centers.shape == (2, NY, NX)
    assert labels.shape == (NT,)
    assert not np.any(pred_positive & pred_negative)


if __name__ == "__main__":
    main()
