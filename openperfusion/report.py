"""Figures for the phantom and for reference-software comparisons."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pipeline import PipelineResult
from .phantom import PhantomTruth


def phantom_figure(res: PipelineResult, truth: PhantomTruth, z: int, path: str, title: str = ""):
    m = res.maps
    panels = [
        ("CBF true", truth.cbf[:, :, z], (0, 80)), ("CBF est", m.cbf[:, :, z], (0, 80)),
        ("CBV true", truth.cbv[:, :, z], (0, 20)), ("CBV est", m.cbv[:, :, z], (0, 20)),
        ("Tmax true (s)", truth.tmax[:, :, z] * truth.tissue_mask[:, :, z], (0, 8)), ("Tmax est (s)", m.tmax[:, :, z], (0, 8)),
        ("MTT true (s)", truth.mtt[:, :, z], (0, 18)), ("MTT est (s)", m.mtt[:, :, z], (0, 18)),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    axes = axes.ravel()
    for ax, (name, img, clim) in zip(axes[:8], panels):
        im = ax.imshow(img.T, origin="lower", cmap="jet", vmin=clim[0], vmax=clim[1])
        ax.set_title(name); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[8]
    t = truth.t
    ax.plot(t, truth.aif, "k-", label="true AIF")
    ax.plot(t, res.aif.aif, "r--", label=f"selected AIF x k_av={res.aif.k_av:.2f}")
    if res.aif.vof is not None:
        ax.plot(t, res.aif.vof, "b:", label="selected VOF")
    ax.set_xlabel("s"); ax.set_ylabel("HU"); ax.legend(fontsize=8); ax.set_title("Input functions")
    ax = axes[9]
    ax.imshow(res.mask[:, :, z].T, origin="lower", cmap="gray")
    ax.contour(res.aif.aif_mask[:, :, z].T, colors="r", linewidths=0.8)
    if res.aif.vof_mask is not None:
        ax.contour(res.aif.vof_mask[:, :, z].T, colors="b", linewidths=0.8)
    ax.contour(res.thresholds.core[:, :, z].T, colors="magenta", linewidths=0.8)
    ax.contour(res.thresholds.hypo[:, :, z].T, colors="lime", linewidths=0.8)
    ax.set_title("mask, AIF(red) VOF(blue), core(magenta) Tmax>6(green)"); ax.axis("off")
    fig.suptitle(title or f"Digital phantom, slice delay={truth.delay[:, :, z].max():.0f} s, method={m.method}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def case_figure(ours, ref: dict, thr_ours, thr_ref, z: int, path: str, lesion=None, title: str = ""):
    """Side-by-side of our maps vs reference maps for one slice."""
    items = [("cbf", (0, 80)), ("cbv", (0, 10)), ("mtt", (0, 20)), ("tmax", (0, 12))]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for j, (k, clim) in enumerate(items):
        axes[0, j].imshow(getattr(ours, k)[:, :, z].T, origin="lower", cmap="jet", vmin=clim[0], vmax=clim[1]); axes[0, j].set_title(f"ours {k}")
        if k in ref:
            axes[1, j].imshow(ref[k][:, :, z].T, origin="lower", cmap="jet", vmin=clim[0], vmax=clim[1]); axes[1, j].set_title(f"reference {k}")
        axes[0, j].axis("off"); axes[1, j].axis("off")
    for j, (name, t) in enumerate([("ours", thr_ours), ("reference", thr_ref)]):
        ax = axes[2, j]
        ax.imshow(ours.mask[:, :, z].T, origin="lower", cmap="gray")
        ax.contour(t.core[:, :, z].T, colors="magenta", linewidths=0.8)
        ax.contour(t.hypo[:, :, z].T, colors="lime", linewidths=0.8)
        if lesion is not None:
            ax.contour(lesion[:, :, z].T, colors="yellow", linewidths=0.8)
        ax.set_title(f"{name}: core {t.core_ml:.0f} mL, Tmax>6 {t.hypo_ml:.0f} mL"); ax.axis("off")
    axes[2, 2].axis("off"); axes[2, 3].axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
