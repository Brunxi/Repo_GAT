from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..config import PipelineConfig
from .common import ensure_dir, prepare_sequence_artifacts


def plot_attention_and_importance(
    sequence: str,
    protein_name: str,
    checkpoint_path: Path,
    config: PipelineConfig,
    fold_number: Optional[int] = None,
    inference_dir: Path | str = Path("inference_results"),
    explain_dir: Path | str = Path("gnn_results"),
    output_dir: Path | str = Path("graficos"),
    top_fraction: float = 0.1,
    explainer_steps: int = 11,
    explainer_epochs: Optional[int] = None,
    explainer_seed: int = 42,
) -> tuple[Path, Path]:
    """Generate separate figures for attention/importance lines and the contact map."""

    sequence = sequence.strip()
    checkpoint_path = Path(checkpoint_path)
    charts_root = ensure_dir(output_dir)

    attention_df, importance_df, contact_map = prepare_sequence_artifacts(
        sequence=sequence,
        protein_name=protein_name,
        checkpoint_path=checkpoint_path,
        config=config,
        fold_number=fold_number,
        inference_dir=inference_dir,
        explain_dir=explain_dir,
        top_fraction=top_fraction,
        explainer_steps=explainer_steps,
        explainer_epochs=explainer_epochs,
        explainer_seed=explainer_seed,
    )

    chart_dir = ensure_dir(Path(charts_root) / protein_name)
    chart_lines_path = chart_dir / f"{protein_name}_attention_importance.png"
    chart_contact_path = chart_dir / f"{protein_name}_contact_map_teal.png"
    chart_contact_alt_path = chart_dir / f"{protein_name}_contact_map_diverging.png"

    _style_plot()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.6]},
    )

    colours = {
        "sand": "#dfc07b",
        "bronze": "#b06b32",
        "pearl": "#ced1cf",
        "mist": "#dfe4e1",
        "slate": "#808c86",
        "drift": "#8ea59d",
        "teal": "#2f6a6a",
        "graphite": "#222222",
    }

    ax_att = axes[0]
    ax_att.fill_between(
        attention_df["position"],
        attention_df["total_attention"],
        color=colours["drift"],
        alpha=0.6,
        linewidth=0,
    )
    ax_att.plot(
        attention_df["position"],
        attention_df["total_attention"],
        color=colours["teal"],
        linewidth=2.2,
    )
    ax_att.set_ylabel("Attention score", color=colours["graphite"])
    _style_axis(ax_att, colours)
    ax_att.set_title(f"{protein_name} · Attention & GNNExplainer profile", color=colours["graphite"], pad=18)

    ax_imp = axes[1]
    ax_imp.fill_between(
        importance_df["position"],
        importance_df["node_importance"],
        color=colours["sand"],
        alpha=0.6,
        linewidth=0,
    )
    ax_imp.plot(
        importance_df["position"],
        importance_df["node_importance"],
        color=colours["bronze"],
        linewidth=2.0,
    )
    ax_imp.set_ylabel("Node importance", color=colours["graphite"])
    _style_axis(ax_imp, colours)

    ax_domains = axes[2]
    sequence_end = int(attention_df["position"].iloc[-1])
    _plot_domains(ax_domains, colours, sequence_end)
    ax_domains.set_xlabel("Residue position", color=colours["graphite"])
    _style_axis(ax_domains, colours, show_ticks=False)

    plt.tight_layout(h_pad=2.2, pad=1.6)
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
    fig.savefig(chart_lines_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)

    fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5.5))
    teal_cmap = mcolors.LinearSegmentedColormap.from_list(
        "contact_teal",
        ["#ffffff", colours["pearl"], colours["slate"], colours["teal"]],
    )
    im = ax_cm.imshow(contact_map, cmap=teal_cmap, origin="lower", aspect="equal")
    ax_cm.set_xlabel("Residue index", color=colours["graphite"])
    ax_cm.set_ylabel("Residue index", color=colours["graphite"])
    ax_cm.tick_params(axis="both", colors=colours["graphite"])
    for spine in ax_cm.spines.values():
        spine.set_color("#c0c2c1")
        spine.set_linewidth(0.8)
    ax_cm.set_facecolor("#FFFFFF")
    cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Contact probability", color=colours["graphite"])
    cbar.ax.tick_params(color=colours["graphite"], labelcolor=colours["graphite"])
    cbar.outline.set_edgecolor("#c0c2c1")
    cbar.outline.set_linewidth(0.8)
    fig_cm.tight_layout()
    fig_cm.savefig(chart_contact_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig_cm)

    fig_alt, ax_alt = plt.subplots(figsize=(6.5, 5.5))
    diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
        "contact_diverging",
        ["#b7d4f4", "#ffffff", "#d04f46"],
    )
    vmax_value = max(0.6, float(contact_map.max()))
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.2, vmax=vmax_value)
    im_alt = ax_alt.imshow(contact_map, cmap=diverging_cmap, norm=norm, origin="lower", aspect="equal")
    ax_alt.set_xlabel("Residue index", color=colours["graphite"])
    ax_alt.set_ylabel("Residue index", color=colours["graphite"])
    ax_alt.tick_params(axis="both", colors=colours["graphite"])
    for spine in ax_alt.spines.values():
        spine.set_color("#c0c2c1")
        spine.set_linewidth(0.8)
    ax_alt.set_facecolor("#FFFFFF")
    cbar_alt = fig_alt.colorbar(im_alt, ax=ax_alt, fraction=0.046, pad=0.04)
    cbar_alt.ax.set_ylabel("Contact probability", color=colours["graphite"])
    cbar_alt.ax.tick_params(color=colours["graphite"], labelcolor=colours["graphite"])
    cbar_alt.outline.set_edgecolor("#c0c2c1")
    cbar_alt.outline.set_linewidth(0.8)
    fig_alt.tight_layout()
    fig_alt.savefig(chart_contact_alt_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig_alt)

    return chart_lines_path, chart_contact_path, chart_contact_alt_path


def _style_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.edgecolor": "#c0c2c1",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.facecolor": "#FFFFFF",
        }
    )


def _style_axis(ax, colours, show_ticks: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if show_ticks:
        ax.tick_params(axis="both", which="both", colors=colours["graphite"])
    else:
        ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelright=False)


def _plot_domains(ax, colours, sequence_end: int) -> None:
    domain_intervals = [
        (1, 17, "Signal peptide", "#a3c948"),
        (34, min(270, sequence_end), "Pectate lyase domain", "#d07d3a"),
    ]
    catalytic_sites = [122, 151, 155, 208]
    active_sites = [125, 135, 157, 175, 178, 181, 207, 213, 259]

    ax.set_ylim(0, 1.2)
    ax.set_xlim(1, max(sequence_end, max(end for _, end, _, _ in domain_intervals)))
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c0c2c1")
        spine.set_linewidth(0.8)

    for start, end, label, color in domain_intervals:
        if start > end:
            continue
        rect = Rectangle((start, 0.45), end - start + 1, 0.2, facecolor=color, alpha=0.85, edgecolor="none")
        ax.add_patch(rect)

    site_alpha = 0.85
    for residue in catalytic_sites:
        if 1 <= residue <= sequence_end:
            rect = Rectangle((residue - 0.5, 0.05), 1.0, 0.25, facecolor="#c2474b", alpha=site_alpha, edgecolor="none")
            ax.add_patch(rect)
    for residue in active_sites:
        if 1 <= residue <= sequence_end:
            rect = Rectangle((residue - 0.5, 0.05), 1.0, 0.25, facecolor="#8961d1", alpha=site_alpha, edgecolor="none")
            ax.add_patch(rect)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#a3c948", edgecolor="none", alpha=0.85, label="Signal peptide"),
        Rectangle((0, 0), 1, 1, facecolor="#d07d3a", edgecolor="none", alpha=0.85, label="Pectate lyase domain"),
        Rectangle((0, 0), 1, 1, facecolor="#c2474b", edgecolor="none", alpha=site_alpha, label="Catalytic site"),
        Rectangle((0, 0), 1, 1, facecolor="#8961d1", edgecolor="none", alpha=site_alpha, label="Active site"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        frameon=False,
        ncol=2,
        columnspacing=2.5,
        handletextpad=0.9,
    )
