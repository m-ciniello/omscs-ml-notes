"""Build every report figure from the on-disk results.

Usage:
    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py --only bj_dp    # one group
    python scripts/make_figures.py --list          # list available groups

All figures write to `figures/` as PNGs at 150 DPI. This script is
idempotent — re-running overwrites in place. Nothing here touches the
results directory, so it's safe to re-run as often as you want while
iterating on styling.

Figure index:
    01_bj_dp_convergence.png        VI/PI convergence traces
    02_bj_policy_heatmap.png        VI-derived hit/stick policy grids
    03_bj_tabular_curves.png        SARSA vs Q-Learning training curves
    04_bj_hp_sensitivity.png        α / γ / ε-decay / n_episodes sweeps
    05_bj_agent_comparison.png      Final eval return bar chart, all agents
    06_cp_tabular_curves.png        CartPole SARSA vs Q-Learning curves
    07_cp_tabular_hp.png            α / γ / n_bins sweeps
    08_cp_dp_nbins.png              VI / PI / trained-ε over grids
    09_cp_dp_budget_and_eps.png     sampling budget + exploration study
    10_cp_agent_comparison.png      Final eval return bar chart, all agents
    11_dqn_ablation_bars.png        DQN Rainbow-medium ablation (final eval)
    12_dqn_learning_curves.png      DQN training curves across variants
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Matplotlib's default config dir isn't writable on sandboxed runs. Redirect
# before any other mpl-touching import.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".logs" / ".mpl-cache"))
(Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.configs import _fmt_value
from src.experiments.runner import load_runs

FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Styling + IO helpers
# ---------------------------------------------------------------------------

def set_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


def save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def _safe_load(name: str):
    """Load runs, returning None (with a warning) when the experiment is
    either unregistered or not-yet-run on disk. Lets figure functions
    gracefully degrade instead of failing the whole orchestrator.
    """
    try:
        return load_runs(name)
    except FileNotFoundError:
        print(f"  [skip] {name}: no on-disk results")
        return None
    except KeyError:
        print(f"  [skip] {name}: not in experiment registry")
        return None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Same-length rolling mean with unbiased edges.

    Uses cumulative sums and divides by the actual count at each position,
    so the first and last few samples aren't pulled toward zero by a naive
    `np.convolve(..., mode='same')` that pads with zeros.
    """
    x = x.astype(float)
    if window <= 1:
        return x
    half = window // 2
    cs = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half + 1, n)
    return (cs[hi] - cs[lo]) / (hi - lo)


def smoothed_seeds(runs, key: str, window: int) -> np.ndarray:
    """Stack smoothed curves across seeds.

    Returns an array of shape (n_seeds, min_length). Trailing values are
    clipped to the shortest run so that a naive mean across seeds stays
    aligned when runs differ in length (shouldn't happen in our setup
    since episode counts are fixed per experiment, but guarded anyway).
    """
    curves = [rolling_mean(np.asarray(r.result[key], dtype=float), window)
              for r in runs]
    min_len = min(c.shape[0] for c in curves)
    return np.stack([c[:min_len] for c in curves], axis=0)


def mean_ci(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed-mean + symmetric 95% CI around that mean.

    With only 5 seeds we use 1.96 · SE (treat the CLT as good enough for
    visualisation — the report text is careful to quote CIs as "approximate").
    """
    mean = stacked.mean(axis=0)
    n = stacked.shape[0]
    se = stacked.std(axis=0, ddof=1) / np.sqrt(max(n, 2))
    half = 1.96 * se
    return mean, mean - half, mean + half


def final_eval_scalar(runs) -> np.ndarray:
    return np.array([
        float(np.mean(r.result["eval_returns"])) for r in runs
    ])


def final_eval_stats(runs) -> tuple[float, float]:
    """Return (mean, 1.96·SE) of per-seed mean eval return."""
    scalars = final_eval_scalar(runs)
    n = max(len(scalars), 2)
    se = scalars.std(ddof=1) / np.sqrt(n)
    return float(scalars.mean()), 1.96 * se


def collect_sweep_points(
    values,
    name_fn,
    *,
    skip_missing: bool = True,
) -> tuple[list, list[float], list[float]]:
    """Load runs for each value, return ``(xs, means, ci95_halfwidths)``.

    ``skip_missing`` toggles alignment behaviour for missing experiments:
    True drops them entirely, False keeps the x-position with NaN/0 for
    bar plots that need fixed positions.
    """
    xs, means, errs = [], [], []
    for v in values:
        runs = _safe_load(name_fn(v))
        if runs is None:
            if not skip_missing:
                xs.append(v)
                means.append(np.nan)
                errs.append(0.0)
            continue
        m, e = final_eval_stats(runs)
        xs.append(v)
        means.append(m)
        errs.append(e)
    return xs, means, errs


def pad_and_mean(
    arrays: list,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-pad 1-D arrays to equal length; return ``(stacked, nanmean)``."""
    max_len = max(len(a) for a in arrays)
    padded = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=float)
        padded[i, :len(a)] = a
    return padded, np.nanmean(padded, axis=0)


def errorbar_sweep(
    ax,
    xs,
    means,
    errs,
    *,
    color: str,
    label: str | None = None,
    xlabels: list | None = None,
    rotation: int = 0,
    markersize: int = 5,
) -> None:
    """Standard error-bar plot with integer positions and optional xtick labels."""
    if not xs:
        return
    positions = list(range(len(xs)))
    ax.errorbar(positions, means, yerr=errs, marker="o",
                markersize=markersize, linewidth=1.5, capsize=3,
                color=color, label=label)
    ax.set_xticks(positions)
    if xlabels is not None:
        ax.set_xticklabels(xlabels, rotation=rotation)


# ---------------------------------------------------------------------------
# Blackjack figures
# ---------------------------------------------------------------------------

def fig_bj_dp_convergence() -> None:
    vi = _safe_load("blackjack_vi_default")
    pi = _safe_load("blackjack_pi_default")
    if vi is None or pi is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # VI: sweep_deltas vs sweep number (one curve per seed + mean)
    ax = axes[0]
    for r in vi:
        deltas = r.result["history"]["sweep_deltas"]
        ax.plot(np.arange(1, len(deltas) + 1), deltas,
                color="C0", alpha=0.25, linewidth=1)
    _, mean = pad_and_mean([r.result["history"]["sweep_deltas"] for r in vi])
    ax.plot(np.arange(1, len(mean) + 1), mean, color="C0", linewidth=2, label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Bellman residual  max_s |V_new(s) − V_old(s)|")
    ax.set_title("Value Iteration")
    ax.legend()

    # PI: show two complementary views of "work done" per outer iteration.
    #   - bars: eval_sweeps_per_outer (the actual cost of each outer iter,
    #     since each iter runs a full policy-evaluation sub-loop)
    #   - line: policy_changes_per_outer (how many states flipped actions,
    #     which is what drives termination — PI stops when this hits 0)
    # This avoids the log-scale pathology of plotting `bellman_residual_per_outer`
    # directly, where outer-iter-1 often has residual 0 (V=0 policy=0 is a
    # self-consistent starting point, so eval converges instantly).
    ax = axes[1]

    _, sweeps_mean = pad_and_mean([r.result["history"]["eval_sweeps_per_outer"] for r in pi])
    _, changes_mean = pad_and_mean([r.result["history"]["policy_changes_per_outer"] for r in pi])
    outer = np.arange(1, len(sweeps_mean) + 1)

    ax.bar(outer, sweeps_mean, color="C1", alpha=0.7, edgecolor="black",
           linewidth=0.6, label="eval sweeps (mean)")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Eval sweeps per outer iteration")
    ax.set_xticks(outer)

    ax2 = ax.twinx()
    ax2.plot(outer, changes_mean, color="C3", linewidth=2, marker="o",
             markersize=6, label="policy changes (mean)")
    ax2.set_ylabel("States flipping action")
    ax2.grid(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.set_title("Policy Iteration")

    fig.suptitle("Blackjack: DP convergence (5 seeds)", y=1.02)
    save(fig, "01_bj_dp_convergence.png")


def fig_bj_policy_heatmap() -> None:
    """Greedy-policy hit/stick grid for VI (per usable-ace setting).

    Uses seed 0 of blackjack_vi_default (deterministic DP — all seeds produce
    the same policy anyway, so picking seed 0 is not a cherry-pick).
    """
    vi = _safe_load("blackjack_vi_default")
    if vi is None:
        return
    policy = vi[0].result["policy"]

    # Player sum 4-21, dealer showing 1-10, one grid per usable_ace value.
    player_sums = list(range(4, 22))
    dealer_cards = list(range(1, 11))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, usable_ace in zip(axes, (False, True)):
        grid = np.full((len(player_sums), len(dealer_cards)), np.nan)
        for i, ps in enumerate(player_sums):
            for j, dc in enumerate(dealer_cards):
                a = policy.get((ps, dc, int(usable_ace)))
                if a is not None:
                    grid[i, j] = a
        sns.heatmap(
            grid,
            ax=ax,
            xticklabels=dealer_cards,
            yticklabels=player_sums,
            cmap=sns.color_palette(["#ffb347", "#77b4ff"]),  # 0=stick, 1=hit
            cbar=False,
            linewidths=0.3,
            linecolor="white",
        )
        ax.set_title(f"Usable ace: {usable_ace}")
        ax.set_xlabel("Dealer showing")
        ax.invert_yaxis()
    axes[0].set_ylabel("Player sum")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#ffb347", edgecolor="black", label="stick (0)"),
        Patch(facecolor="#77b4ff", edgecolor="black", label="hit (1)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.suptitle("Blackjack: VI-derived optimal policy", y=1.08)
    save(fig, "02_bj_policy_heatmap.png")


def fig_bj_tabular_curves() -> None:
    sarsa = _safe_load("blackjack_sarsa_default")
    ql = _safe_load("blackjack_qlearning_default")
    if sarsa is None or ql is None:
        return

    window = 5000  # 200k episodes → ~2.5% smoothing window
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for runs, color, label in [
        (sarsa, "C2", "SARSA"),
        (ql,    "C3", "Q-Learning"),
    ]:
        stacked = smoothed_seeds(runs, "train_returns", window)
        mean, lo, hi = mean_ci(stacked)
        x = np.arange(mean.shape[0])
        ax.plot(x, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)

    # Optimal baseline (VI final eval)
    vi = _safe_load("blackjack_vi_default")
    if vi is not None:
        vi_mean = final_eval_scalar(vi).mean()
        ax.axhline(vi_mean, color="black", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"VI optimum ≈ {vi_mean:+.3f}")

    ax.set_xlabel(f"Training episode (rolling mean, window={window})")
    ax.set_ylabel("Training return")
    ax.set_title("Blackjack tabular learning curves (5 seeds, 95% CI)")
    ax.legend(loc="lower right")
    save(fig, "03_bj_tabular_curves.png")


def _sweep_bar(ax, runs_by_value: dict, title: str, xlabel: str,
               value_formatter=str) -> None:
    """Helper: draw one error-bar point per sweep value."""
    xs = sorted(runs_by_value.keys())
    means, errs = [], []
    for v in xs:
        scalars = final_eval_scalar(runs_by_value[v])
        means.append(scalars.mean())
        se = scalars.std(ddof=1) / np.sqrt(max(len(scalars), 2))
        errs.append(1.96 * se)
    labels = [value_formatter(v) for v in xs]
    ax.errorbar(range(len(xs)), means, yerr=errs, marker="o",
                markersize=6, linewidth=1.5, capsize=4, color="C4")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def fig_bj_hp_sensitivity() -> None:
    """Two tabular panels (α / ε-decay) + one DP panel (γ).

    γ isn't swept for tabular Blackjack (undiscounted makes sense since the
    task is a single hand), but is the headline DP study — worth showing on
    the same figure to anchor the "DP has different HPs than tabular RL" story.
    """
    tabular_sweep_config = [
        ("alpha",       "Learning rate α",
         [0.01, 0.05, 0.1, 0.2], lambda v: f"{v:g}"),
        ("eps_decay",   "ε-decay episodes",
         [10_000, 50_000, 100_000, 200_000], lambda v: f"{v//1000}k"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for panel_idx, (sweep_key, xlabel, values, value_label) in enumerate(tabular_sweep_config):
        ax = axes[panel_idx]
        for agent, color, prefix in [
            ("SARSA", "C2", f"blackjack_sarsa_{sweep_key}_sweep_"),
            ("Q-Learning", "C3", f"blackjack_qlearning_{sweep_key}_sweep_"),
        ]:
            xs, means, errs = collect_sweep_points(
                values, lambda v, p=prefix: f"{p}{_fmt_value(v)}"
            )
            errorbar_sweep(ax, xs, means, errs, color=color, label=agent,
                           xlabels=[value_label(v) for v in xs])
        ax.set_title(f"tabular: {xlabel}")
        ax.set_xlabel(xlabel)
        if panel_idx == 0:
            ax.legend(loc="lower right")

    # Third panel: γ sweep for VI and PI.
    ax = axes[2]
    gamma_values = [0.8, 0.9, 0.95, 0.99, 1.0]
    for agent, color, prefix in [
        ("VI", "C0", "blackjack_vi_gamma_sweep_"),
        ("PI", "C1", "blackjack_pi_gamma_sweep_"),
    ]:
        xs, means, errs = collect_sweep_points(
            gamma_values, lambda v, p=prefix: f"{p}{_fmt_value(v)}"
        )
        errorbar_sweep(ax, xs, means, errs, color=color, label=agent,
                       xlabels=[f"{v:g}" for v in xs])
    ax.set_title("DP: Discount γ")
    ax.set_xlabel("γ")
    ax.legend(loc="lower right")

    axes[0].set_ylabel("Final eval return")
    fig.suptitle("Blackjack: 1-D hyperparameter sweeps (5 seeds each, 95% CI)",
                 y=1.03)
    fig.tight_layout()
    save(fig, "04_bj_hp_sensitivity.png")


def fig_bj_agent_comparison() -> None:
    """Final eval return for each agent at its default config."""
    agents = [
        ("Random",    None,  "C7"),
        ("VI",        "blackjack_vi_default", "C0"),
        ("PI",        "blackjack_pi_default", "C1"),
        ("SARSA (default)",      "blackjack_sarsa_default",      "C2"),
        ("Q-Learning (default)", "blackjack_qlearning_default",  "C3"),
    ]
    # Random baseline: derive analytically from game, or hardcode. Blackjack
    # uniform-random loses ~30% of hands → return ≈ -0.30. Skip Random for now.
    names = [a[0] for a in agents if a[1] is not None]
    colors = [a[2] for a in agents if a[1] is not None]
    means, errs = [], []
    for a in agents:
        if a[1] is None:
            continue
        runs = _safe_load(a[1])
        m, e = final_eval_stats(runs)
        means.append(m)
        errs.append(e)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=errs, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Final eval return")
    ax.set_title("Blackjack: agent comparison (5 seeds, 95% CI)")
    # Annotate bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m,
                f"{m:+.3f}",
                ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
    save(fig, "05_bj_agent_comparison.png")


# ---------------------------------------------------------------------------
# CartPole figures
# ---------------------------------------------------------------------------

def fig_cp_tabular_curves() -> None:
    sarsa = _safe_load("cartpole_sarsa_default")
    ql = _safe_load("cartpole_qlearning_default")
    if sarsa is None or ql is None:
        return

    # CartPole episode count is ~10k, so use a smaller window than Blackjack.
    n_ep = min(len(r.result["train_returns"]) for r in sarsa + ql)
    window = max(50, n_ep // 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for runs, color, label in [
        (sarsa, "C2", "SARSA"),
        (ql,    "C3", "Q-Learning"),
    ]:
        stacked = smoothed_seeds(runs, "train_returns", window)
        mean, lo, hi = mean_ci(stacked)
        x = np.arange(mean.shape[0])
        ax.plot(x, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.axhline(500, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label="max (truncation at 500)")
    ax.set_xlabel(f"Training episode (rolling mean, window={window})")
    ax.set_ylabel("Training return (step count)")
    ax.set_title("CartPole tabular learning curves at default n_bins "
                 "(5 seeds, 95% CI)")
    ax.legend(loc="upper left")
    save(fig, "06_cp_tabular_curves.png")


def fig_cp_tabular_hp() -> None:
    sweep_config = [
        ("alpha",  "Learning rate α",
         [0.05, 0.1, 0.2, 0.5], lambda v: f"{v:g}"),
        ("gamma",  "Discount γ",
         [0.9, 0.95, 0.99, 1.0], lambda v: f"{v:g}"),
        ("nbins",  "n_bins",
         [(1, 1, 6, 6), (3, 3, 6, 6), (3, 3, 8, 12), (5, 5, 12, 16)],
         lambda v: "x".join(map(str, v))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for panel_idx, (sweep_key, xlabel, values, value_label) in enumerate(sweep_config):
        ax = axes[panel_idx]
        rotation = 20 if sweep_key == "nbins" else 0
        for agent, color, prefix in [
            ("SARSA",      "C2", f"cartpole_sarsa_{sweep_key}_sweep_"),
            ("Q-Learning", "C3", f"cartpole_qlearning_{sweep_key}_sweep_"),
        ]:
            xs, means, errs = collect_sweep_points(
                values, lambda v, p=prefix: f"{p}{_fmt_value(v)}"
            )
            errorbar_sweep(ax, xs, means, errs, color=color, label=agent,
                           xlabels=[value_label(v) for v in xs],
                           rotation=rotation)
        ax.set_title(xlabel)
        ax.set_xlabel(xlabel)
        if panel_idx == 0:
            ax.legend(loc="lower right")
    axes[0].set_ylabel("Final eval return (mean step count)")
    fig.suptitle("CartPole: tabular 1-D hyperparameter sweeps "
                 "(5 seeds, 95% CI)", y=1.02)
    fig.tight_layout()
    save(fig, "07_cp_tabular_hp.png")


def fig_cp_dp_nbins() -> None:
    """VI / PI / trained-ε-sampling final eval across n_bins grids."""
    grids = [(1, 1, 6, 6), (3, 3, 6, 6), (3, 3, 8, 12), (5, 5, 12, 16)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(grids))
    bar_width = 0.28
    for offset, (label, prefix, color) in enumerate([
        ("VI (random sampling)", "cartpole_vi_nbins_sweep_",        "C0"),
        ("PI (random sampling)", "cartpole_pi_nbins_sweep_",        "C1"),
    ]):
        _, means, errs = collect_sweep_points(
            grids, lambda g, p=prefix: f"{p}{_fmt_value(g)}",
            skip_missing=False,
        )
        ax.bar(x + (offset - 1) * bar_width, means, yerr=errs,
               width=bar_width, color=color, capsize=3, label=label,
               edgecolor="black", linewidth=0.5)

    # Overlay best trained-ε (0.7) where available — skip missing grids.
    trained_means, trained_errs, trained_x = [], [], []
    for i, g in enumerate(grids):
        name = f"cartpole_vi_trained_eps_{_fmt_value(g)}_0p7"
        runs = _safe_load(name)
        if runs is None:
            continue
        m, e = final_eval_stats(runs)
        trained_means.append(m)
        trained_errs.append(e)
        trained_x.append(i + bar_width)
    if trained_means:
        ax.bar(trained_x, trained_means, yerr=trained_errs, width=bar_width,
               color="C4", capsize=3, edgecolor="black", linewidth=0.5,
               label="VI (ε=0.7 on trained SARSA)")

    ax.axhline(500, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label="max (truncation at 500)")
    ax.set_xticks(x)
    ax.set_xticklabels(["x".join(map(str, g)) for g in grids])
    ax.set_xlabel("n_bins (cart_pos × cart_vel × pole_angle × pole_vel)")
    ax.set_ylabel("Final eval return")
    ax.set_title("CartPole DP on estimated MDP: n_bins + sampling-policy "
                 "study (5 seeds, 95% CI)")
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    save(fig, "08_cp_dp_nbins.png")


def fig_cp_dp_budget_and_eps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: sampling budget at (3,3,8,12). Use a log-scaled numeric x-axis
    # (not integer positions) so the spacing between budgets is meaningful.
    ax = axes[0]
    budgets = [500, 2000, 5000, 10_000]
    _, means, errs = collect_sweep_points(
        budgets, lambda b: f"cartpole_vi_samples_sweep_{b}",
        skip_missing=False,
    )
    ax.errorbar(budgets, means, yerr=errs, marker="o", markersize=6,
                linewidth=1.5, capsize=4, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("Sampling rollout budget (episodes)")
    ax.set_ylabel("Final eval return")
    ax.set_title("VI eval return vs sampling budget @ (3,3,8,12)")
    ax.set_ylim(bottom=0)

    # Right: ε sweep on trained SARSA at both fine grids. Numeric ε x-axis
    # so we can overlay matching random-sampling baselines as axhlines.
    ax = axes[1]
    eps_vals = [0.1, 0.3, 0.5, 0.7]
    for grid, color, label in [
        ((3, 3, 8, 12), "C4", "(3,3,8,12)"),
        ((5, 5, 12, 16), "C5", "(5,5,12,16)"),
    ]:
        grid_str = _fmt_value(grid)
        _, means, errs = collect_sweep_points(
            eps_vals,
            lambda e, g=grid_str: f"cartpole_vi_trained_eps_{g}_{_fmt_value(e)}",
            skip_missing=False,
        )
        ax.errorbar(eps_vals, means, yerr=errs, marker="o", markersize=6,
                    linewidth=1.5, capsize=4, color=color,
                    label=f"trained-ε @ {label}")
        base = _safe_load(f"cartpole_vi_nbins_sweep_{grid_str}")
        if base is not None:
            ax.axhline(final_eval_scalar(base).mean(), color=color,
                       linestyle="--", linewidth=1, alpha=0.6,
                       label=f"random baseline @ {label}")
    ax.set_xlabel("Sampling ε (on top of trained SARSA policy)")
    ax.set_ylabel("Final eval return")
    ax.set_title("VI eval return vs sampling-policy ε")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(bottom=0)

    fig.suptitle("CartPole DP: sample-complexity and exploration-policy studies "
                 "(5 seeds, 95% CI)", y=1.03)
    fig.tight_layout()
    save(fig, "09_cp_dp_budget_and_eps.png")


def fig_cp_agent_comparison() -> None:
    """Compare all agents on CartPole at each agent's strongest config."""
    # For tabular, pick the best grid we found. For DP, pick the best overall.
    entries = [
        ("SARSA default",   "cartpole_sarsa_default",                "C2"),
        ("Q-Learning default", "cartpole_qlearning_default",         "C3"),
        ("SARSA best (3x3x6x6)",     "cartpole_sarsa_nbins_sweep_3x3x6x6",      "C2"),
        ("Q-Learning best (3x3x6x6)","cartpole_qlearning_nbins_sweep_3x3x6x6",  "C3"),
        ("VI random (1x1x6x6)",  "cartpole_vi_nbins_sweep_1x1x6x6",  "C0"),
        ("PI random (1x1x6x6)",  "cartpole_pi_nbins_sweep_1x1x6x6",  "C1"),
        ("VI random (3x3x6x6)",  "cartpole_vi_nbins_sweep_3x3x6x6",  "C0"),
        ("VI ε=0.7 (5x5x12x16)", "cartpole_vi_trained_eps_5x5x12x16_0p7", "C4"),
    ]
    names, colors, means, errs = [], [], [], []
    for label, exp_name, color in entries:
        runs = _safe_load(exp_name)
        if runs is None:
            continue
        m, e = final_eval_stats(runs)
        names.append(label)
        colors.append(color)
        means.append(m)
        errs.append(e)

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=errs, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.axhline(500, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label="truncation cap")
    ax.set_ylabel("Final eval return")
    ax.set_title("CartPole: agent comparison (5 seeds, 95% CI)")
    ax.legend(loc="upper left")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 5,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save(fig, "10_cp_agent_comparison.png")


# ---------------------------------------------------------------------------
# Phase 4 — DQN Rainbow-medium ablation
# ---------------------------------------------------------------------------

_DQN_VARIANTS = [
    ("baseline", "Vanilla DQN",        "#888888"),
    ("double",   "+ Double",           "C0"),
    ("dueling",  "+ Dueling",          "C1"),
    ("per",      "+ PER",              "C2"),
    ("nstep",    "+ N-step",           "C3"),
    ("rainbow",  "Rainbow (all 4)",    "C4"),
]


def fig_dqn_ablation_bars() -> None:
    """Final eval-return bar chart: baseline + each single component + full."""
    names, colors, means, errs = [], [], [], []
    for suffix, label, color in _DQN_VARIANTS:
        runs = _safe_load(f"dqn_ablation_{suffix}")
        if runs is None:
            continue
        m, e = final_eval_stats(runs)
        names.append(label)
        colors.append(color)
        means.append(m)
        errs.append(e)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=errs, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.axhline(500, color="black", linestyle="--", linewidth=1,
               alpha=0.4, label="truncation cap")

    # Reference baseline for visual comparison.
    if means:
        ax.axhline(means[0], color="#888888", linestyle=":", linewidth=1,
                   alpha=0.5, label="baseline level")

    ax.set_ylabel("Final eval return")
    ax.set_title("DQN Rainbow-medium ablation on CartPole-v1 "
                 "(5 seeds, 95% CI, 300 train episodes)")
    ax.legend(loc="upper left")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 10,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(bottom=0, top=550)
    fig.tight_layout()
    save(fig, "11_dqn_ablation_bars.png")


def fig_dqn_learning_curves() -> None:
    """Training-return learning curves across the six Rainbow variants.

    Shows per-episode mean return over seeds with a rolling-mean smoother
    so the noisy short-episode early phase doesn't dominate the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    window = 20
    any_drawn = False
    for suffix, label, color in _DQN_VARIANTS:
        runs = _safe_load(f"dqn_ablation_{suffix}")
        if runs is None:
            continue
        curves = [np.asarray(r.result.get("train_returns", []), dtype=float)
                  for r in runs]
        stacked, _ = pad_and_mean(curves)
        if stacked.size == 0:
            continue
        mean_curve = np.nanmean(stacked, axis=0)
        if len(mean_curve) >= window:
            smoothed = np.convolve(mean_curve, np.ones(window) / window,
                                   mode="valid")
            xs = np.arange(window - 1, len(mean_curve))
        else:
            smoothed, xs = mean_curve, np.arange(len(mean_curve))

        # Seed-variability band using inter-quartile range on the raw (not
        # smoothed) curves, then lightly smoothed to match the line.
        q25 = np.nanpercentile(stacked, 25, axis=0)
        q75 = np.nanpercentile(stacked, 75, axis=0)
        if len(q25) >= window:
            q25 = np.convolve(q25, np.ones(window) / window, mode="valid")
            q75 = np.convolve(q75, np.ones(window) / window, mode="valid")

        ax.plot(xs, smoothed, color=color, linewidth=1.8, label=label)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.12)
        any_drawn = True

    if not any_drawn:
        plt.close(fig)
        return

    ax.axhline(500, color="black", linestyle="--", linewidth=1, alpha=0.4,
               label="truncation cap")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Episode return (rolling mean, w={window})")
    ax.set_title("DQN Rainbow-medium: learning curves by variant "
                 "(shaded = IQR across 5 seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "12_dqn_learning_curves.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

FIGURES: dict[str, callable] = {
    "bj_dp_convergence":    fig_bj_dp_convergence,
    "bj_policy_heatmap":    fig_bj_policy_heatmap,
    "bj_tabular_curves":    fig_bj_tabular_curves,
    "bj_hp_sensitivity":    fig_bj_hp_sensitivity,
    "bj_agent_comparison":  fig_bj_agent_comparison,
    "cp_tabular_curves":    fig_cp_tabular_curves,
    "cp_tabular_hp":        fig_cp_tabular_hp,
    "cp_dp_nbins":          fig_cp_dp_nbins,
    "cp_dp_budget_and_eps": fig_cp_dp_budget_and_eps,
    "cp_agent_comparison":  fig_cp_agent_comparison,
    "dqn_ablation_bars":    fig_dqn_ablation_bars,
    "dqn_learning_curves":  fig_dqn_learning_curves,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*",
                        help="Only build the listed figure keys (see --list).")
    parser.add_argument("--list", action="store_true",
                        help="List available figure keys and exit.")
    args = parser.parse_args()

    if args.list:
        for k in FIGURES:
            print(k)
        return

    set_style()
    keys = args.only or list(FIGURES.keys())
    for k in keys:
        if k not in FIGURES:
            print(f"[warn] unknown figure key {k!r}, skipping")
            continue
        print(f"[{k}]")
        FIGURES[k]()


if __name__ == "__main__":
    main()
