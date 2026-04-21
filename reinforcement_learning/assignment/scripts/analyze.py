"""One-shot number extraction for ANALYSIS.md.

Prints headline stats per hypothesis. Intended to be run once after the
campaign finishes and piped into ANALYSIS.md for report drafting.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.experiments.runner import load_runs


def _mean_ci(xs):
    a = np.asarray(xs, dtype=float)
    n = max(len(a), 2)
    se = a.std(ddof=1) / np.sqrt(n)
    return float(a.mean()), float(1.96 * se), int(n)


def eval_return(name):
    runs = load_runs(name)
    return _mean_ci([r.summary["eval_return_mean"] for r in runs])


def train_return(name):
    runs = load_runs(name)
    return _mean_ci([r.summary["train_return_mean"] for r in runs])


def wall(name):
    runs = load_runs(name)
    return _mean_ci([r.summary["wall_clock_seconds"] for r in runs])


def history_field(name, path):
    """Return per-seed list of a history scalar/array."""
    runs = load_runs(name)
    out = []
    for r in runs:
        cur = r.result["history"]
        for key in path:
            cur = cur[key]
        out.append(cur)
    return out


def header(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def fmt(m, hw, unit=""):
    return f"{m:.3f} ± {hw:.3f}{unit}"


def main():
    # -----------------------------------------------------------------
    # H1 — Blackjack VI vs PI
    # -----------------------------------------------------------------
    header("H1 — Blackjack VI vs PI (analytical MDP)")

    # Claim 1: matched eval returns
    print("\n[Claim 1] Policy equivalence (eval return over 20k eval episodes)")
    for name in ("blackjack_vi_default", "blackjack_pi_default"):
        m, hw, n = eval_return(name)
        print(f"  {name:40s}  {fmt(m, hw)}  (n={n})")

    # Gamma sweep: outer iters, total sweeps, eval return
    print("\n[Claim 2+3] γ sweep (θ=1e-9) — outer iters and total Bellman backups")
    gammas = ["0p5", "0p8", "0p9", "0p95", "1p0"]
    print(f"  {'γ':>6}  {'VI sweeps':>15}  {'PI outer':>12}  {'PI PE total':>15}  "
          f"{'VI eval':>15}  {'PI eval':>15}")
    for g in gammas:
        vi_hist = history_field(f"blackjack_vi_gamma_sweep_{g}", ("sweep_deltas",))
        vi_sweeps = [len(h) for h in vi_hist]
        m_vi, hw_vi, _ = _mean_ci(vi_sweeps)

        pi_outer_h = history_field(f"blackjack_pi_gamma_sweep_{g}", ("eval_sweeps_per_outer",))
        pi_outer = [len(h) for h in pi_outer_h]
        m_po, hw_po, _ = _mean_ci(pi_outer)
        pi_total = [sum(h) for h in pi_outer_h]
        m_pt, hw_pt, _ = _mean_ci(pi_total)

        m_ve, hw_ve, _ = eval_return(f"blackjack_vi_gamma_sweep_{g}")
        m_pe, hw_pe, _ = eval_return(f"blackjack_pi_gamma_sweep_{g}")

        print(f"  {g:>6}  {fmt(m_vi, hw_vi):>15}  {fmt(m_po, hw_po):>12}  "
              f"{fmt(m_pt, hw_pt):>15}  {fmt(m_ve, hw_ve):>15}  {fmt(m_pe, hw_pe):>15}")

    # Theta sweep: total sweeps for VI + PI at gamma=1
    print("\n[Claim 3] θ sweep (γ=1.0) — sensitivity of VI sweeps to θ")
    thetas = ["0p1", "0p001", "1e-05", "1e-07", "1e-09"]
    print(f"  {'θ':>6}  {'VI sweeps':>15}  {'PI PE total':>15}")
    for t in thetas:
        try:
            vi_hist = history_field(f"blackjack_vi_theta_sweep_{t}", ("sweep_deltas",))
            vi_sweeps = [len(h) for h in vi_hist]
            m_vi, hw_vi, _ = _mean_ci(vi_sweeps)
        except Exception as e:
            m_vi, hw_vi = np.nan, 0.0
        try:
            pi_outer_h = history_field(f"blackjack_pi_theta_sweep_{t}", ("eval_sweeps_per_outer",))
            pi_total = [sum(h) for h in pi_outer_h]
            m_pt, hw_pt, _ = _mean_ci(pi_total)
        except Exception:
            m_pt, hw_pt = np.nan, 0.0
        print(f"  {t:>6}  {fmt(m_vi, hw_vi):>15}  {fmt(m_pt, hw_pt):>15}")

    # Wall clock defaults
    print("\n[Wall clock] defaults")
    for name in ("blackjack_vi_default", "blackjack_pi_default"):
        m, hw, _ = wall(name)
        print(f"  {name:40s}  {fmt(m, hw, ' s')}")

    # -----------------------------------------------------------------
    # H2 — Sampling-policy dependence on CartPole DP
    # -----------------------------------------------------------------
    header("H2 — CartPole DP: random vs trained-ε sampling")

    grids = ["1x1x6x6", "3x3x6x6", "3x3x8x12", "5x5x12x16"]
    print(f"\n[Random sampling, VI] eval return @ each grid")
    for g in grids:
        try:
            m, hw, _ = eval_return(f"cartpole_vi_nbins_sweep_{g}")
            print(f"  VI  {g:>12}  {fmt(m, hw)}")
        except Exception as e:
            print(f"  VI  {g:>12}  MISSING ({e})")
    print(f"\n[Random sampling, PI] eval return @ each grid")
    for g in grids:
        try:
            m, hw, _ = eval_return(f"cartpole_pi_nbins_sweep_{g}")
            print(f"  PI  {g:>12}  {fmt(m, hw)}")
        except Exception:
            print(f"  PI  {g:>12}  MISSING")

    print(f"\n[Sampling budget sweep @ 3x3x8x12]")
    for b in ["500", "5000", "10000"]:
        try:
            m, hw, _ = eval_return(f"cartpole_vi_samples_sweep_{b}")
            print(f"  budget {b:>6}  {fmt(m, hw)}")
        except Exception:
            print(f"  budget {b:>6}  MISSING")

    print(f"\n[Trained-ε sampling — 3x3x8x12]")
    for e in ["0p1", "0p3", "0p5", "0p7"]:
        try:
            m, hw, _ = eval_return(f"cartpole_vi_trained_eps_3x3x8x12_{e}")
            print(f"  ε={e:>4}  {fmt(m, hw)}")
        except Exception:
            print(f"  ε={e:>4}  MISSING")

    print(f"\n[Trained-ε sampling — 5x5x12x16]")
    for e in ["0p1", "0p3", "0p5", "0p7"]:
        try:
            m, hw, _ = eval_return(f"cartpole_vi_trained_eps_5x5x12x16_{e}")
            print(f"  ε={e:>4}  {fmt(m, hw)}")
        except Exception:
            print(f"  ε={e:>4}  MISSING")

    # -----------------------------------------------------------------
    # H3 — SARSA vs Q-Learning on both MDPs
    # -----------------------------------------------------------------
    header("H3 — SARSA vs Q-Learning")

    print("\n[Blackjack] defaults")
    for name in ("blackjack_sarsa_default", "blackjack_qlearning_default"):
        m, hw, _ = eval_return(name)
        wc, _, _ = wall(name)
        print(f"  {name:40s}  eval={fmt(m, hw)}  wall={wc:.2f}s")

    print("\n[Blackjack] α sweep")
    for alg in ("sarsa", "qlearning"):
        print(f"  {alg}:")
        for a in ["0p01", "0p05", "0p1", "0p2"]:
            try:
                m, hw, _ = eval_return(f"blackjack_{alg}_alpha_sweep_{a}")
                print(f"    α={a:>5}  {fmt(m, hw)}")
            except Exception:
                pass

    print("\n[Blackjack] ε-decay-budget sweep (n episodes over which ε anneals)")
    for alg in ("sarsa", "qlearning"):
        print(f"  {alg}:")
        for d in ["10000", "50000", "100000", "200000"]:
            try:
                m, hw, _ = eval_return(f"blackjack_{alg}_eps_decay_sweep_{d}")
                print(f"    decay_eps={d:>7}  {fmt(m, hw)}")
            except Exception:
                pass

    print("\n[CartPole] defaults")
    for name in ("cartpole_sarsa_default", "cartpole_qlearning_default"):
        m, hw, _ = eval_return(name)
        wc, _, _ = wall(name)
        print(f"  {name:40s}  eval={fmt(m, hw)}  wall={wc:.2f}s")

    print("\n[CartPole] α sweep")
    for alg in ("sarsa", "qlearning"):
        print(f"  {alg}:")
        for a in ["0p05", "0p1", "0p2", "0p5"]:
            try:
                m, hw, _ = eval_return(f"cartpole_{alg}_alpha_sweep_{a}")
                print(f"    α={a:>5}  {fmt(m, hw)}")
            except Exception:
                pass

    print("\n[CartPole] ε-decay sweep — NOT RUN (only blackjack has this)")

    # -----------------------------------------------------------------
    # H4 — CartPole discretization + γ
    # -----------------------------------------------------------------
    header("H4 — CartPole discretization + γ")

    print("\n[nbins sweep, SARSA]")
    for g in grids:
        try:
            m, hw, _ = eval_return(f"cartpole_sarsa_nbins_sweep_{g}")
            print(f"  SARSA  {g:>12}  {fmt(m, hw)}")
        except Exception:
            print(f"  SARSA  {g:>12}  MISSING")

    print("\n[nbins sweep, Q-Learning]")
    for g in grids:
        try:
            m, hw, _ = eval_return(f"cartpole_qlearning_nbins_sweep_{g}")
            print(f"  Q-L    {g:>12}  {fmt(m, hw)}")
        except Exception:
            print(f"  Q-L    {g:>12}  MISSING")

    print("\n[γ sweep, SARSA]")
    for gm in ["0p9", "0p95", "0p99", "1p0"]:
        try:
            m, hw, _ = eval_return(f"cartpole_sarsa_gamma_sweep_{gm}")
            print(f"  SARSA  γ={gm:>5}  {fmt(m, hw)}")
        except Exception:
            pass
    print("\n[γ sweep, Q-Learning]")
    for gm in ["0p9", "0p95", "0p99", "1p0"]:
        try:
            m, hw, _ = eval_return(f"cartpole_qlearning_gamma_sweep_{gm}")
            print(f"  Q-L    γ={gm:>5}  {fmt(m, hw)}")
        except Exception:
            pass

    # -----------------------------------------------------------------
    # H5 — DQN ablation
    # -----------------------------------------------------------------
    header("H5 — DQN Rainbow-medium ablation")
    variants = ["baseline", "double", "dueling", "per", "nstep", "rainbow"]
    final_train, final_eval = {}, {}
    for v in variants:
        try:
            m_e, hw_e, _ = eval_return(f"dqn_ablation_{v}")
            m_t, hw_t, _ = train_return(f"dqn_ablation_{v}")
            wc, _, _ = wall(f"dqn_ablation_{v}")
            print(f"  {v:10s}  eval={fmt(m_e, hw_e):>15}  train={fmt(m_t, hw_t):>15}  wall={wc:.1f}s")
            final_train[v] = (m_t, hw_t)
            final_eval[v] = (m_e, hw_e)
        except Exception as ex:
            print(f"  {v:10s}  MISSING ({ex})")

    # Marginal improvement deltas vs baseline
    if "baseline" in final_eval:
        base_m, base_hw = final_eval["baseline"]
        print(f"\n[Marginal Δ vs baseline, eval]")
        for v in ["double", "dueling", "per", "nstep", "rainbow"]:
            if v in final_eval:
                m, hw = final_eval[v]
                print(f"  {v:10s}  Δ={m - base_m:+.1f}  (base={base_m:.1f}, v={m:.1f})")


if __name__ == "__main__":
    main()
