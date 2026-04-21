"""Final audit script: verify every numerical claim in rl_report_v3.tex against results/."""
import pickle, glob
import numpy as np
from pathlib import Path

RES = Path("results")

def load_dir(pattern):
    files = sorted(glob.glob(str(RES / pattern)))
    return [pickle.load(open(f, "rb")) for f in files]

def stat(values, fmt="{:.4f}"):
    v = np.array(values)
    n = len(v)
    mean = np.mean(v)
    ci = 1.96 * np.std(v, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, ci, n

def eval_stats(results_list, mode="last"):
    """mode='last' (CartPole per-episode eval history),
    'full' (BJ flat per-episode eval list — take whole-mean)."""
    finals = []
    for r in results_list:
        er = np.array(r["eval_returns"])
        if mode == "full":
            finals.append(np.mean(er))
        else:
            finals.append(er[-1])
    return stat(finals)

def fmt(t, digits=2):
    m, ci, n = t
    return f"{m:.{digits}f} ± {ci:.{digits}f}  (n={n})"

print("=" * 70)
print("BLACKJACK DP")
print("=" * 70)

# Default (γ=1.0, θ=1e-9)
for agent in ["vi", "pi"]:
    r = load_dir(f"blackjack_{agent}_default/seed_*/result.pkl")
    print(f"  {agent.upper()} default (γ=1, θ=1e-9): {fmt(eval_stats(r, 'full'), 4)}")
    # sweeps / outer iters
    if agent == "vi":
        sweeps = [len(x["history"]["sweep_deltas"]) for x in r]
        print(f"    VI sweeps: mean={np.mean(sweeps):.1f}, range=[{min(sweeps)},{max(sweeps)}]")
    else:
        outers = [x["history"]["outer_iters"] for x in r]
        pe_sweeps = [sum(x["history"]["eval_sweeps_per_outer"]) for x in r]
        print(f"    PI outer iters K: mean={np.mean(outers):.1f}, range=[{min(outers)},{max(outers)}]")
        print(f"    PI total PE sweeps N_PE: mean={np.mean(pe_sweeps):.1f}, range=[{min(pe_sweeps)},{max(pe_sweeps)}]")
        print(f"    PI eval_sweeps_per_outer seed0: {r[0]['history']['eval_sweeps_per_outer']}")

# γ sweep
print("\n  γ sweep (θ=1e-9):")
for gamma in ["0.5", "0.8", "0.9", "0.95", "1.0"]:
    for agent in ["vi", "pi"]:
        r = load_dir(f"blackjack_{agent}_gamma_sweep/{gamma.replace('.','p')}/seed_*/result.pkl")
        if not r:
            continue
        eval_ = eval_stats(r, "full")
        if agent == "vi":
            sweeps = [len(x["history"]["sweep_deltas"]) for x in r]
            print(f"    γ={gamma} {agent}: {fmt(eval_, 4)}, N_VI={np.mean(sweeps):.0f}")
        else:
            outers = [x["history"]["outer_iters"] for x in r]
            pe = [sum(x["history"]["eval_sweeps_per_outer"]) for x in r]
            print(f"    γ={gamma} {agent}: {fmt(eval_, 4)}, K={np.mean(outers):.0f}, N_PE={np.mean(pe):.0f}")

# θ sweep
print("\n  θ sweep (γ=1.0):")
theta_dirs = [("1e-1", "0p1"), ("1e-3", "0p001"), ("1e-5", "1e-05"), ("1e-7", "1e-07"), ("1e-9", "1e-09")]
for theta, theta_dir in theta_dirs:
    for agent in ["vi", "pi"]:
        r = load_dir(f"blackjack_{agent}_theta_sweep/{theta_dir}/seed_*/result.pkl")
        if not r:
            continue
        eval_ = eval_stats(r, "full")
        if agent == "vi":
            sweeps = [len(x["history"]["sweep_deltas"]) for x in r]
            print(f"    θ={theta} {agent}: {fmt(eval_, 4)}, N_VI={np.mean(sweeps):.0f}")
        else:
            outers = [x["history"]["outer_iters"] for x in r]
            pe = [sum(x["history"]["eval_sweeps_per_outer"]) for x in r]
            print(f"    θ={theta} {agent}: {fmt(eval_, 4)}, K={np.mean(outers):.0f}, N_PE={np.mean(pe):.0f}")


print("\n" + "=" * 70)
print("BLACKJACK TABULAR")
print("=" * 70)

for agent in ["sarsa", "qlearning"]:
    r = load_dir(f"blackjack_{agent}_default/seed_*/result.pkl")
    # Tabular eval_returns: check structure
    er0 = np.array(r[0]["eval_returns"])
    print(f"  {agent} default eval_returns shape: {er0.shape}")

# α sweep
print("\n  α sweep:")
for agent in ["sarsa", "qlearning"]:
    print(f"    {agent}:")
    for alpha in ["0p01", "0p05", "0p1", "0p2"]:
        r = load_dir(f"blackjack_{agent}_alpha_sweep/{alpha}/seed_*/result.pkl")
        if not r:
            continue
        eval_ = eval_stats(r, "full")
        print(f"      α={alpha.replace('p','.')}: {fmt(eval_, 4)}")

# ε-decay sweep
print("\n  ε-decay sweep:")
for agent in ["sarsa", "qlearning"]:
    print(f"    {agent}:")
    for eps_d in sorted(Path(RES / f"blackjack_{agent}_eps_decay_sweep").iterdir()):
        r = load_dir(f"blackjack_{agent}_eps_decay_sweep/{eps_d.name}/seed_*/result.pkl")
        if not r:
            continue
        eval_ = eval_stats(r, "full")
        print(f"      eps_decay={eps_d.name}: {fmt(eval_, 4)}")


print("\n" + "=" * 70)
print("CARTPOLE DP")
print("=" * 70)

# Default
for agent in ["vi", "pi"]:
    r = load_dir(f"cartpole_{agent}_default/seed_*/result.pkl")
    print(f"  {agent.upper()} default (grid=(3,3,8,12), 5k random): {fmt(eval_stats(r), 2)}")

# Samples sweep (VI at default grid)
print("\n  Rollout-budget sweep (VI, random, (3,3,8,12)):")
for b in sorted(Path(RES / "cartpole_vi_samples_sweep").iterdir()):
    r = load_dir(f"cartpole_vi_samples_sweep/{b.name}/seed_*/result.pkl")
    print(f"    budget={b.name}: {fmt(eval_stats(r), 2)}")

# nbins sweep
print("\n  nbins sweep:")
for agent in ["vi", "pi"]:
    print(f"    {agent}:")
    for g in sorted(Path(RES / f"cartpole_{agent}_nbins_sweep").iterdir()):
        r = load_dir(f"cartpole_{agent}_nbins_sweep/{g.name}/seed_*/result.pkl")
        print(f"      grid={g.name}: {fmt(eval_stats(r), 2)}")

# ε-sampling sweep (if exists)
for d in sorted(RES.iterdir()):
    if "sampling" in d.name or "eps_sampling" in d.name:
        print(f"\n  {d.name}:")
        for sub in sorted(d.iterdir()):
            r = load_dir(f"{d.name}/{sub.name}/seed_*/result.pkl")
            if r:
                print(f"    {sub.name}: {fmt(eval_stats(r), 2)}")


print("\n" + "=" * 70)
print("CARTPOLE TABULAR")
print("=" * 70)

for agent in ["sarsa", "qlearning"]:
    r = load_dir(f"cartpole_{agent}_default/seed_*/result.pkl")
    print(f"  {agent.upper()} default: {fmt(eval_stats(r), 2)}")

for sweep in ["alpha_sweep", "gamma_sweep", "nbins_sweep"]:
    print(f"\n  {sweep}:")
    for agent in ["sarsa", "qlearning"]:
        print(f"    {agent}:")
        d = RES / f"cartpole_{agent}_{sweep}"
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            r = load_dir(f"cartpole_{agent}_{sweep}/{sub.name}/seed_*/result.pkl")
            if r:
                print(f"      {sub.name}: {fmt(eval_stats(r), 2)}")


print("\n" + "=" * 70)
print("DQN ABLATION")
print("=" * 70)

for variant in ["baseline", "double", "dueling", "per", "nstep", "rainbow"]:
    r = load_dir(f"dqn_ablation/{variant}/seed_*/result.pkl")
    if r:
        print(f"  {variant:10s}: {fmt(eval_stats(r), 2)}")

