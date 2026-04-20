"""Schema + registry + sweep generator for experiment specs.

This is the machinery layer. The actual experiment declarations live in
topic-focused submodules (`blackjack_dp`, `blackjack_tabular`,
`cartpole_tabular`, `cartpole_dp`, `dqn_ablation`) so each is short
enough to hold in your head.

Naming convention for registered experiments:
    <env>_<agent>_<variant>              # single-point
    <env>_<agent>_<hp>_sweep_<value>     # shared-prefix sweep variants

Sweep variants live under `results/<prefix>/<frag>/`; single-point specs
use `results/<name>/`. Both paths are driven by
`ExperimentSpec.results_path_parts`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any


# --- Spec dataclasses ------------------------------------------------------

@dataclass(frozen=True)
class EnvSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """env + agent + training budget + seeds.

    `results_path_parts` drives the on-disk location: empty -> results/<name>/,
    else results/<parts...>/. Sweep variants use ("<prefix>", "<frag>") so
    every variant of a sweep nests under a shared parent.
    """
    name: str
    env: EnvSpec
    agent: AgentSpec
    n_episodes: int = 5000
    eval_episodes: int = 100
    seeds: tuple[int, ...] = tuple(range(10))
    gamma: float = 0.99
    description: str = ""
    results_path_parts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Registry --------------------------------------------------------------

EXPERIMENTS: dict[str, ExperimentSpec] = {}


def register(exp: ExperimentSpec) -> ExperimentSpec:
    if exp.name in EXPERIMENTS:
        raise ValueError(f"Experiment {exp.name!r} already registered")
    EXPERIMENTS[exp.name] = exp
    return exp


def get(name: str) -> ExperimentSpec:
    if name not in EXPERIMENTS:
        raise KeyError(
            f"No experiment named {name!r}. "
            f"Registered: {sorted(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]


def list_experiments(name_prefix: str | None = None) -> list[str]:
    """Sorted list of registered experiment names, optionally prefix-filtered."""
    out = [
        name for name in EXPERIMENTS
        if name_prefix is None or name.startswith(name_prefix)
    ]
    return sorted(out)


# --- Sweep generator -------------------------------------------------------

def register_sweep(
    *,
    name_prefix: str,
    base: ExperimentSpec,
    sweep_path: str,
    values: list,
    description: str | None = None,
) -> list[ExperimentSpec]:
    """Register one variant per sweep value. Each variant gets a unique name
    `{name_prefix}_{value_str}` and lives under results/<prefix>/<frag>/."""
    if description is None:
        description = (
            f"Sweep over {sweep_path} in {list(values)!r}. "
            f"Base: env={base.env.name}, agent={base.agent.name}."
        )

    variants: list[ExperimentSpec] = []
    for value in values:
        value_frag = _fmt_value(value)
        variant_name = f"{name_prefix}_{value_frag}"
        variant = _override_at_path(
            base,
            sweep_path=sweep_path,
            value=value,
            new_name=variant_name,
            new_description=description,
            new_results_path_parts=(name_prefix, value_frag),
        )
        register(variant)
        variants.append(variant)
    return variants


def _fmt_value(v: Any) -> str:
    """Safe identifier fragment for a sweep value.
    0.95 -> "0p95", 1e-09 -> "1e-09", (3,3,8,12) -> "3x3x8x12"."""
    if isinstance(v, float):
        s = repr(v)
        if "e" in s or "E" in s:
            return s
        return s.replace(".", "p").replace("-", "m")
    if isinstance(v, (tuple, list)):
        return "x".join(_fmt_value(x) for x in v)
    return str(v)


def override_at_path(spec: ExperimentSpec, path: str, value: Any) -> ExperimentSpec:
    """Return a copy of `spec` with the dotted `path` replaced by `value`.
    Paths: top-level fields (gamma, n_episodes, eval_episodes, seeds),
    agent.hyperparams.<key>, or env.kwargs.<key>."""
    parts = path.split(".")

    if len(parts) == 3 and parts[0] == "agent" and parts[1] == "hyperparams":
        new_hp = {**spec.agent.hyperparams, parts[2]: value}
        return dataclasses.replace(
            spec, agent=dataclasses.replace(spec.agent, hyperparams=new_hp)
        )

    if len(parts) == 3 and parts[0] == "env" and parts[1] == "kwargs":
        new_kw = {**spec.env.kwargs, parts[2]: value}
        return dataclasses.replace(
            spec, env=dataclasses.replace(spec.env, kwargs=new_kw)
        )

    if len(parts) == 1 and parts[0] in {"gamma", "n_episodes", "eval_episodes", "seeds"}:
        return dataclasses.replace(spec, **{parts[0]: value})

    raise ValueError(
        f"Unsupported override path {path!r}. Use 'gamma', 'n_episodes', "
        f"'eval_episodes', 'seeds', 'agent.hyperparams.<key>', "
        f"or 'env.kwargs.<key>'."
    )


def _override_at_path(
    base: ExperimentSpec,
    *,
    sweep_path: str,
    value: Any,
    new_name: str,
    new_description: str,
    new_results_path_parts: tuple[str, ...],
) -> ExperimentSpec:
    """Sweep-variant override: apply `value` at `sweep_path` and overwrite
    identity metadata (name, description, results_path_parts).
    """
    spec = override_at_path(base, sweep_path, value)
    return dataclasses.replace(
        spec,
        name=new_name,
        description=new_description,
        results_path_parts=new_results_path_parts,
    )
