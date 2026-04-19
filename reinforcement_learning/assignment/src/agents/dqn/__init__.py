"""DQN + Rainbow-ablation components.

Component toggles supported (medium-scope Rainbow):
    - double   : Double DQN action/value decoupling.
    - dueling  : Dueling network head (V + A streams).
    - per      : Prioritized experience replay (proportional, sum-tree).
    - nstep    : N-step TD targets.

The full "Rainbow" variant just enables all four. Components are exposed
as plain hyperparameter flags in ``AgentSpec.hyperparams``; the factory in
``src/agents/__init__.py`` dispatches on the agent name ``"dqn"`` and
builds a single ``DQNAgent`` parameterised by those flags.
"""

from src.agents.dqn.agent import DQNAgent

__all__ = ["DQNAgent"]
