"""
output_paths.py

Every pipeline script writes ONLY into its own subfolder under
outputs/, named after the script (e.g. outputs/06_spatial_feature_engineering/).
This keeps each phase's outputs cleanly separated -- easy to see which
script produced what, easy to clear and re-run one step without
touching another's files, and necessary once a single step (like
10_ml_modeling.py) produces multiple variants (one per forecast
horizon) that each need their own space.

A script that needs a file from an EARLIER step imports step_dir() and
points at that step's folder explicitly -- there is no flat, shared
outputs/ namespace anymore. The dependency is explicit in the code,
not implicit in a shared folder.
"""
from pathlib import Path

OUTPUTS_ROOT = Path("outputs")


def step_dir(step_name: str) -> Path:
    """Returns outputs/<step_name>/, creating it (and outputs/ itself)
    if it doesn't exist yet."""
    d = OUTPUTS_ROOT / step_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def horizon_subdir(step_name: str, horizon_months: int) -> Path:
    """Returns outputs/<step_name>/horizon_<N>month/ -- used by the
    multi-horizon modelling and risk-mapping scripts, where each
    forecast horizon (1, 3, 6 months) gets its own space within that
    step's folder rather than overwriting a shared set of files."""
    d = step_dir(step_name) / f"horizon_{horizon_months}month"
    d.mkdir(parents=True, exist_ok=True)
    return d
