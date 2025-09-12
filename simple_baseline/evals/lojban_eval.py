# %%

from inspect_ai import eval

from simple_baseline.evals.config import CONFIG

eval(
    "./simple_baseline/tasks/lojban_baseline.py",
    **CONFIG
)
