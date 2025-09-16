# %%

from inspect_ai import eval

from simple_baseline.evals.config import OUTPUT_CONFIG, DEBATER_LOJBAN_EVAL_CONFIG
from simple_baseline.tasks.lojban_baseline import lojban_simple_baseline

eval(
    lojban_simple_baseline,
    **DEBATER_LOJBAN_EVAL_CONFIG,
    **OUTPUT_CONFIG
)
