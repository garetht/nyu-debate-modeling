# %%

from inspect_ai import eval

from simple_baseline.evals.config import OUTPUT_CONFIG, JUDGE_EVAL_CONFIG
from simple_baseline.tasks.lojban_baseline import lojban_judge_simple_baseline

eval(
    lojban_judge_simple_baseline(),
    **JUDGE_EVAL_CONFIG,
    **OUTPUT_CONFIG
)
