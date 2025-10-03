# %%

from inspect_ai import eval

from simple_baseline.evals.config import OUTPUT_CONFIG, JUDGE_EVAL_CONFIG
from simple_baseline.tasks.quality_baseline import quality_judge_simple_baseline

eval(
    quality_judge_simple_baseline,
    **JUDGE_EVAL_CONFIG,
    **OUTPUT_CONFIG
)
