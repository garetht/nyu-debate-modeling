# %%

from inspect_ai import eval

from simple_baseline.evals.config import OUTPUT_CONFIG, DEBATER_QUALITY_EVAL_CONFIG
from simple_baseline.tasks.quality_baseline import quality_debater_simple_baseline

eval(
    quality_debater_simple_baseline,
    **DEBATER_QUALITY_EVAL_CONFIG,
    **OUTPUT_CONFIG
)
