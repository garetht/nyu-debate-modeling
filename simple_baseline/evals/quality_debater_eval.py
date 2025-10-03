# %%

from inspect_ai import eval

from simple_baseline.evals.config import OUTPUT_CONFIG, create_debater_eval_config
from simple_baseline.tasks.quality_baseline import quality_debater_simple_baseline

eval(
    quality_debater_simple_baseline,
    **create_debater_eval_config(),
    **OUTPUT_CONFIG
)
