from inspect_ai import Task, task
from inspect_ai.scorer import choice

from simple_baseline.datasets.quality_dataset import debater_quality_dataset, judge_quality_dataset
from simple_baseline.solvers.forgiving_multiple_choice import forgiving_multiple_choice
from simple_baseline.tasks.templates import MULTIPLE_CHOICE_TEMPLATE


@task
def quality_debater_simple_baseline() -> Task:
    return Task(
        dataset=debater_quality_dataset,
        solver=[
            forgiving_multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE)
        ],
        scorer=choice()
    )


@task
def quality_judge_simple_baseline() -> Task:
    return Task(
        dataset=judge_quality_dataset,
        solver=[
            forgiving_multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE)
        ],
        scorer=choice()
    )


