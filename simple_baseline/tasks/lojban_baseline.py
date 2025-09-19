from inspect_ai import Task, task
from inspect_ai.scorer import choice

from simple_baseline.datasets.lojban_dataset import debater_lojban_dataset, judge_lojban_dataset
from simple_baseline.solvers.forgiving_multiple_choice import forgiving_multiple_choice
from simple_baseline.tasks.templates import MULTIPLE_CHOICE_TEMPLATE


@task
def lojban_debater_simple_baseline():
    return Task(
        dataset=debater_lojban_dataset,
        solver=[
            forgiving_multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE)
        ],
        scorer=choice()
    )


@task
def lojban_judge_simple_baseline():
    return Task(
        dataset=judge_lojban_dataset,
        solver=[
            forgiving_multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE)
        ],
        scorer=choice()
    )


