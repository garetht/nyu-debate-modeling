from inspect_ai import Task, task
from inspect_ai.scorer import choice

from simple_baseline.datasets.lojban_dataset import lojban_dataset
from simple_baseline.solvers.forgiving_multiple_choice import forgiving_multiple_choice
from simple_baseline.tasks.templates import MULTIPLE_CHOICE_TEMPLATE


@task
def lojban_simple_baseline():
    return Task(
        dataset=lojban_dataset,
        solver=[
            forgiving_multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE)
        ],
        scorer=choice()
    )


