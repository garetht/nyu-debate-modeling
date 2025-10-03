import re
from inspect_ai.solver import TaskState, Generate,  solver, Solver
from inspect_ai.solver._multiple_choice import valid_template, prompt, set_choices_based_on_generated_response

def parse_answers(state: TaskState) -> re.Match[str] | None:
    """
    Convenience function for extracting answers from the state output.

    The generated response must be in the format 'ANSWER: <answers>',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    However, if the answer isn't in the expected format the model has
    failed in the task so we'll ultimately just mark it as incorrect
    """
    # First check whether the string strictly ends with the expected answer
    # In this case, we're looking for a single line which contains the expected
    # ANSWER: B,C string with only whitespace after it
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z ,]+)\s*(?:$|\n)",
        state.output.completion,
        flags=re.MULTILINE,
    )

    # If we couldn't match the strict version, we can try the less strict
    # version for backward compatibility
    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z ,]+)(?:\W|\n|$)", state.output.completion
        )

    if match is None:
        return re.match(
            r"(?i)\s*([A-Za-z ,]+)\)", state.output.completion
        )
    else:
        return match

@solver
def forgiving_multiple_choice(
        *,
        template: str
) -> Solver:
    if template and not valid_template(template):
        raise ValueError(
            "The template must contain '{question}' and '{choices}' placeholders for string substitution."
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("The multiple_choice solver requires samples with choices")

        state.user_prompt.text = prompt(
            question=state.user_prompt.text,
            choices=state.choices,
            template=str(template),
        )

        state = await generate(state)

        answers = parse_answers(state)
        if answers and answers.group(1):
            # If we've found answers, update the state appropriately
            set_choices_based_on_generated_response(
                state=state, answers=answers.group(1)
            )

        return state

    return solve
