# %%
import random
from collections import defaultdict, Counter
from typing import Callable

from inspect_ai.dataset import json_dataset, Sample

from simple_baseline import locations
from simple_baseline.data_models.quality import QualityTranscript, Question

randomness = random.Random(189239411823)

def get_question_ids() -> dict[str, set[str]]:
    questions_file = locations.PROJECT_ROOT / "all_debate_questions.txt"
    question_ids = defaultdict(set)
    with open(questions_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                question_id, question = line.split("_")
                question_ids[question_id.strip()].add(question.strip())
    return question_ids


def assign_random_values(data: dict[str, set[str]], total: int = 2502) -> dict[str, dict[str, int]]:
    """
    Distributes a total number uniformly across all leaves in a nested structure.

    Args:
        data: Dictionary mapping keys to sets of leaf values
        total: Total number to distribute (default: 2502)

    Returns:
        Dictionary mapping each key to a dictionary of leaf -> count assignments
    """
    # Collect all unique leaves
    all_leaves = set()
    for leaf_set in data.values():
        all_leaves.update(leaf_set)

    total_leaves = len(all_leaves)

    if total_leaves == 0:
        return {key: {} for key in data.keys()}

    # Calculate base assignment and remainder
    base_assignment = total // total_leaves
    remainder = total % total_leaves

    # Convert leaves to list for random selection
    leaves_list = list(all_leaves)

    # Randomly select which leaves get an extra +1
    bonus_leaves = set(randomness.sample(leaves_list, remainder)) if remainder > 0 else set()

    # Create assignments for each leaf
    leaf_assignments = {}
    for leaf in all_leaves:
        leaf_assignments[leaf] = base_assignment + (1 if leaf in bonus_leaves else 0)

    # Build result structure
    result = {}
    for key, leaf_set in data.items():
        result[key] = {}
        for leaf in leaf_set:
            result[key][leaf] = leaf_assignments[leaf]

    return result


def debater_input_creator(transcript: QualityTranscript, question: Question):
    return f"{transcript.article} {question.question}"


def judge_input_creator(transcript: QualityTranscript, question: Question):
    return question.question


def quality_record_to_sample(input_creator: Callable[[QualityTranscript, Question], str]):
    question_ids = get_question_ids()
    weights = assign_random_values(question_ids)

    def record_to_sample(record: dict):
        transcript = QualityTranscript.from_dict(record)
        samples = []

        for question in transcript.questions:
            if question.question.strip() not in question_ids[transcript.title.strip()]:
                continue

            # Get the correct answer index (0-based)
            correct_idx = question.gold_label - 1

            # Get all other indices
            other_indices = [i for i in range(len(question.options)) if i != correct_idx]

            # Randomly select one incorrect option
            distractor_idx = randomness.choice(other_indices)

            # Create the two options - correct and one distractor
            # Randomly decide whether correct answer comes first or second
            if randomness.choice([True, False]):
                # Correct answer first
                new_options = [question.options[correct_idx], question.options[distractor_idx]]
                new_target = 'A'
            else:
                # Distractor first, then correct answer
                new_options = [question.options[distractor_idx], question.options[correct_idx]]
                new_target = 'B'

            new_samples = [
                Sample(
                    input=input_creator(transcript, question),
                    choices=new_options,
                    target=new_target,
                    id=f"{transcript.title}_{question.question_unique_id}_{i}",
                    metadata={
                        "id": f"{transcript.article_id}_{question.question_unique_id}",
                        "title": transcript.title.strip(),
                        "question": question.question.strip(),
                    }
                )
                for i in range(int(weights[transcript.title.strip()][question.question.strip()]))
            ]

            samples.extend(new_samples)

        return samples

    return record_to_sample


quality_location = str(locations.PROJECT_ROOT / "data" / "datasets" / "quality" / "QuALITY.v1.0.1.htmlstripped.dev.jsonl")

debater_quality_dataset = json_dataset(
    quality_location,
    quality_record_to_sample(debater_input_creator)
)

judge_quality_dataset = json_dataset(
    quality_location,
    quality_record_to_sample(judge_input_creator)
)


