from script_utils import ScriptUtils
import os
import json
from datetime import datetime
from tqdm import tqdm

ScriptUtils.setup_script()

from debate import DebateRoundSummary, QuestionMetadata
from experiments import ExperimentLoader, ResultsCollector
from utils import logger_utils, constants

# 1. Setup
args = ScriptUtils.get_args()
config = ScriptUtils.get_debate_round_script_config(args)
logger = logger_utils.get_default_logger(__name__)
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
should_save_results = not args.test or args.force_save_results
should_save_transcripts = not args.test or args.force_save_transcripts

# 2. Load experiment
experiment_loader = ExperimentLoader(
    config_filepath=config.experiment_file_path,
    config_name=config.experiment_name,
    is_local=args.local,
    num_iterations=args.num_iters,
    starting_index=args.starting_index,
)
experiment = experiment_loader.load_experiment()

# 3. Setup ResultsCollector
results_collector = ResultsCollector(
    experiment=experiment,
    graphs_path_prefix=f"{config.graphs_path_prefix}/{start_time}_",
    full_record_path_prefix=f"{config.full_record_path_prefix}/{start_time}_",
    stats_path_prefix=f"{config.stats_path_prefix}/{start_time}",
    should_save=should_save_results,
)

# 4. Run debates or load transcripts
if args.transcripts_dir:
    if not os.path.isdir(args.transcripts_dir):
        raise ValueError(f"Provided transcripts_dir is not a directory: {args.transcripts_dir}")

    for filename in tqdm(os.listdir(args.transcripts_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(args.transcripts_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    transcript_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping {filename} due to JSON decoding error.")
                    continue

            metadata_json = transcript_data.get("metadata")
            if not metadata_json:
                logger.warning(f"Skipping {filename} due to missing metadata.")
                continue
            
            if isinstance(metadata_json, list):
                if not metadata_json:
                    logger.warning(f"Skipping {filename} due to empty metadata list.")
                    continue
                metadata_json = metadata_json[0]

            metadata = QuestionMetadata(**metadata_json)
            
            speeches = transcript_data.get("speeches", [])
            if not speeches:
                logger.warning(f"Skipping {filename} due to missing speeches.")
                continue

            judge_speech = speeches[-1]
            if judge_speech.get("speaker") != constants.DEFAULT_JUDGE_NAME:
                logger.warning(f"Skipping {filename} as last speaker is not the judge.")
                continue

            supplemental = judge_speech.get("supplemental")
            if not supplemental or "probabilistic_decision" not in supplemental:
                # This case can happen for branched transcripts for training data generation
                logger.debug(f"Skipping {filename} due to missing decision in judge's speech.")
                continue

            prob_decision = supplemental["probabilistic_decision"]
            if not prob_decision:
                logger.warning(f"Skipping {filename} due to empty probabilistic_decision.")
                continue
            
            first_debater_alias = experiment.agents.debaters[0].model_settings.alias
            if len(experiment.agents.debaters) > 1:
                second_debater_alias = experiment.agents.debaters[1].model_settings.alias
            else: # self-play
                second_debater_alias = first_debater_alias

            first_debater_win_prob = prob_decision.get(constants.DEFAULT_DEBATER_A_NAME, 0.0)
            second_debater_win_prob = prob_decision.get(constants.DEFAULT_DEBATER_B_NAME, 0.0)

            total_prob = first_debater_win_prob + second_debater_win_prob
            if total_prob > 0:
                first_debater_win_prob /= total_prob
                second_debater_win_prob /= total_prob
            
            first_debater_wins = first_debater_win_prob > 0.5

            summary = DebateRoundSummary(
                metadata=metadata,
                transcript="", # Not needed for graphing
                winning_alias=first_debater_alias if first_debater_wins else second_debater_alias,
                losing_alias=second_debater_alias if first_debater_wins else first_debater_alias,
                first_debater_alias=first_debater_alias,
                second_debater_alias=second_debater_alias,
                first_debater_wins=first_debater_wins,
                judge_alias=experiment.agents.judge.model_settings.alias,
                winning_debater_prob=max(first_debater_win_prob, second_debater_win_prob),
                first_debater_win_prob=first_debater_win_prob,
                second_debater_win_prob=second_debater_win_prob,
            )
            results_collector.record_result(summary)

else:
    debate_rounds = experiment_loader.load_debate_rounds(experiment=experiment)
    for i, debate_round in enumerate(tqdm(debate_rounds)):
        logger.info(f"Beginning round {i} out of {len(debate_rounds)}")
        save_file_path_prefix = f"{config.transcript_path_prefix}/{start_time}_{i}" if should_save_transcripts else None
        summary = debate_round(save_file_path_prefix=save_file_path_prefix)
        results_collector.record_result(summary)

# 5. Graph results
if not args.suppress_graphs:
    results_collector.graph_results()