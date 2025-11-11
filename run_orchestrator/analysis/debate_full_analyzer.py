from collections.abc import Iterable, Iterator
from itertools import tee

from run_orchestrator.analysis.analysis_models.evaluation_configuration import EvaluationConfiguration
from run_orchestrator.analysis.debate_distribution_analyzer import (
    analyze_debate_distribution,
)
from run_orchestrator.analysis.debate_emptiness_analyzer import (
    analyze_debate_emptiness,
)
from run_orchestrator.analysis.debate_lengths_analyzer import (
    analyze_debate_lengths,
)
from run_orchestrator.analysis.debate_stats_analyzer import (
    analyze_debate_statistics,
)
from run_orchestrator.analysis.analysis_models.debate_stats import DebateStats
from run_orchestrator.analysis.analysis_models.full_debate_analysis import FullDebateAnalysis
from run_orchestrator.analysis.transcript_model import Transcript
from run_orchestrator.evals_generator.configuration_name import ConfigurationName


def full_debate_analysis(
    configuration: ConfigurationName,
    transcripts: Iterable[Transcript],
) -> FullDebateAnalysis:
    """Run all debate analytics against the same lazy transcript stream."""
    emptiness_iter, lengths_iter, distribution_iter, stats_iter = tee(transcripts, 4)
    emptiness_transcripts: Iterator[Transcript] = emptiness_iter
    lengths_transcripts: Iterator[Transcript] = lengths_iter
    distribution_transcripts: Iterator[Transcript] = distribution_iter
    stats_transcripts: Iterator[Transcript] = stats_iter

    emptiness_analysis = analyze_debate_emptiness(emptiness_transcripts)
    lengths_analysis = analyze_debate_lengths(lengths_transcripts)
    distribution_analysis = analyze_debate_distribution(distribution_transcripts)
    stats_analysis: DebateStats = analyze_debate_statistics(stats_transcripts)

    return FullDebateAnalysis(
        emptiness=emptiness_analysis,
        lengths=lengths_analysis,
        distribution=distribution_analysis,
        stats=stats_analysis,
        configuration=EvaluationConfiguration.from_configuration_name(configuration),
    )
