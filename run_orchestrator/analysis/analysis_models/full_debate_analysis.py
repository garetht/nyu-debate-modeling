from run_orchestrator.analysis.analysis_models.analysis_result import AnalysisResult
from run_orchestrator.analysis.analysis_models.debate_distribution import DebateDistributionAnalysis
from run_orchestrator.analysis.analysis_models.debate_emptiness import DebateEmptinessAnalysis
from run_orchestrator.analysis.analysis_models.debate_lengths import DebateLengthAnalysis
from run_orchestrator.analysis.analysis_models.debate_stats import DebateStats
from run_orchestrator.analysis.analysis_models.evaluation_configuration import EvaluationConfiguration
from run_orchestrator.analysis.analysis_models.evaluation_configuration import EvaluationConfiguration


class FullDebateAnalysis(AnalysisResult):
    emptiness: DebateEmptinessAnalysis
    lengths: DebateLengthAnalysis
    distribution: DebateDistributionAnalysis
    stats: DebateStats
    configuration: EvaluationConfiguration
    configuration: EvaluationConfiguration
