"""
Layer 2 — Experiment Design data structures.

Schema is driven by backend/prompts/experiment_design.txt (the AIDEN-PI prompt).
If that prompt changes its JSON output shape, update these dataclasses and the
parser in pipeline.py to match.

The old BioResearchContext / PipelineOutput schema is preserved in old_models.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# LLM output components  (match experiment_design.txt JSON schema exactly)
# ---------------------------------------------------------------------------

@dataclass
class RankedExperiment:
    rank:               int
    experiment_type:    str           # enzyme_inhibition | mic_assay | time_kill | ...
    title:              str
    protocol_summary:   str           # 2-3 sentence protocol outline
    rationale:          str           # cites actual data values from the program
    controls:           list[str]     # ["positive control: ...", "negative control: ..."]
    expected_outcome:   str           # what result confirms/denies the hypothesis
    blocking:           bool          # must complete before anything else
    stage_gate:         bool          # determines stage progression
    estimated_weeks:    Optional[int]
    estimated_cost_usd: Optional[int]
    cro_type:           str           # who does this experiment
    biomni_tools:       list[str]     # relevant Biomni/bioinformatics tools


@dataclass
class BioinformaticsTask:
    analysis: str   # analysis name
    tool:     str   # specific tool or pipeline
    rationale: str  # why this analysis


@dataclass
class ClarificationQuestion:
    """A targeted scientific question the pipeline cannot resolve from data alone."""
    question:             str
    why_needed:           str
    option_a:             str
    option_b:             str
    impact_if_unresolved: str


# ---------------------------------------------------------------------------
# Full Layer 2 output
# ---------------------------------------------------------------------------

@dataclass
class ExperimentDesignOutput:
    """
    Complete output from the Layer 2 LLM pipeline.
    This is what the experiment_design router serialises and returns.
    """
    stage_confirmed:        str
    reasoning_steps:        list[str]
    ranked_experiments:     list[RankedExperiment]
    missing_controls:       list[str]
    key_hypothesis:         str
    literature_queries:     list[str]
    bioinformatics_analyses: list[BioinformaticsTask]

    # Pipeline metadata (feedback loop state)
    overall_confidence:     float = 1.0
    needs_clarification:    bool  = False
    clarification_questions: list[ClarificationQuestion] = field(default_factory=list)
    pipeline_notes:         list[str] = field(default_factory=list)
    status:                 str  = "final"   # "final" | "needs_clarification" | "forced_output"


# ---------------------------------------------------------------------------
# Internal pipeline state (not exposed externally)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    iteration:            int  = 0
    prior_outputs:        list[ExperimentDesignOutput] = field(default_factory=list)
    cumulative_questions: list[ClarificationQuestion]  = field(default_factory=list)
    resolved_qa:          list[dict]                   = field(default_factory=list)
    clarification_rounds: int  = 0
    converged:            bool = False
