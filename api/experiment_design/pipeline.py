"""
Layer 2 — Experiment Design Pipeline

Accepts ExperimentDesignInput (from Layer 1 ingestion) plus RAG-fetched documents,
calls the AIDEN-PI LLM using the system prompt in backend/prompts/experiment_design.txt,
and returns ExperimentDesignOutput.

Speed features retained from old_pipeline.py:
  1. Prompt caching  — system prompt cached server-side (ephemeral)
  2. Tiered models   — Opus for reasoning, Haiku for JSON formatting fallback
  3. Speculative parallel execution — divergent clarification paths run concurrently
  4. Context compression — prior reasoning compressed by Haiku between iterations
  5. Adaptive max_tokens — reduced on refinement iterations
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv

from api.schemas.layer2 import (
    BioinformaticsTask,
    ClarificationQuestion,
    ExperimentDesignOutput,
    PipelineState,
    RankedExperiment,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REASONING_MODEL   = "claude-sonnet-4-20250514"
FORMATTING_MODEL  = "claude-sonnet-4-20250514"
COMPRESSION_MODEL = "claude-sonnet-4-20250514"

MAX_ITERATIONS           = 4
CONFIDENCE_THRESHOLD     = 0.72
MAX_CLARIFICATION_ROUNDS = 2

# Load system prompt from the agreed file (do not duplicate it here)
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "experiment_design.txt"
EXPERIMENT_DESIGN_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

# Haiku fallback prompt: re-formats malformed Opus JSON output
_FORMATTING_SYSTEM_PROMPT = """\
You are a JSON formatter for a biological research pipeline. You will receive the raw
output of a scientific reasoning model that should have returned valid JSON but may have
included prose or malformed structure.

Extract all content and return ONLY a valid JSON object with these exact keys:
  stage_confirmed, reasoning_steps, ranked_experiments, missing_controls,
  key_hypothesis, literature_queries, bioinformatics_analyses.

ranked_experiments items must have: rank, experiment_type, title, protocol_summary,
rationale, controls (list of strings), expected_outcome, blocking (bool), stage_gate
(bool), estimated_weeks (int or null), estimated_cost_usd (int or null), cro_type,
biomni_tools (list of strings).

bioinformatics_analyses items must have: analysis, tool, rationale.

Output ONLY valid JSON. No prose, no fences.
"""

_COMPRESSION_SYSTEM_PROMPT = """\
You are a scientific summarizer. Compress the following reasoning trace to ≤200 words
of bullet points. Preserve key mechanistic conclusions, confidence levels, and
unresolved questions. Output plain text bullets only.
"""

log = logging.getLogger("experiment_design.pipeline")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ExperimentDesignPipeline:
    """
    Async Layer 2 pipeline. Wraps the LLM call with a confidence-gated
    feedback loop and speculative parallel execution for divergent paths.

    Usage:
        pipeline = ExperimentDesignPipeline()
        output   = await pipeline.run(experiment_design_input_dict, rag_documents)
    """

    def __init__(
        self,
        reasoning_model:          str   = REASONING_MODEL,
        formatting_model:         str   = FORMATTING_MODEL,
        compression_model:        str   = COMPRESSION_MODEL,
        confidence_threshold:     float = CONFIDENCE_THRESHOLD,
        max_iterations:           int   = MAX_ITERATIONS,
        max_clarification_rounds: int   = MAX_CLARIFICATION_ROUNDS,
        speculative_execution:    bool  = True,
    ):
        self.client                   = anthropic.AsyncAnthropic()
        self.reasoning_model          = reasoning_model
        self.formatting_model         = formatting_model
        self.compression_model        = compression_model
        self.confidence_threshold     = confidence_threshold
        self.max_iterations           = max_iterations
        self.max_clarification_rounds = max_clarification_rounds
        self.speculative_execution    = speculative_execution

    async def run(
        self,
        experiment_design_input: dict[str, Any],
        rag_documents: list[dict[str, Any]],
    ) -> ExperimentDesignOutput:
        """
        Main entry point.

        experiment_design_input: serialised ExperimentDesignInput dict from Layer 1
        rag_documents: list of RAG-fetched document dicts (from rag_service)
        """
        state = PipelineState()
        resolved_qa: list[dict[str, str]] = []

        while state.iteration < self.max_iterations and not state.converged:
            state.iteration += 1
            log.info(f"=== Layer 2 iteration {state.iteration}/{self.max_iterations} ===")

            user_msg = _build_user_message(
                experiment_design_input, rag_documents, state, resolved_qa
            )
            tokens   = 4096 if state.iteration == 1 else 2560
            raw      = await self._call_reasoning_llm(user_msg, tokens)
            output   = await self._parse_output(raw)

            state.prior_outputs.append(output)

            if self._is_converged(output):
                state.converged  = True
                output.status    = "final"
                log.info(f"Converged at iteration {state.iteration}")
                return output

            if output.needs_clarification and output.clarification_questions:
                state.clarification_rounds += 1
                new_qs = _deduplicate_questions(
                    output.clarification_questions, state.cumulative_questions
                )
                state.cumulative_questions.extend(new_qs)

                if self.speculative_execution and new_qs:
                    log.info("Speculative parallel execution for divergent paths")
                    output = await self._run_speculative_paths(
                        experiment_design_input, rag_documents, state, new_qs
                    )
                    state.prior_outputs[-1] = output
                    if self._is_converged(output):
                        state.converged = True
                        output.status   = "final"
                        return output

                if state.clarification_rounds >= self.max_clarification_rounds:
                    log.warning("Max clarification rounds reached — forcing output")
                    output.status = "forced_output"
                    output.pipeline_notes.append(
                        "FORCED: unanswered questions resolved with stated assumptions: "
                        + "; ".join(f'"{q.question}" → {q.impact_if_unresolved}'
                                    for q in state.cumulative_questions)
                    )
                    return output

                # Auto-apply stated assumptions and continue
                auto = [
                    {"question": q.question, "answer": f"[AUTO] {q.impact_if_unresolved}"}
                    for q in new_qs
                ]
                resolved_qa.extend(auto)
                log.info("Applied auto-assumptions. Continuing.")

        final = state.prior_outputs[-1]
        final.status = "forced_output"
        final.pipeline_notes.append(
            f"Hit max iterations ({self.max_iterations}) without convergence."
        )
        return final

    # ------------------------------------------------------------------
    # Speculative parallel execution
    # ------------------------------------------------------------------

    async def _run_speculative_paths(
        self,
        edi:       dict[str, Any],
        rag_docs:  list[dict[str, Any]],
        state:     PipelineState,
        questions: list[ClarificationQuestion],
    ) -> ExperimentDesignOutput:
        path_a_qa = [
            {"question": q.question, "answer": f"[PATH A] {q.option_a}"}
            for q in questions
        ]
        path_b_qa = [
            {"question": q.question, "answer": f"[PATH B] {q.option_b}"}
            for q in questions
        ]
        snap = copy.deepcopy(state)
        out_a, out_b = await asyncio.gather(
            self._run_single(edi, rag_docs, snap, path_a_qa),
            self._run_single(edi, rag_docs, snap, path_b_qa),
        )
        winner = out_a if out_a.overall_confidence >= out_b.overall_confidence else out_b
        label  = "A" if winner is out_a else "B"
        winner.pipeline_notes.append(
            f"Speculative execution: path A={out_a.overall_confidence:.2f}, "
            f"path B={out_b.overall_confidence:.2f}. Path {label} selected."
        )
        return winner

    async def _run_single(
        self,
        edi:         dict[str, Any],
        rag_docs:    list[dict[str, Any]],
        state:       PipelineState,
        resolved_qa: list[dict[str, str]],
    ) -> ExperimentDesignOutput:
        user_msg = _build_user_message(edi, rag_docs, state, resolved_qa)
        raw      = await self._call_reasoning_llm(user_msg, 2560)
        return await self._parse_output(raw)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _call_reasoning_llm(self, user_message: str, max_tokens: int) -> str:
        log.info(f"Reasoning call — {self.reasoning_model}, max_tokens={max_tokens}")
        response = await self.client.messages.create(
            model=self.reasoning_model,
            max_tokens=max_tokens,
            system=EXPERIMENT_DESIGN_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
                # Response prefilling: force the model to start with '{' for JSON
                {"role": "assistant", "content": "{"},
            ],
        )
        text = _extract_text_block(response)
        # Prepend the '{' we used for prefilling
        if text and not text.startswith("{"):
            text = "{" + text
        return text

    async def _call_formatting_llm(self, raw_opus_output: str) -> str:
        """Haiku fallback: convert malformed Opus output → clean JSON."""
        log.info("Formatting fallback call — Haiku")
        response = await self.client.messages.create(
            model=self.formatting_model,
            max_tokens=4096,
            system=_FORMATTING_SYSTEM_PROMPT,
            messages=[
                {
                    "role":    "user",
                    "content": (
                        "Convert the following scientific reasoning output into the "
                        "required JSON schema. Output only valid JSON.\n\n"
                        + raw_opus_output
                    ),
                }
            ],
        )
        return _extract_text_block(response)

    async def _compress_reasoning(self, trace: str) -> str:
        if len(trace) < 600:
            return trace
        log.info("Compressing prior reasoning trace")
        response = await self.client.messages.create(
            model=self.compression_model,
            max_tokens=400,
            system=_COMPRESSION_SYSTEM_PROMPT,
            messages=[
                {
                    "role":    "user",
                    "content": "Compress:\n\n" + trace,
                }
            ],
        )
        return _extract_text_block(response)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    async def _parse_output(self, raw: str) -> ExperimentDesignOutput:
        json_str = _extract_json(raw)
        if not json_str:
            log.warning("No JSON found — running formatting fallback")
            json_str = await self._haiku_fix(raw)
            if not json_str:
                return _error_output("No JSON extractable from LLM output")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            log.warning(f"JSON parse error: {exc} — running formatting fallback")
            json_str = await self._haiku_fix(raw)
            if not json_str:
                return _error_output(str(exc))
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as exc2:
                log.error(f"Formatting fallback also failed: {exc2}")
                return _error_output(str(exc2))

        return _build_output(data)

    async def _haiku_fix(self, raw: str) -> str:
        try:
            result = await self._call_formatting_llm(raw)
            return _repair_json(result) if result else ""
        except Exception as exc:
            log.error(f"Formatting fallback failed: {exc}")
            return ""

    def _is_converged(self, output: ExperimentDesignOutput) -> bool:
        return (
            not output.needs_clarification
            and output.overall_confidence >= self.confidence_threshold
            and bool(output.ranked_experiments)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_block(response) -> str:
    """
    Extract the first text block from an Anthropic response.

    Claude 4 models (Opus 4.6) may return extended thinking blocks before
    the text block, so we can't assume content[0] is always a TextBlock.
    Falls back to an empty string if no text block is found, which triggers
    the Haiku formatting fallback in _parse_output.
    """
    if not response.content:
        log.warning(
            f"Empty content list from API. stop_reason={getattr(response, 'stop_reason', 'unknown')}"
        )
        return ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    # No text block found (e.g. only thinking blocks) — return empty string
    log.warning(f"No text block in response content: {[b.type for b in response.content]}")
    return ""


async def _compress_prior(pipeline: ExperimentDesignPipeline, trace: str) -> str:
    return await pipeline._compress_reasoning(trace)


def _build_user_message(
    edi:         dict[str, Any],
    rag_docs:    list[dict[str, Any]],
    state:       PipelineState,
    resolved_qa: list[dict[str, str]],
) -> str:
    parts: list[str] = []

    if state.iteration > 1 and state.prior_outputs:
        prior = state.prior_outputs[-1]
        parts.append(
            f"=== ITERATION {state.iteration} — refinement ===\n"
            f"Focus on unresolved questions and low-confidence areas.\n\n"
            f"PRIOR REASONING SUMMARY:\n{prior.reasoning_steps[0] if prior.reasoning_steps else ''}\n\n"
            f"STILL OPEN: {', '.join(str(f) for f in [])}\n"
        )

    if resolved_qa:
        parts.append("=== CLARIFICATION ANSWERS ===")
        for qa in resolved_qa:
            parts.append(f"Q: {qa['question']}\nA: {qa['answer']}")
        parts.append("")

    parts.append("=== PROGRAM STATE (structured input from ingestion layer) ===")
    parts.append(json.dumps(edi, indent=2, default=str))

    if rag_docs:
        parts.append("\n=== RAG-FETCHED RESISTANCE LITERATURE (CARD / AlphaFold / IMGT) ===")
        for doc in rag_docs[:8]:   # cap at 8 to stay within context budget
            parts.append(json.dumps(doc, default=str))

    parts.append(
        "\nBased on the above, design the next experiments. "
        "CRITICAL: Respond with ONLY a valid JSON object. No markdown fences, no comments, no trailing commas. "
        "Every string value must be properly escaped. The response must parse with json.loads()."
    )
    return "\n".join(parts)


def _repair_json(text: str) -> str:
    """
    Attempt to repair common LLM JSON issues:
    - Trailing commas before } or ]
    - Single-line // comments
    - Python-style # comments
    - Unescaped newlines in strings
    - Missing closing braces
    """
    # Remove single-line comments (// and #)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'(?<!["\w])#[^\n]*', '', text)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Balance braces — append missing closing braces
    opens = text.count('{') - text.count('}')
    if opens > 0:
        text += '}' * opens
    brackets = text.count('[') - text.count(']')
    if brackets > 0:
        text += ']' * brackets

    return text.strip()


def _extract_json(text: str) -> Optional[str]:
    """Extract first ```json...``` fence, or fall back to bare {…}, with auto-repair."""
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        return _repair_json(m.group(1).strip())
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        return _repair_json(m.group(1).strip())
    # If text starts with { (from prefilling), treat entire text as JSON
    text = text.strip()
    if text.startswith("{"):
        return _repair_json(text)
    return None


def _build_output(data: dict[str, Any]) -> ExperimentDesignOutput:
    ranked = [
        RankedExperiment(
            rank=int(e.get("rank", 99)),
            experiment_type=e.get("experiment_type", ""),
            title=e.get("title", ""),
            protocol_summary=e.get("protocol_summary", ""),
            rationale=e.get("rationale", ""),
            controls=e.get("controls", []),
            expected_outcome=e.get("expected_outcome", ""),
            blocking=bool(e.get("blocking", False)),
            stage_gate=bool(e.get("stage_gate", False)),
            estimated_weeks=e.get("estimated_weeks"),
            estimated_cost_usd=e.get("estimated_cost_usd"),
            cro_type=e.get("cro_type", ""),
            biomni_tools=e.get("biomni_tools", []),
        )
        for e in data.get("ranked_experiments", [])
    ]
    bioinf = [
        BioinformaticsTask(
            analysis=b.get("analysis", ""),
            tool=b.get("tool", ""),
            rationale=b.get("rationale", ""),
        )
        for b in data.get("bioinformatics_analyses", [])
    ]
    return ExperimentDesignOutput(
        stage_confirmed=data.get("stage_confirmed", "unknown"),
        reasoning_steps=data.get("reasoning_steps", []),
        ranked_experiments=ranked,
        missing_controls=data.get("missing_controls", []),
        key_hypothesis=data.get("key_hypothesis", ""),
        literature_queries=data.get("literature_queries", []),
        bioinformatics_analyses=bioinf,
        overall_confidence=float(data.get("overall_confidence", 0.85)),
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_questions=[
            ClarificationQuestion(
                question=q.get("question", ""),
                why_needed=q.get("why_needed", ""),
                option_a=q.get("option_a", ""),
                option_b=q.get("option_b", ""),
                impact_if_unresolved=q.get("impact_if_unresolved", ""),
            )
            for q in data.get("clarification_questions", [])
        ],
        pipeline_notes=data.get("pipeline_notes", []),
    )


def _error_output(error_msg: str) -> ExperimentDesignOutput:
    return ExperimentDesignOutput(
        stage_confirmed="unknown",
        reasoning_steps=[],
        ranked_experiments=[],
        missing_controls=[],
        key_hypothesis="",
        literature_queries=[],
        bioinformatics_analyses=[],
        overall_confidence=0.0,
        needs_clarification=True,
        status="forced_output",
        pipeline_notes=[f"Parse error: {error_msg}"],
    )


def _deduplicate_questions(
    new_qs: list[ClarificationQuestion],
    existing: list[ClarificationQuestion],
) -> list[ClarificationQuestion]:
    seen = {q.question for q in existing}
    return [q for q in new_qs if q.question not in seen]
