"""
Instruction generation layer.

Intentionally isolated from perception and the FSM to:
- keep the core pipeline deterministic and debuggable
- plug in an LLM for optional instruction generation

Provides:
- InstructionClient (interface)
- TemplateInstructionClient (no network calls, always available)
- DeepSeekInstructionClient (real LLM call using OpenAI-compatible SDK + DeepSeek base_url)
- procedure_to_event_log + save_event_log_json (stable JSON export for later use)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models import ActionLabel, StepEvent, Procedure




class InstructionClient(ABC):
    """
    Turns a Procedure into a human-readable instruction document.
    """

    @abstractmethod
    def generate_instructions(self, procedure: Procedure) -> str:
        raise NotImplementedError




_TEMPLATE_TEXT: Dict[ActionLabel, str] = {
    ActionLabel.IDLE: "Prepare the parts and workspace before starting.",

    ActionLabel.LID_REORIENT: (
        "Pick up and reorient the glass lid so the center hole is accessible. "
        "This may include flipping the lid and adjusting your grip."
    ),
    ActionLabel.PICK_WASHER: (
        "Pick up the washer. Check which side is soft or matte, since that side should face the glass."
    ),
    ActionLabel.SEAT_WASHER: (
        "Place the washer into the lid’s center hole and press it down until it is seated firmly."
    ),
    ActionLabel.PICK_SCREW: "Pick up the screw and bring it to the lid.",
    ActionLabel.INSERT_SCREW: (
        "Insert the screw through the washer and the lid’s center hole, then press it down so it sits correctly."
    ),
    ActionLabel.PICK_HANDLE: "Pick up the handle and bring it to the lid.",
    ActionLabel.INSERT_HANDLE: "Place the handle onto the screw and align it so the thread engages.",
    ActionLabel.HAND_TIGHTEN_HANDLE: (
        "Rotate the handle clockwise by hand until it is snug and the assembly holds together."
    ),

    ActionLabel.PLACE_LID_FOR_TOOL: (
        "Flip and place the lid so the screw head is accessible for the screwdriver "
        "(the handle typically faces down)."
    ),
    ActionLabel.PICK_SCREWDRIVER: "Pick up the screwdriver and move it toward the screw head.",
    ActionLabel.TOOL_SEAT: (
        "Seat the screwdriver tip into the screw head and apply gentle downward pressure to avoid slipping."
    ),
    ActionLabel.TOOL_TIGHTEN: (
        "Tighten the screw with the screwdriver using short turns and regrips as needed. "
        "Stop once the handle is firmly attached. Avoid overtightening to reduce stress on the glass."
    ),
    ActionLabel.SET_TOOL_DOWN: "Remove the screwdriver from the screw head and set it down, then regrip the lid.",

    ActionLabel.FINAL_FLIP_AND_PLACE: "Flip the lid so the handle faces up and place the assembled lid on the table.",
    ActionLabel.RELEASE: "Release your hands and finish the task.",
}


class TemplateInstructionClient(InstructionClient):
    """
    Deterministic, offline instructions from fixed templates.
    """

    def __init__(self, include_timings: bool = True, skip_labels: Optional[Sequence[ActionLabel]] = None):
        self.include_timings = include_timings
        self.skip_labels = set(skip_labels or [])

    def generate_instructions(self, procedure: Procedure) -> str:
        lines: List[str] = []
        lines.append(f"Task: {procedure.task_name}")
        lines.append(f"Source video: {procedure.video_file}")
        lines.append("")
        lines.append("Automatically derived step by step instructions:")
        lines.append("")

        for idx, step in enumerate(procedure.steps, start=1):
            if step.label in self.skip_labels:
                continue

            template = _TEMPLATE_TEXT.get(step.label, f"Perform action: {step.label.value}.")
            header = f"Step {idx}: {step.label.value}"
            if self.include_timings:
                header += f"  (approx. {step.start_time:.1f} s → {step.end_time:.1f} s)"

            lines.append(header)
            lines.append(f"  {template}")
            lines.append("")

        return "\n".join(lines)




def procedure_to_event_log(procedure: Procedure) -> List[Dict[str, Any]]:
    """
    Convert a Procedure into a JSON-serializable list of event dicts.
    """
    events: List[Dict[str, Any]] = []
    for idx, step in enumerate(procedure.steps, start=1):
        events.append(
            {
                "step_id": idx,
                "label": step.label.value,
                "description_key": step.description_key,
                "start_time": float(step.start_time),
                "end_time": float(step.end_time),
                "confidence": float(step.confidence),
                "features_summary": dict(step.features_summary),
            }
        )
    return events


def save_event_log_json(procedure: Procedure, path: str) -> None:
    events = procedure_to_event_log(procedure)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)




_DEFAULT_SKIP_LABELS = {ActionLabel.IDLE, ActionLabel.RELEASE}

_DEFAULT_WARNINGS = [
    "Do not overtighten the screw to avoid stressing or cracking the glass.",
]

_DEFAULT_CHECKS = [
    "After tightening, gently wiggle the handle to confirm it is firmly attached.",
]


_DEEPSEEK_SYSTEM_PROMPT = """
You convert a detected action event log into a step-by-step work instruction document.

You MUST return ONLY valid json.

Schema (json):
{
  "task_title": string,
  "steps": [
    {
      "source_step_id": number,
      "label": string,
      "start_time": number,
      "end_time": number,
      "confidence": number,
      "instruction": string
    }
  ],
  "warnings": [string],
  "checks": [string]
}

Rules:
- Use ONLY the provided events. Keep the same order.
- Do NOT invent extra steps or remove steps. The number of output steps must equal the number of input events provided.
- Each instruction must be 1–2 short imperative sentences.
- If confidence < 0.55, the instruction MUST start with "Verify:".
- Do not mention objects or tools that are not implied by the provided label guide and draft instruction.
"""


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    JSON mode should already return pure JSON, but this keeps us safe if extra text appears.
    """
    text = text.strip()
    return json.loads(text)


def _build_llm_events(
    procedure: Procedure,
    skip_labels: Sequence[ActionLabel],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a compact event list for the LLM and a list of skipped events for bookkeeping.
    We pass a short "draft_instruction" per step to constrain the LLM and prevent hallucination.
    """
    llm_events: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for idx, step in enumerate(procedure.steps, start=1):
        if step.label in skip_labels:
            skipped.append({"source_step_id": idx, "label": step.label.value, "reason": "skipped_by_policy"})
            continue

        draft = _TEMPLATE_TEXT.get(step.label, f"Perform action: {step.label.value}.")
        llm_events.append(
            {
                "source_step_id": idx,
                "label": step.label.value,
                "start_time": float(step.start_time),
                "end_time": float(step.end_time),
                "confidence": float(step.confidence),
                "draft_instruction": draft,
            }
        )

    return llm_events, skipped


def _validate_llm_json(
    obj: Dict[str, Any],
    expected_events: List[Dict[str, Any]],
) -> None:
    """
    Strict validation to keep outputs deterministic and easy to present.
    """
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object.")

    if "steps" not in obj or not isinstance(obj["steps"], list):
        raise ValueError("LLM output missing 'steps' list.")

    steps = obj["steps"]
    if len(steps) != len(expected_events):
        raise ValueError(f"LLM returned {len(steps)} steps, expected {len(expected_events)}.")

    for i, (out_step, exp) in enumerate(zip(steps, expected_events), start=1):
        if not isinstance(out_step, dict):
            raise ValueError(f"Step {i} is not an object.")

        for k in ("source_step_id", "label", "start_time", "end_time", "confidence", "instruction"):
            if k not in out_step:
                raise ValueError(f"Step {i} missing key: {k}")

        if int(out_step["source_step_id"]) != int(exp["source_step_id"]):
            raise ValueError(f"Step {i} source_step_id mismatch.")

        if str(out_step["label"]) != str(exp["label"]):
            raise ValueError(f"Step {i} label mismatch.")

        instr = out_step.get("instruction", "")
        if not isinstance(instr, str) or not instr.strip():
            raise ValueError(f"Step {i} instruction is empty.")

    for k in ("warnings", "checks"):
        if k in obj:
            if not isinstance(obj[k], list) or any((not isinstance(x, str)) for x in obj[k]):
                raise ValueError(f"'{k}' must be a list of strings.")


def _enforce_verify_prefix(obj: Dict[str, Any], verify_threshold: float) -> None:
    """
    Ensure low-confidence steps start with 'Verify:' even if the model forgets.
    """
    steps = obj.get("steps", [])
    for st in steps:
        conf = float(st.get("confidence", 1.0))
        instr = str(st.get("instruction", "")).strip()
        if conf < verify_threshold and not instr.startswith("Verify:"):
            st["instruction"] = f"Verify: {instr}"


def _render_instruction_text(
    instruction_json: Dict[str, Any],
    procedure: Procedure,
    include_timings: bool,
) -> str:
    """
    Render JSON into a human-readable text document.
    """
    title = str(instruction_json.get("task_title", procedure.task_name))

    lines: List[str] = []
    lines.append(f"Task: {title}")
    lines.append(f"Source video: {procedure.video_file}")
    lines.append("")
    lines.append("Automatically derived step by step instructions:")
    lines.append("")

    for idx, st in enumerate(instruction_json.get("steps", []), start=1):
        label = str(st.get("label", ""))
        sid = int(st.get("source_step_id", idx))
        start_t = float(st.get("start_time", 0.0))
        end_t = float(st.get("end_time", 0.0))
        instr = str(st.get("instruction", "")).strip()

        header = f"Step {idx}: {label}  (source {sid})"
        if include_timings:
            header += f"  (approx. {start_t:.1f} s → {end_t:.1f} s)"

        lines.append(header)
        lines.append(f"  {instr}")
        lines.append("")

    warnings = instruction_json.get("warnings") or []
    checks = instruction_json.get("checks") or []

    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w.strip()}")
        lines.append("")

    if checks:
        lines.append("Checks:")
        for c in checks:
            lines.append(f"- {c.strip()}")
        lines.append("")

    return "\n".join(lines)


class DeepSeekInstructionClient(InstructionClient):
    """
    DeepSeek-backed instruction generator.

    Design objectives:
    - JSON-only responses (stable parsing)
    - Strong constraints using draft instructions per step
    - Strict validation and fallback to TemplateInstructionClient
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 1200,
        include_timings: bool = True,
        skip_labels: Optional[Sequence[ActionLabel]] = None,
        verify_threshold: float = 0.55,
        fallback_to_template: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.include_timings = bool(include_timings)
        self.skip_labels = list(skip_labels) if skip_labels is not None else list(_DEFAULT_SKIP_LABELS)
        self.verify_threshold = float(verify_threshold)
        self.fallback_to_template = bool(fallback_to_template)

        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError(
                "DeepSeekInstructionClient requires the 'openai' Python package. "
                "Install it with: pip install openai"
            ) from e

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def from_env(
        cls,
        env_var: str = "DEEPSEEK_API_KEY",
        **kwargs: Any,
    ) -> "DeepSeekInstructionClient":
        api_key = os.getenv(env_var, "").strip()
        if not api_key:
            raise ValueError(f"Missing API key. Set environment variable {env_var}.")
        return cls(api_key=api_key, **kwargs)

    def generate_instructions_json(self, procedure: Procedure) -> Dict[str, Any]:
        """
        Returns the parsed JSON object (validated).
        """
        llm_events, _skipped = _build_llm_events(procedure, skip_labels=self.skip_labels)

        if not llm_events:
            return {
                "task_title": procedure.task_name,
                "steps": [],
                "warnings": list(_DEFAULT_WARNINGS),
                "checks": list(_DEFAULT_CHECKS),
            }

        user_payload = {
            "task_title": procedure.task_name,
            "label_guide": {
                "LID_REORIENT": "Reorient the lid so the center hole is accessible.",
                "PICK_WASHER": "Pick up the washer.",
                "SEAT_WASHER": "Place and press the washer into the center hole.",
                "PICK_SCREW": "Pick up the screw.",
                "INSERT_SCREW": "Insert and press the screw through the center.",
                "PICK_HANDLE": "Pick up the handle.",
                "INSERT_HANDLE": "Place and align the handle onto the screw.",
                "HAND_TIGHTEN_HANDLE": "Hand-tighten the handle clockwise.",
                "PLACE_LID_FOR_TOOL": "Flip/place the lid to access the screw head for tool use.",
                "PICK_SCREWDRIVER": "Pick up the screwdriver.",
                "TOOL_SEAT": "Seat the driver tip into the screw head.",
                "TOOL_TIGHTEN": "Tighten the screw using the screwdriver.",
                "SET_TOOL_DOWN": "Remove and set down the screwdriver.",
                "FINAL_FLIP_AND_PLACE": "Flip and place the assembled lid on the table.",
            },
            "events": llm_events,
            "default_warnings": list(_DEFAULT_WARNINGS),
            "default_checks": list(_DEFAULT_CHECKS),
        }

        messages = [
            {"role": "system", "content": _DEEPSEEK_SYSTEM_PROMPT},
            {"role": "user", "content": "Generate json from this payload:\n" + json.dumps(user_payload)},
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content or ""
        obj = _safe_json_loads(content)

        _validate_llm_json(obj, expected_events=llm_events)
        _enforce_verify_prefix(obj, verify_threshold=self.verify_threshold)

        if not obj.get("warnings"):
            obj["warnings"] = list(_DEFAULT_WARNINGS)
        if not obj.get("checks"):
            obj["checks"] = list(_DEFAULT_CHECKS)

        if not obj.get("task_title"):
            obj["task_title"] = procedure.task_name

        return obj

    def generate_instructions(self, procedure: Procedure) -> str:
        """
        Returns a rendered text document. Falls back to templates if enabled.
        """
        try:
            obj = self.generate_instructions_json(procedure)
            return _render_instruction_text(obj, procedure, include_timings=self.include_timings)
        except Exception:
            if not self.fallback_to_template:
                raise
            fallback = TemplateInstructionClient(include_timings=self.include_timings, skip_labels=self.skip_labels)
            return fallback.generate_instructions(procedure)
