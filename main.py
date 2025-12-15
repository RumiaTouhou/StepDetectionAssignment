# main.py

"""
Entry point for the pot lid assembly action recognition prototype.

Pipeline:
- Load video
- Extract per-frame features (vision, hand tracking, optical flow, audio)
- Run the ordered context FSM to obtain a Procedure (sequence of steps)
- Save:
    - features.csv
    - steps.csv
    - process_log.json (event log suitable as input to an instruction generator)
    - instructions_template.txt (offline templates) OR instructions_llm.txt (DeepSeek LLM)
    - timeline.png (action timeline)
    - features.png (features over time with step boundaries)
    - optionally an annotated video with overlayed labels

LLM notes:
- DeepSeek is optional. Default is template instructions.
- To use DeepSeek, set an API key in an environment variable (default: DEEPSEEK_API_KEY)
  and run with: --instruction-mode deepseek
"""


from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional



from models import FrameFeatures, Procedure, ActionLabel
from perception import extract_frame_features, PerceptionConfig
from context_fsm import ContextStateMachine, FSMConfig
from llm_socket import (
    TemplateInstructionClient,
    DeepSeekInstructionClient,
    save_event_log_json,
)
from visualization import (
    plot_action_timeline,
    plot_features_with_steps,
    write_annotated_video,
)


# ---------------------------------------------------------------------------
# Helpers for saving tabular outputs
# ---------------------------------------------------------------------------


def save_features_csv(features: List[FrameFeatures], out_path: str) -> None:
    """
    Save the list of FrameFeatures as a CSV file.
    """
    if not features:
        print("save_features_csv: no features to save.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Use dataclasses.asdict for convenience
    fieldnames = list(asdict(features[0]).keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in features:
            writer.writerow(asdict(feat))

    print(f"Frame features saved to: {out_path}")


def save_steps_csv(procedure: Procedure, out_path: str) -> None:
    """
    Save the detected steps as a CSV file.
    """
    if not procedure.steps:
        print("save_steps_csv: no steps to save.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "step_id",
        "label",
        "description_key",
        "start_time",
        "end_time",
        "duration",
        "confidence",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(procedure.steps, start=1):
            duration = step.end_time - step.start_time
            writer.writerow(
                {
                    "step_id": idx,
                    "label": step.label.value,
                    "description_key": step.description_key,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "duration": duration,
                    "confidence": step.confidence,
                }
            )

    print(f"Step list saved to: {out_path}")


def save_instructions_text(
    procedure: Procedure,
    out_path: str,
    mode: str = "template",
    deepseek_model: str = "deepseek-chat",
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY",
    deepseek_temperature: float = 0.2,
    deepseek_max_tokens: int = 1200,
) -> Dict[int, str]:
    """
    Save a .txt instruction document and return a mapping for video overlay.

    Returns:
      Dict[int, str] mapping procedure step index (0-based) -> instruction string

    mode:
      - "template": offline, deterministic templates
      - "deepseek": calls DeepSeek to rewrite step instructions
    """
    mode = (mode or "template").strip().lower()
    skip_labels = {ActionLabel.IDLE, ActionLabel.RELEASE}

    if mode == "deepseek":
        api_key = os.getenv(deepseek_api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"Instruction mode is 'deepseek' but no API key was found in env var {deepseek_api_key_env}."
            )

        client = DeepSeekInstructionClient(
            api_key=api_key,
            base_url=deepseek_base_url,
            model=deepseek_model,
            temperature=deepseek_temperature,
            max_tokens=deepseek_max_tokens,
            include_timings=True,
            fallback_to_template=True,
        )
        text = client.generate_instructions(procedure)
        label = "LLM instructions"
    else:
        client = TemplateInstructionClient(include_timings=True, skip_labels=list(skip_labels))
        text = client.generate_instructions(procedure)
        label = "Template-based instructions"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"{label} saved to: {out_path}")

    # ----------------------------
    # Build instruction map for overlay
    # ----------------------------
    parsed_steps: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for line in text.splitlines():
        line = line.strip("\n")

        if line.startswith("Step "):
            # Example formats:
            # "Step 1: LID_REORIENT  (source 2)  (approx. ...)"
            # "Step 1: LID_REORIENT  (approx. ...)"
            cur = {"source_step_id": None, "label": "", "instruction": ""}

            # Label
            try:
                rhs = line.split(":", 1)[1].strip()
                label_part = rhs.split("  (", 1)[0].strip()
                cur["label"] = label_part
            except Exception:
                cur["label"] = ""

            # Source step id (optional)
            s_pos = line.find("(source ")
            if s_pos != -1:
                s_end = line.find(")", s_pos)
                if s_end != -1:
                    num_str = line[s_pos + len("(source ") : s_end].strip()
                    if num_str.isdigit():
                        cur["source_step_id"] = int(num_str)

            parsed_steps.append(cur)
            continue

        if cur is not None and line.startswith("  ") and line.strip():
            # First indented non-empty line after header is the instruction
            if not cur.get("instruction"):
                cur["instruction"] = line.strip()

    instruction_map: Dict[int, str] = {}

    # If source IDs exist, map directly to procedure indices
    has_source = any((isinstance(s.get("source_step_id"), int)) for s in parsed_steps)
    if has_source:
        for s in parsed_steps:
            sid = s.get("source_step_id")
            instr = str(s.get("instruction", "")).strip()
            if isinstance(sid, int) and 1 <= sid <= len(procedure.steps) and instr:
                instruction_map[sid - 1] = instr
        return instruction_map

    # Otherwise align sequentially to procedure steps excluding IDLE/RELEASE
    kept_indices = [i for i, st in enumerate(procedure.steps) if st.label not in skip_labels]
    for j, s in enumerate(parsed_steps):
        if j >= len(kept_indices):
            break
        instr = str(s.get("instruction", "")).strip()
        if instr:
            instruction_map[kept_indices[j]] = instr

    return instruction_map




# ---------------------------------------------------------------------------
# CLI and main pipeline
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pot lid assembly action recognition prototype.",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=False,
        default="20251205_203444.mp4",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=False,
        default="Pot Lid Assembly",
        help="Task name to embed in outputs.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=False,
        default="outputs",
        help="Directory where outputs (CSVs, plots, etc.) will be saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving PNG plots.",
    )
    parser.add_argument(
        "--annotate-video",
        action="store_true",
        help="Generate an annotated video with overlayed labels.",
    )

    # Instruction generation (template or DeepSeek)
    parser.add_argument(
        "--instruction-mode",
        type=str,
        choices=["template", "deepseek"],
        default="template",
        help="Instruction generator backend: offline templates or DeepSeek LLM.",
    )
    parser.add_argument(
        "--deepseek-model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek model name (used only when --instruction-mode deepseek).",
    )
    parser.add_argument(
        "--deepseek-base-url",
        type=str,
        default="https://api.deepseek.com",
        help="DeepSeek base URL for the OpenAI-compatible API (used only in deepseek mode).",
    )
    parser.add_argument(
        "--deepseek-api-key-env",
        type=str,
        default="DEEPSEEK_API_KEY",
        help="Environment variable that contains your DeepSeek API key.",
    )
    parser.add_argument(
        "--deepseek-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for DeepSeek (used only in deepseek mode).",
    )
    parser.add_argument(
        "--deepseek-max-tokens",
        type=int,
        default=1200,
        help="Max tokens for DeepSeek response (used only in deepseek mode).",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    video_path = args.video
    task_name = args.task_name
    output_dir = args.output_dir
    save_plots = not args.no_plots
    make_annotated_video = args.annotate_video

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Task:  {task_name}")
    print(f"Output directory: {output_dir}")

    # --- 1. Perception: per-frame features --------------------------------
    print("Step 1/4: extracting frame features...")
    perception_cfg = PerceptionConfig()
    features: List[FrameFeatures] = extract_frame_features(video_path, cfg=perception_cfg)
    print(f"  Extracted {len(features)} analysis frames.")

    if not features:
        print("No features extracted, aborting.")
        return

    # Save features as CSV
    features_csv_path = os.path.join(output_dir, "features.csv")
    save_features_csv(features, features_csv_path)

    # --- 2. Context FSM: infer Procedure -----------------------------------
    print("Step 2/4: running context state machine...")
    fsm_cfg = FSMConfig()
    fsm = ContextStateMachine(cfg=fsm_cfg)
    procedure: Procedure = fsm.run(features, task_name=task_name, video_file=video_path)

    print(f"  Detected {len(procedure.steps)} steps:")
    for idx, step in enumerate(procedure.steps, start=1):
        print(
            f"    {idx:02d}. {step.label.value:16s} "
            f"{step.start_time:5.2f} -> {step.end_time:5.2f}  conf={step.confidence:.2f}"
        )

    # Save steps as CSV
    steps_csv_path = os.path.join(output_dir, "steps.csv")
    save_steps_csv(procedure, steps_csv_path)

    # Save event log JSON (LLM socket)
    process_log_path = os.path.join(output_dir, "process_log.json")
    save_event_log_json(procedure, process_log_path)
    print(f"Event log (LLM socket) saved to: {process_log_path}")

    # --- 3. Instructions (template or DeepSeek) ----------------------------
    print("Step 3/4: generating instructions...")

    instruction_mode = args.instruction_mode

    # If DeepSeek is selected but the API key is missing, fall back to templates
    if instruction_mode == "deepseek":
        api_key_env = args.deepseek_api_key_env
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            print(
                f"DeepSeek selected but no API key found in env var {api_key_env}. "
                "Falling back to template instructions."
            )
            instruction_mode = "template"

    if instruction_mode == "deepseek":
        instructions_path = os.path.join(output_dir, "instructions_llm.txt")
    else:
        instructions_path = os.path.join(output_dir, "instructions_template.txt")

    instruction_map = save_instructions_text(
        procedure,
        instructions_path,
        mode=instruction_mode,
        deepseek_model=args.deepseek_model,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_api_key_env=args.deepseek_api_key_env,
        deepseek_temperature=args.deepseek_temperature,
        deepseek_max_tokens=args.deepseek_max_tokens,
    )



    # --- 4. Visualizations -------------------------------------------------
    print("Step 4/4: generating visualizations...")

    if save_plots:
        timeline_png = os.path.join(output_dir, "timeline.png")
        features_png = os.path.join(output_dir, "features.png")
        plot_action_timeline(procedure, out_path=timeline_png, show=False)
        plot_features_with_steps(features, procedure, out_path=features_png, show=False)
        print(f"  Timeline plot saved to:  {timeline_png}")
        print(f"  Features plot saved to:  {features_png}")
    else:
        print("  Skipping plots (per --no-plots).")

    if make_annotated_video:
        annotated_path = os.path.join(output_dir, "annotated.mp4")
        write_annotated_video(
            video_path,
            procedure,
            annotated_path,
            step_instructions=instruction_map,
            preserve_audio=True,
        )
    else:
        print("  Skipping annotated video (enable with --annotate-video).")


    print("Done.")


if __name__ == "__main__":
    main()
