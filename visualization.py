
"""
Visualization helpers for the pot lid assembly prototype.

Contains:
- A timeline plot of detected actions over time.
- Feature curves (motion, rotation, audio) with step boundaries.
- Optional annotated video with overlayed action labels.

Optional and not required for the core recognition pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from models import FrameFeatures, Procedure, ActionLabel




_LABEL_COLORS: Dict[ActionLabel, Tuple[float, float, float]] = {
    ActionLabel.IDLE: (0.85, 0.85, 0.85),

    ActionLabel.LID_REORIENT: (0.75, 0.85, 1.00),
    ActionLabel.PICK_WASHER: (0.70, 0.90, 0.90),
    ActionLabel.SEAT_WASHER: (0.45, 0.85, 0.70),

    ActionLabel.PICK_SCREW: (0.90, 0.85, 0.55),
    ActionLabel.INSERT_SCREW: (0.95, 0.75, 0.45),

    ActionLabel.PICK_HANDLE: (0.90, 0.75, 0.25),
    ActionLabel.INSERT_HANDLE: (0.90, 0.65, 0.20),
    ActionLabel.HAND_TIGHTEN_HANDLE: (0.85, 0.55, 0.15),

    ActionLabel.PLACE_LID_FOR_TOOL: (0.90, 0.60, 0.60),
    ActionLabel.PICK_SCREWDRIVER: (0.95, 0.55, 0.80),
    ActionLabel.TOOL_SEAT: (0.85, 0.50, 0.90),
    ActionLabel.TOOL_TIGHTEN: (0.65, 0.40, 0.95),
    ActionLabel.SET_TOOL_DOWN: (0.75, 0.75, 0.95),

    ActionLabel.FINAL_FLIP_AND_PLACE: (0.60, 0.90, 0.60),
    ActionLabel.RELEASE: (0.70, 0.70, 0.90),
}


def _color_for_label(label: ActionLabel) -> Tuple[float, float, float]:
    return _LABEL_COLORS.get(label, (0.8, 0.8, 0.8))




def plot_action_timeline(
    procedure: Procedure,
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot a readable action timeline: one row per step.
    """
    if not procedure.steps:
        print("plot_action_timeline: no steps to plot.")
        return

    n = len(procedure.steps)
    fig_h = max(2.5, 0.35 * n + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    ax.set_title("Detected action timeline")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Steps")

    y_ticks = []
    y_labels = []

    for i, step in enumerate(procedure.steps):
        start = float(step.start_time)
        duration = float(max(0.001, step.end_time - step.start_time))
        color = _color_for_label(step.label)

        y = i
        ax.broken_barh(
            [(start, duration)],
            (y - 0.35, 0.7),
            facecolors=color,
            edgecolors="k",
            linewidth=0.5,
        )

        y_ticks.append(y)
        y_labels.append(f"{i + 1:02d} {step.label.value}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_ylim(-1, n)
    ax.invert_yaxis()

    ax.grid(axis="x", linestyle="--", alpha=0.3)

    start_times = [s.start_time for s in procedure.steps]
    end_times = [s.end_time for s in procedure.steps]
    ax.set_xlim(min(start_times), max(end_times))

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)





def plot_features_with_steps(
    features: List[FrameFeatures],
    procedure: Optional[Procedure] = None,
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot motion magnitude, rotation index, and audio RMS over time,
    optionally overlaying vertical lines where steps begin.

    Parameters
    ----------
    features : list[FrameFeatures]
        The per-frame features from perception.
    procedure : Procedure or None
        If given, step boundaries and labels are shown.
    out_path : str or None
        If provided, save the figure here.
    show : bool
        If True, display the figure.
    """
    if not features:
        print("plot_features_with_steps: no features to plot.")
        return

    t = np.array([f.t for f in features], dtype=np.float32)
    motion = np.array([f.motion_mag for f in features], dtype=np.float32)
    rot = np.array([f.rot_index for f in features], dtype=np.float32)
    audio = np.array([f.audio_rms for f in features], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

    ax0, ax1, ax2 = axes

    ax0.plot(t, motion)
    ax0.set_ylabel("Motion\nmagnitude")

    ax1.plot(t, rot)
    ax1.set_ylabel("Rotation\nindex")

    ax2.plot(t, audio)
    ax2.set_ylabel("Audio\nRMS")
    ax2.set_xlabel("Time (seconds)")

    if procedure is not None and procedure.steps:
        for step in procedure.steps:
            for ax in axes:
                ax.axvline(step.start_time, color="k", linestyle="--", alpha=0.2)
            ax2.text(
                step.start_time,
                0.0,
                step.label.value,
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=6,
                alpha=0.6,
            )

    fig.suptitle("Per-frame features with detected step boundaries", y=0.98)
    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)




def _find_step_at_time(procedure: Procedure, t: float) -> Optional[int]:
    """
    Return the index of the step active at time t, or None if t is outside.
    """
    for idx, step in enumerate(procedure.steps):
        if step.start_time <= t <= step.end_time:
            return idx
    return None

def _wrap_text_to_width(
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    max_width_px: int,
    max_lines: int = 2,
) -> List[str]:
    """
    Word-wrap text to fit within a pixel width for cv2 overlays.
    Limits to max_lines and adds "..." if truncated.
    """
    words = (text or "").strip().split()
    if not words:
        return []

    lines: List[str] = []
    cur = ""

    for w in words:
        candidate = (cur + " " + w).strip()
        (tw, _th), _ = cv2.getTextSize(candidate, font, font_scale, thickness)

        if tw <= max_width_px or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = w

        if len(lines) >= max_lines:
            break

    if len(lines) < max_lines and cur:
        lines.append(cur)

    used_words = sum(len(s.split()) for s in lines)
    if used_words < len(words) and lines:
        last = lines[-1]
        if not last.endswith("..."):
            lines[-1] = last.rstrip(".") + "..."

    return lines[:max_lines]


def write_annotated_video(
    video_path: str,
    procedure: Procedure,
    out_path: str,
    font_scale: float = 0.7,
    thickness: int = 2,
    step_instructions: Optional[Dict[int, str]] = None,
    preserve_audio: bool = True,
    max_instruction_lines: int = 2,
) -> None:
    """
    Create a copy of the video with overlayed action labels and short instructions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"write_annotated_video: could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base, ext = os.path.splitext(out_path)
    silent_path = base + ".silent" + (ext if ext else ".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (width, height))

    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    margin = 18
    box_w = min(width - 2 * margin, 980)
    max_text_w = box_w - 2 * margin

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        step_index = _find_step_at_time(procedure, t)

        if step_index is not None:
            step = procedure.steps[step_index]

            label_text = f"Step {step_index + 1}: {step.label.value}"
            time_text = f"{step.start_time:.1f}–{step.end_time:.1f} s   conf={step.confidence:.2f}"

            instr_lines: List[str] = []
            if step_instructions is not None:
                instr = step_instructions.get(step_index, "")
                if instr:
                    instr_lines = _wrap_text_to_width(
                        instr,
                        font=font,
                        font_scale=font_scale * 0.9,
                        thickness=max(1, thickness - 1),
                        max_width_px=max_text_w,
                        max_lines=max_instruction_lines,
                    )

            line_h = int(28 * font_scale)
            header_lines = 2
            instr_count = len(instr_lines)
            total_lines = header_lines + instr_count
            box_h = margin + total_lines * line_h + margin

            x0, y0 = margin, margin
            x1, y1 = x0 + box_w, y0 + box_h

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)

            tx = x0 + margin
            ty = y0 + margin + line_h

            cv2.putText(frame, label_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            ty += line_h
            cv2.putText(
                frame,
                time_text,
                (tx, ty),
                font,
                font_scale * 0.85,
                (200, 200, 200),
                max(1, thickness - 1),
                cv2.LINE_AA,
            )
            ty += line_h

            for ln in instr_lines:
                cv2.putText(
                    frame,
                    ln,
                    (tx, ty),
                    font,
                    font_scale * 0.9,
                    (230, 230, 230),
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )
                ty += line_h

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if preserve_audio:
        try:
            try:
                from moviepy import VideoFileClip
            except Exception:
                from moviepy.editor import VideoFileClip

            src = VideoFileClip(video_path)
            silent = VideoFileClip(silent_path)

            if src.audio is None:
                if os.path.exists(out_path):
                    os.remove(out_path)
                os.replace(silent_path, out_path)
                print(f"Annotated video written to: {out_path} (source video has no audio track)")
                try:
                    src.close()
                except Exception:
                    pass
                try:
                    silent.close()
                except Exception:
                    pass
                return

            final = silent.set_audio(src.audio)

            if os.path.exists(out_path):
                os.remove(out_path)

            final.write_videofile(
                out_path,
                codec="libx264",
                audio_codec="aac",
                fps=silent.fps,
                verbose=False,
                logger=None,
            )

            for clip in (final, silent, src):
                try:
                    clip.close()
                except Exception:
                    pass

            try:
                os.remove(silent_path)
            except Exception:
                pass

            print(f"Annotated video (with audio) written to: {out_path}")
            return

        except Exception as e:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
                os.replace(silent_path, out_path)
            except Exception:
                pass
            print(f"Annotated video written to: {out_path} (audio not preserved: {e})")
            return

    if os.path.exists(out_path):
        os.remove(out_path)
    os.replace(silent_path, out_path)
    print(f"Annotated video written to: {out_path}")

