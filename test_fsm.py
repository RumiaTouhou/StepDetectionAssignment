import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import numpy as np

from perception import extract_frame_features
from context_fsm import ContextStateMachine

video_path = "20251205_203444.mp4"

feats = extract_frame_features(video_path)
print(f"Extracted {len(feats)} frame features")

fsm = ContextStateMachine()
procedure = fsm.run(feats, task_name="Pot Lid Assembly", video_file=video_path, debug=True)

print("\n--- Thresholds ---")
th = fsm.thresholds
if th is None:
    print("No thresholds computed.")
else:
    print(f"motion_idle_max        : {th.motion_idle_max:.5f}")
    print(f"motion_active_min      : {th.motion_active_min:.5f}")
    print(f"rot_center_high        : {th.rot_center_high:.5f}")
    print(f"rot_rim_high           : {th.rot_rim_high:.5f}")
    print(f"audio_rms_spike        : {th.audio_rms_spike:.5f}")
    print(f"audio_onset_spike      : {th.audio_onset_spike:.5f}")
    print(f"hand_to_center_close   : {th.hand_to_center_close:.5f}")

if hasattr(fsm, "_handle_baseline"):
    print("\n--- Handle table baseline (debug) ---")
    print(f"handle_baseline        : {getattr(fsm, '_handle_baseline', 0.0):.3f}")
    print(f"handle_enter_threshold : {getattr(fsm, '_handle_enter', 0.0):.3f}")
    print(f"handle_exit_threshold  : {getattr(fsm, '_handle_exit', 0.0):.3f}")

tool_scores = np.array([f.tool_score for f in feats], dtype=np.float32)
tool_center = np.array([getattr(f, "tool_center_score", 0.0) for f in feats], dtype=np.float32)
handle_score = np.array([getattr(f, "handle_table_score", 0.0) for f in feats], dtype=np.float32)

print("\n--- Feature ranges ---")
print(f"tool_score        min={tool_scores.min():.3f}  mean={tool_scores.mean():.3f}  max={tool_scores.max():.3f}")
print(f"tool_center_score min={tool_center.min():.3f}  mean={tool_center.mean():.3f}  max={tool_center.max():.3f}")
print(f"handle_table_score min={handle_score.min():.3f} mean={handle_score.mean():.3f} max={handle_score.max():.3f}")

print(f"\nDetected {len(procedure.steps)} steps")
print("\n--- Steps (with summaries) ---")
for i, step in enumerate(procedure.steps, start=1):
    fs = step.features_summary
    dur = step.end_time - step.start_time
    print(f"{i:02d} {step.label.value:20s} {step.start_time:5.2f} -> {step.end_time:5.2f}  dur={dur:5.2f}  conf={step.confidence:.2f}")

    print(
        "   "
        f"motion={fs.get('motion_mag_mean', 0.0):.5f}  "
        f"rot_c={fs.get('rot_center_mean', 0.0):.3f}  "
        f"rot_r={fs.get('rot_rim_mean', 0.0):.3f}  "
        f"onset_max={fs.get('audio_onset_max', 0.0):.3f}  "
        f"hand_center_min={fs.get('hand_to_center_min', -1.0):.3f}  "
        f"tool_max={fs.get('tool_score_max', 0.0):.3f}  "
        f"tool_center_max={fs.get('tool_center_score_max', 0.0):.3f}  "
        f"handle_mean={fs.get('handle_table_score_mean', 0.0):.3f}"
    )

print("\n--- Committed transitions ---")
for tr in fsm.debug_transitions:
    print(
        f"t={tr['t']:5.2f}  {tr['prev_state']} -> {tr['next_state']}  "
        f"reason={tr.get('reason','')}  "
        f"elapsed={tr['elapsed']:.2f}  min_dur={tr['min_dur']:.2f}  "
        f"parts_ctr={tr['parts_reach_counter']}  handle_ctr={tr['handle_reach_counter']}  tool_ctr={tr['tool_reach_counter']}  "
        f"onset_seen={int(tr['onset_seen_in_state'])}  flip_seen={int(tr['flip_seen_in_state'])}  flip_after={int(tr.get('flip_after_onset_seen', 0))}  "
        f"tool_score={tr['tool_score']:.2f}  tool_c={tr['tool_center_score']:.2f}  handle_s={tr['handle_table_score']:.2f}  "
        f"rot_c={tr['rot_center']:.2f}  rot_r={tr['rot_rim']:.2f}  motion={tr['motion_mag']:.5f}  "
        f"onset={tr['audio_onset']:.2f}  hand={tr['hand_to_center']:.3f}"
    )


print("\n--- First candidate transitions (candidate != state) ---")
seen = set()
for r in fsm.debug_frame_trace:
    if r["candidate"] != r["state"]:
        key = (r["state"], r["candidate"])
        if key in seen:
            continue
        seen.add(key)
        print(
            f"first {r['state']} -> {r['candidate']} at t={r['t']:.2f}  "
            f"elapsed={r['elapsed']:.2f}  min_dur={r['min_dur']:.2f}  "
            f"parts_ctr={r['parts_reach_counter']}  handle_ctr={r['handle_reach_counter']}  tool_ctr={r['tool_reach_counter']}  "
            f"tool_score={r['tool_score']:.2f}  tool_c={r['tool_center_score']:.2f}  handle_s={r['handle_table_score']:.2f}  "
            f"rot_c={r['rot_center']:.2f}  rot_r={r['rot_rim']:.2f}  motion={r['motion_mag']:.5f}  "
            f"onset={r['audio_onset']:.2f}  hand={r['hand_to_center']:.3f}"
        )
