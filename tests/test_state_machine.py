from src.common.schemas import ControlState, FusionState
from src.control.command_policy import decide_command
from src.control.state_machine import next_state


CFG = {
    "align_x": 0.04,
    "align_y": 0.04,
    "grasp_x": 0.03,
    "grasp_y": 0.03,
    "grasp_z": 0.06,
    "deadband_x": 0.08,
    "deadband_y": 0.08,
    "invert_x_guidance": False,
    "grasp_stable_frames": 2,
    "target_found_frames": 2,
    "hand_found_frames": 2,
    "target_lost_frames": 3,
    "hand_lost_frames": 2,
}


def make_state(target_visible=True, hand_visible=True, dx=0.2, dy=0.0, dz=0.0):
    return FusionState(
        target_visible=target_visible,
        hand_visible=hand_visible,
        target_locked=target_visible,
        target_center_xy=(10.0, 10.0) if target_visible else None,
        hand_center_xy=(0.0, 10.0) if hand_visible else None,
        target_depth=0.7 if target_visible else None,
        hand_depth=0.5 if hand_visible else None,
        dx_norm=dx if target_visible and hand_visible else None,
        dy_norm=dy if target_visible and hand_visible else None,
        dz_rel=dz if target_visible and hand_visible else None,
        target_confidence=0.9,
        hand_confidence=0.9,
        frame_id=1,
        timestamp_ms=1,
    )


def test_search_target_when_target_missing() -> None:
    state = next_state(None, make_state(target_visible=False, hand_visible=False), CFG)
    assert state.name == "SEARCH_TARGET"


def test_wait_hand_when_hand_missing() -> None:
    state = next_state(None, make_state(target_visible=True, hand_visible=False), CFG)
    assert state.name == "WAIT_HAND"


def test_approach_and_command_direction() -> None:
    control = next_state(None, make_state(dx=0.2, dy=0.01, dz=0.01), CFG)
    command = decide_command(control, make_state(dx=0.2, dy=0.01, dz=0.01), CFG)
    assert control.name == "APPROACH"
    assert command.command == "right"


def test_approach_direction_inverts_when_configured() -> None:
    cfg = dict(CFG)
    cfg["invert_x_guidance"] = True
    control = next_state(None, make_state(dx=0.2, dy=0.01, dz=0.01), cfg)
    command = decide_command(control, make_state(dx=0.2, dy=0.01, dz=0.01), cfg)
    assert command.command == "left"


def test_grasp_ready_after_close_alignment() -> None:
    fusion = make_state(dx=0.01, dy=0.01, dz=0.01)
    control = next_state(ControlState(name="GRASP_READY", stable_frames=1), fusion, CFG)
    command = decide_command(control, fusion, CFG)
    assert control.name == "GRASP_READY"
    assert command.command == "grasp"


def test_brief_target_loss_keeps_previous_state() -> None:
    previous = ControlState(name="APPROACH", stable_frames=2, target_visible_frames=3)
    control = next_state(
        previous, make_state(target_visible=False, hand_visible=True), CFG
    )
    assert control.name == "APPROACH"
    assert control.target_lost_frames == 1


def test_search_target_requires_multiple_visible_frames_to_reacquire() -> None:
    previous = ControlState(name="SEARCH_TARGET", target_visible_frames=0)
    control = next_state(
        previous, make_state(target_visible=True, hand_visible=True), CFG
    )
    assert control.name == "SEARCH_TARGET"
    control = next_state(
        control, make_state(target_visible=True, hand_visible=True), CFG
    )
    assert control.name != "SEARCH_TARGET"


def test_brief_hand_loss_keeps_previous_state() -> None:
    previous = ControlState(name="ALIGN", stable_frames=2, hand_visible_frames=3)
    control = next_state(
        previous, make_state(target_visible=True, hand_visible=False), CFG
    )
    assert control.name == "ALIGN"
    assert control.hand_lost_frames == 1
