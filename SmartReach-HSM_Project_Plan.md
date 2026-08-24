# SmartReach-HSM Project Plan

## 1. System Goal

Build a training-free, voice-driven assistive grasping system for a fixed tabletop scene. The user speaks a request such as `help me grab the coke can in front`, the system parses the command, triggers RGB camera screenshots, extracts exactly three same-source visual modalities from each screenshot, and outputs real-time audio guidance to help the user reach and grasp the target object.

The system is designed for a course project demo, so the priority order is:

1. Stable end-to-end behavior.
2. Clear and explainable multimodal fusion.
3. Low engineering risk.
4. No training or fine-tuning.

## 2. Hard Constraints

### 2.1 Functional Constraints

- The interaction starts from user voice input.
- An LLM parses the voice request and controls screenshot-based camera capture.
- The perception stage uses exactly three same-source visual modalities extracted from the same RGB screenshot:
  - monocular depth map
  - object detection result
  - hand pose estimation result
- The system helps the user grasp an object through audio guidance.

### 2.2 Technical Constraints

- All perception models must be open source.
- All perception models must be inference-only.
- No training, fine-tuning, distillation, or custom model fitting is allowed.
- No extra sensor is allowed in the core design.
- The core visual input source is one RGB camera.
- The multimodal fusion must be rule-based and geometric, not learned.

### 2.3 Demo Constraints

- Use a fixed tabletop scene.
- Use a fixed camera position.
- Prefer a small set of known target objects.
- Support one active target per grasping session.
- Support one active hand per session.

## 3. Recommended Model Choices

### 3.1 Voice and Control Layer

- ASR: `FunASR Paraformer-zh` or `Whisper small`
- Intent parsing LLM: `Qwen2.5-1.5B-Instruct` or `Qwen2.5-3B-Instruct`
- Audio output: `Piper TTS` or pre-recorded short command clips

Reasoning:

- ASR and LLM belong to the interaction layer and do not count as the three visual modalities.
- A local small LLM is sufficient because the task is structured command parsing, not open-ended reasoning.
- Short command playback is more stable than long-form TTS.

### 3.2 Three Visual Modalities

#### A. Monocular Depth Estimation

- Model: `Depth Anything V2-Small`
- Input: RGB screenshot
- Output: dense relative depth map
- Why:
  - open source
  - strong zero-shot depth estimation
  - no fine-tuning needed
  - suitable for relative forward/back guidance

#### B. Object Detection

- Model: `Grounding DINO`
- Input: RGB screenshot plus text prompt from the LLM parser
- Output: target bounding box candidates with scores
- Why:
  - open-vocabulary detection suits voice-specified targets
  - no retraining needed
  - works better than closed-set COCO labels for objects such as `coke can` or `milk carton`

#### C. Hand Pose Estimation

- Model: `MediaPipe Hands`
- Input: RGB screenshot
- Output: 21 hand landmarks, handedness, confidence
- Why:
  - open source
  - high speed
  - no training needed
  - directly supports grasp guidance through palm-center extraction

## 4. Runtime Pipeline

## 4.1 End-to-End Flow

1. User speaks the grasp request.
2. ASR converts speech to text.
3. LLM parses the text into a structured command.
4. Camera controller runs an initialization screenshot burst.
5. Frame selector picks the sharpest frame.
6. Grounding DINO detects the requested target.
7. If the target is found, the system starts a screenshot guidance session.
8. During the session, periodic screenshots are processed by:
   - Grounding DINO
   - Depth Anything V2
   - MediaPipe Hands
9. Fusion engine computes the hand-target relation.
10. State machine converts the relation into one audio command at a time.
11. Session ends on grasp success, timeout, target loss, or user stop command.

## 4.2 Why Screenshot Loop Instead of Single Image

Single-image guidance is not enough for grasping because hand position changes after each instruction. The final design therefore uses:

- initialization burst screenshots for target acquisition
- periodic screenshots every 300 to 500 ms during guidance

This still matches the user-requested interaction style because the LLM controls screenshot triggering rather than using a continuous high-rate video model loop.

## 5. Structured Command Schema

The LLM parser must output a JSON-compatible object with this exact schema:

```json
{
  "action": "grasp",
  "target_name_raw": "coke",
  "target_prompt_en": "red soda can",
  "spatial_hint": "front-center",
  "session_mode": "capture_and_guide",
  "need_confirmation": false,
  "confirmation_question": ""
}
```

### Field Rules

- `action`: one of `grasp`, `stop`, `restart`, `unknown`
- `target_name_raw`: raw object name from ASR output
- `target_prompt_en`: English prompt for Grounding DINO
- `spatial_hint`: one of `left`, `right`, `front`, `center`, `front-left`, `front-right`, `front-center`, `none`
- `session_mode`: always `capture_and_guide` for this project
- `need_confirmation`: true when the command is ambiguous
- `confirmation_question`: filled only if `need_confirmation` is true

## 6. Module Interfaces

This section is mandatory for parallel development.

### 6.1 `src/audio/asr_service.py`

Purpose:

- capture microphone input
- stop on silence or stop keyword
- return transcription result

Primary interface:

```python
class AsrResult(TypedDict):
    text: str
    confidence: float
    latency_ms: int

def listen_once() -> AsrResult:
    ...
```

Error behavior:

- return empty text on failure
- confidence must be `0.0` on failure

QA target:

- test with 10 fixed spoken commands
- transcription success >= 80 percent on demo commands

### 6.2 `src/llm/intent_parser.py`

Purpose:

- parse ASR text into `CommandSpec`
- normalize the target object phrase
- map spatial hints

Primary interface:

```python
class CommandSpec(TypedDict):
    action: str
    target_name_raw: str
    target_prompt_en: str
    spatial_hint: str
    session_mode: str
    need_confirmation: bool
    confirmation_question: str

def parse_intent(text: str) -> CommandSpec:
    ...
```

Rules:

- If no object is found, return `action="unknown"`.
- If stop intent is detected, return `action="stop"`.
- If target phrase is ambiguous, set `need_confirmation=true`.

QA target:

- unit test at least 15 command variants
- all expected fields present in every return object

### 6.3 `src/camera/camera_service.py`

Purpose:

- open and close camera
- capture initialization burst
- capture periodic session frames

Primary interface:

```python
class FramePacket(TypedDict):
    frame_id: int
    timestamp_ms: int
    rgb_image: object
    sharpness_score: float

def capture_burst(n: int = 3) -> list[FramePacket]:
    ...

def capture_once() -> FramePacket:
    ...
```

Rules:

- all frames must include `timestamp_ms`
- `sharpness_score` must be computed for burst selection

QA target:

- burst capture returns exactly `n` frames
- selected best frame is deterministic for a fixed fixture

### 6.4 `src/camera/frame_selector.py`

Purpose:

- choose the sharpest frame from burst capture

Interface:

```python
def select_sharpest(frames: list[FramePacket]) -> FramePacket:
    ...
```

QA target:

- verify Laplacian-based or equivalent sharpness ranking with recorded fixtures

### 6.5 `src/perception/detector_grounding_dino.py`

Purpose:

- detect target objects from RGB screenshots using text prompt

Primary interface:

```python
class Detection(TypedDict):
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    score: float

def detect(rgb_image: object, prompt: str) -> list[Detection]:
    ...
```

Rules:

- coordinates must be in original image pixel space
- boxes must be clipped to image bounds
- results must be sorted by descending score before post-processing

QA target:

- test at least 10 fixture images
- detection must return the intended target in the top 3 candidates for each valid fixture

### 6.6 `src/perception/depth_anything.py`

Purpose:

- estimate a dense relative depth map

Primary interface:

```python
class DepthResult(TypedDict):
    depth_map: object
    min_depth: float
    max_depth: float

def predict_depth(rgb_image: object) -> DepthResult:
    ...
```

Rules:

- resize or remap output to original image size before returning
- normalize depth consistently across frames if session filtering expects comparable values

QA target:

- verify output size matches input size
- verify center-object depth is numerically different from far background on known fixtures

### 6.7 `src/perception/hand_pose_mediapipe.py`

Purpose:

- estimate hand landmarks

Primary interface:

```python
class HandPoseResult(TypedDict):
    hand_present: bool
    score: float
    handedness: str
    landmarks_xy: list[tuple[float, float]]

def estimate_hand(rgb_image: object) -> HandPoseResult:
    ...
```

Rules:

- coordinates must be converted to image pixel space
- if no hand exists, return `hand_present=false` and an empty landmark list

QA target:

- verify stable detection on recorded approach motions
- no crash on empty-hand scenes

### 6.8 `src/fusion/hand_reference.py`

Purpose:

- convert 21 landmarks into one stable hand reference point

Primary interface:

```python
def compute_palm_center(landmarks_xy: list[tuple[float, float]]) -> tuple[float, float]:
    ...
```

Recommended formula:

- average of wrist, index MCP, middle MCP, ring MCP, pinky MCP

QA target:

- palm center movement should be smoother than fingertip movement on the same sequence

### 6.9 `src/fusion/depth_sampler.py`

Purpose:

- sample depth around target center and palm center

Primary interface:

```python
def sample_box_center_depth(depth_map: object, bbox_xyxy: tuple[float, float, float, float]) -> float:
    ...

def sample_point_depth(depth_map: object, point_xy: tuple[float, float]) -> float:
    ...
```

Rules:

- use a center crop for the target box, not the full box average
- use median depth to reduce outliers
- return `nan` if sampling is invalid

QA target:

- invalid depth areas must not crash the pipeline

### 6.10 `src/fusion/target_selector.py`

Purpose:

- select and lock the active target for the session

Primary interface:

```python
class TargetSelectionResult(TypedDict):
    found: bool
    detection: Detection | None
    match_score: float

def select_target(
    detections: list[Detection],
    spatial_hint: str,
    image_size: tuple[int, int],
    previous_box: tuple[float, float, float, float] | None
) -> TargetSelectionResult:
    ...
```

Scoring rule:

- detection score
- spatial hint match
- temporal stability with previous box

QA target:

- the selected target should remain stable under brief hand occlusion in test clips

### 6.11 `src/fusion/fusion_state.py`

Purpose:

- create one structured multimodal state from detection, depth, and hand pose

Primary interface:

```python
class FusionState(TypedDict):
    target_visible: bool
    hand_visible: bool
    target_locked: bool
    target_center_xy: tuple[float, float] | None
    hand_center_xy: tuple[float, float] | None
    target_depth: float | None
    hand_depth: float | None
    dx_norm: float | None
    dy_norm: float | None
    dz_rel: float | None
    target_confidence: float
    hand_confidence: float
    frame_id: int
    timestamp_ms: int
```

Computation rules:

- `dx_norm = (u_t - u_h) / image_width`
- `dy_norm = (v_t - v_h) / image_height`
- `dz_rel = d_t - d_h`

QA target:

- state object must be complete and type-consistent even when target or hand is missing

### 6.12 `src/fusion/temporal_filter.py`

Purpose:

- smooth state over time and prevent command oscillation

Primary interface:

```python
def smooth_state(current: FusionState, previous: FusionState | None) -> FusionState:
    ...
```

Recommended method:

- EMA on `dx_norm`, `dy_norm`, `dz_rel`
- detection persistence counters
- hand visibility persistence counters

QA target:

- state variance on a static scene should be lower after filtering than before filtering

### 6.13 `src/control/state_machine.py`

Purpose:

- convert `FusionState` into a discrete control state

Primary interface:

```python
class ControlState(TypedDict):
    name: str
    stable_frames: int
    target_lost_frames: int
    hand_lost_frames: int

def next_state(prev_state: ControlState | None, fusion_state: FusionState, cfg: dict) -> ControlState:
    ...
```

QA target:

- unit tests must cover every transition path in the transition table below

### 6.14 `src/control/command_policy.py`

Purpose:

- choose one audio command from the current state and filtered offsets

Primary interface:

```python
class GuidanceCommand(TypedDict):
    command: str
    reason: str
    should_play: bool

def decide_command(control_state: ControlState, fusion_state: FusionState, cfg: dict) -> GuidanceCommand:
    ...
```

Rules:

- one command at a time
- command changes require debounce
- `grasp` only after stable close-range conditions

### 6.15 `src/control/safety_guard.py`

Purpose:

- override unstable or unsafe commands

Interface:

```python
def enforce_safety(command: GuidanceCommand, fusion_state: FusionState, cfg: dict) -> GuidanceCommand:
    ...
```

Rules:

- if target is missing, do not output directional commands
- if hand is missing, do not output depth approach commands
- prefer `hold`, `target lost`, or `hand lost` over guessing

### 6.16 `src/audio/command_player.py`

Purpose:

- play short guidance commands

Interface:

```python
def play_if_changed(command: str, timestamp_ms: int) -> bool:
    ...
```

Rules:

- obey debounce interval
- log played commands for evaluation

### 6.17 `src/eval/metrics.py`

Purpose:

- compute project evaluation metrics

Required metrics:

- target acquisition rate
- guidance success rate
- time to first valid instruction
- time to grasp
- average command latency
- target lost frequency
- hand lost frequency
- wrong-command count by manual annotation

## 7. State Machine Specification

### 7.1 States

- `SEARCH_TARGET`
- `WAIT_HAND`
- `APPROACH`
- `ALIGN`
- `GRASP_READY`
- `LOST`

### 7.2 Transition Table

| Current State | Guard Condition | Next State | Action |
|---|---|---|---|
| any | user says stop | LOST | play `hold` then end session |
| SEARCH_TARGET | target visible for `target_found_frames` | WAIT_HAND | play `target found` if enabled |
| SEARCH_TARGET | timeout reached | LOST | play `target lost` |
| WAIT_HAND | hand visible for `hand_found_frames` | APPROACH | play `place hand in view` only once before transition |
| WAIT_HAND | target missing for `target_lost_frames` | SEARCH_TARGET | play `target lost` |
| APPROACH | target or hand missing | LOST | play `target lost` or `hand lost` |
| APPROACH | `abs(dx_norm) < align_x` and `abs(dy_norm) < align_y` | ALIGN | keep guiding |
| ALIGN | target or hand missing | LOST | play loss command |
| ALIGN | `abs(dx_norm) < grasp_x` and `abs(dy_norm) < grasp_y` and `abs(dz_rel) < grasp_z` for `grasp_stable_frames` | GRASP_READY | play `grasp` |
| GRASP_READY | command emitted and session success confirmed | LOST | end session success |
| LOST | restart command received | SEARCH_TARGET | reset counters |

### 7.3 Command Policy by State

#### SEARCH_TARGET

- command: `target lost` or `scan slowly`

#### WAIT_HAND

- command: `place hand in view`

#### APPROACH

- choose the largest absolute error among `dx_norm`, `dy_norm`, `dz_rel`
- output one of:
  - `left`
  - `right`
  - `up`
  - `down`
  - `forward`
  - `back`

#### ALIGN

- prefer fine lateral correction first
- use `hold` when within deadband but not yet grasp-ready

#### GRASP_READY

- output `grasp`

#### LOST

- output `hold`, `target lost`, or `hand lost`

## 8. Configuration Parameters

All parameters must be externalized into YAML.

Example `configs/thresholds.yaml`:

```yaml
capture_interval_ms: 400
command_debounce_ms: 600
target_found_frames: 2
hand_found_frames: 2
target_lost_frames: 5
hand_lost_frames: 4
grasp_stable_frames: 4
deadband_x: 0.08
deadband_y: 0.08
align_x: 0.04
align_y: 0.04
grasp_x: 0.03
grasp_y: 0.03
grasp_z: 0.06
burst_capture_count: 3
session_timeout_s: 30
```

Example `configs/prompts.yaml`:

```yaml
object_map:
  coke:
    - red soda can
    - coke can
    - cola can
  milk_box:
    - milk carton
    - drink carton
  green_bottle:
    - green bottle
    - plastic bottle
```

## 9. Repo Structure

```text
project/
  README.md
  requirements.txt
  configs/
    app.yaml
    prompts.yaml
    thresholds.yaml
  assets/
    audio_prompts/
  src/
    main.py
    app/
      orchestrator.py
      session_manager.py
      capture_controller.py
    audio/
      asr_service.py
      tts_service.py
      command_player.py
    llm/
      intent_parser.py
      prompt_templates.py
    camera/
      camera_service.py
      frame_selector.py
    perception/
      detector_grounding_dino.py
      depth_anything.py
      hand_pose_mediapipe.py
    fusion/
      target_selector.py
      hand_reference.py
      depth_sampler.py
      fusion_state.py
      temporal_filter.py
    control/
      state_machine.py
      command_policy.py
      safety_guard.py
    ui/
      overlay.py
    eval/
      metrics.py
      run_eval.py
  tests/
    test_intent_parser.py
    test_target_selector.py
    test_fusion_state.py
    test_state_machine.py
```

## 10. Developer Task Book

Each task must include owner, dependencies, code deliverable, tests first, acceptance gate, and QA scenario.

### 10.1 Developer A: Voice and Session Orchestration

Owner:

- Developer A

Scope:

- `src/audio/asr_service.py`
- `src/llm/intent_parser.py`
- `src/app/session_manager.py`
- `src/app/orchestrator.py`

Dependencies:

- none at the start

Deliverables:

- ASR wrapper
- structured command parser
- session start, stop, restart flow

Tests first:

- `tests/test_intent_parser.py`
- mock-based tests for stop and restart behavior

Acceptance gate:

- command `help me grab the coke can in front` becomes valid `CommandSpec`
- stop command ends the session cleanly

QA scenario:

- Tool: unit tests plus one recorded voice sample
- Input: 10 spoken command variants
- Expected:
  - parser returns `action="grasp"` for valid grasp commands
  - parser returns `action="stop"` for stop commands
  - no missing fields in output schema

### 10.2 Developer B: Camera and Detection

Owner:

- Developer B

Scope:

- `src/camera/camera_service.py`
- `src/camera/frame_selector.py`
- `src/perception/detector_grounding_dino.py`
- `src/fusion/target_selector.py`

Dependencies:

- `CommandSpec.target_prompt_en`

Deliverables:

- burst capture
- periodic screenshot capture
- text-conditioned object detection
- active target selection and lock

Tests first:

- frame selector tests using stored images
- target selector tests with synthetic detection fixtures

Acceptance gate:

- target is found in at least 8 of 10 fixture scenes
- target lock survives short occlusion in replayed test sequence

QA scenario:

- Tool: replay script on recorded tabletop scenes
- Input: scenes with bottle, can, carton
- Expected:
  - target box returned in original pixel coordinates
  - highest-ranked target matches spatial hint when multiple candidates exist

### 10.3 Developer C: Depth, Hand Pose, and Fusion

Owner:

- Developer C

Scope:

- `src/perception/depth_anything.py`
- `src/perception/hand_pose_mediapipe.py`
- `src/fusion/hand_reference.py`
- `src/fusion/depth_sampler.py`
- `src/fusion/fusion_state.py`
- `src/fusion/temporal_filter.py`

Dependencies:

- camera frame format
- detection output schema

Deliverables:

- depth inference wrapper
- hand pose wrapper
- palm center extraction
- sampled target and hand depth
- filtered fusion state

Tests first:

- fixture-based hand center tests
- depth sampling tests on canned depth arrays
- schema tests for `FusionState`

Acceptance gate:

- complete `FusionState` emitted for valid scenes
- no crash when hand or target is missing
- filtered offsets visually smoother than raw offsets on test replay

QA scenario:

- Tool: recorded approach sequence
- Input: user hand approaching target on tabletop
- Expected:
  - palm center tracks continuously
  - sampled hand depth and target depth differ in the expected direction
  - `dx_norm`, `dy_norm`, `dz_rel` change monotonically in simple motions

### 10.4 Developer D: Control, Audio, Evaluation, and Demo Tools

Owner:

- Developer D

Scope:

- `src/control/state_machine.py`
- `src/control/command_policy.py`
- `src/control/safety_guard.py`
- `src/audio/command_player.py`
- `src/eval/metrics.py`
- `src/ui/overlay.py`

Dependencies:

- `FusionState`

Deliverables:

- state machine
- command debounce logic
- safety overrides
- metrics computation
- visual debug overlay

Tests first:

- full transition tests for every state path
- command debounce tests
- metric computation tests on fixture logs

Acceptance gate:

- all transition tests pass
- command stream does not oscillate excessively on a static scene
- `grasp` only appears after stable close-range condition

QA scenario:

- Tool: replay of filtered `FusionState` logs
- Input: scripted sequences covering target loss, hand loss, and successful grasp
- Expected:
  - correct state transitions
  - correct final audio commands
  - no illegal command in loss states

## 11. Milestones with Acceptance Gates

### M1. Voice Request to Target Detection

Goal:

- user command becomes parsed target prompt and initial target box

Required completion:

- ASR works on demo commands
- parser outputs valid `CommandSpec`
- burst capture works
- detector finds target on selected initialization frame

Acceptance gate:

- end-to-end dry run succeeds for 3 example objects

QA:

- Run one scripted demo command per object
- Pass if the system reaches `WAIT_HAND`

### M2. Three-Modality Extraction on One Screenshot

Goal:

- one screenshot produces target box, depth map, hand landmarks, and complete `FusionState`

Acceptance gate:

- stored fixture images generate valid outputs with no crash

QA:

- run all fixture screenshots
- pass if each fixture emits a valid `FusionState` JSON dump

### M3. Screenshot Guidance Loop

Goal:

- periodic screenshot session outputs stable direction commands

Acceptance gate:

- replay sequences produce sensible command progression such as `right -> forward -> hold -> grasp`

QA:

- use recorded tabletop motion clips
- pass if command history matches manual expectation within tolerance

### M4. Stable Live Demo Build

Goal:

- complete live scenario from voice command to grasp cue

Acceptance gate:

- 3 out of 5 live trials succeed in the same room setup

QA:

- run five trials on two object types
- collect timing and failure notes

## 12. Evaluation Protocol

### 12.1 Quantitative Metrics

- `Target Acquisition Rate`
  - successful initial target detection sessions / total sessions
- `Guidance Success Rate`
  - sessions reaching `GRASP_READY` / total valid sessions
- `Time to First Valid Instruction`
  - time from command end to first directional audio command
- `Time to Grasp`
  - time from session start to `grasp`
- `Average Command Latency`
  - mean time from screenshot timestamp to audio playback timestamp
- `Target Lost Frequency`
  - average count per session
- `Hand Lost Frequency`
  - average count per session

### 12.2 Qualitative Review

- Was the target prompt correctly understood?
- Was the chosen object correct when multiple objects were present?
- Were commands stable or jittery?
- Did `grasp` occur too early or too late?
- Were fallback messages understandable?

### 12.3 Required Test Matrix

| Dimension | Cases |
|---|---|
| Object type | coke can, green bottle, milk carton |
| Position | left, center, right |
| Distance | near, mid |
| Hand entry | from front, from right |
| Scene complexity | single target, multi-object |

Minimum evaluation volume:

- 3 objects
- 2 positions each
- 5 trials each
- total minimum 30 trials

## 13. Demo Script

### Demo Case 1

User says:

- `help me grab the coke can in front`

Expected system behavior:

1. ASR transcribes command.
2. LLM returns `target_prompt_en="red soda can"`.
3. Camera burst captures the scene.
4. Grounding DINO finds the coke can.
5. Audio says target found or directly starts guidance.
6. User moves hand according to commands.
7. System outputs a sequence such as:
   - `right`
   - `forward`
   - `hold`
   - `grasp`

### Demo Case 2

User says:

- `help me grab the milk carton on the right`

Expected behavior:

- target selection follows the right-side bias from `spatial_hint`
- guidance continues until `grasp`

### Demo Rules

- Keep the camera fixed.
- Use matte objects.
- Avoid duplicate instances of the same target prompt during live demo.
- Keep one hand visible once the session enters `WAIT_HAND`.
- Use pre-checked room lighting.

## 14. Report Outline

The course report must stay within the assignment formatting constraints. Recommended structure:

1. Introduction
   - problem setting
   - motivation for assistive grasping
2. System Constraints and Design Goal
   - same-source trimodal design
   - training-free requirement
3. System Architecture
   - voice interaction layer
   - screenshot control
   - three-modality perception
   - fusion and state machine
4. Methodology
   - command parsing
   - target detection
   - hand pose estimation
   - monocular depth estimation
   - multimodal fusion logic
5. Implementation
   - repo structure
   - runtime pipeline
   - command policy
6. Experiments and Evaluation
   - test setup
   - metrics
   - results table
   - failure case analysis
7. Discussion
   - strengths
   - limitations
   - future work without violating project scope
8. Team Contributions

## 15. Risks and Mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| ASR mishears target | wrong prompt sent to detector | restrict demo vocabulary and support confirmation prompt |
| Open-vocabulary detector false positives | wrong target selected | use spatial hint, confidence threshold, and temporal lock |
| Depth is unstable on reflective objects | wrong forward/back guidance | avoid reflective objects and use median center sampling |
| Hand pose missing during fast motion | guidance interruption | use lower screenshot interval and visible-hand reminder |
| Commands oscillate | poor usability | use EMA smoothing and command debounce |
| Target occluded by hand | target lost mid-session | keep target lock for short gaps |
| Latency too high | user overshoots | reduce screenshot rate, shorten command set, use pre-recorded audio |

## 16. Atomic Commit Strategy

Even if git is added later, commits should map to testable milestones.

Recommended commit order:

1. `add command parsing schema and parser tests`
2. `add camera burst capture and frame selection`
3. `integrate grounding dino detection and target selection`
4. `integrate depth anything inference wrapper`
5. `integrate mediapipe hands and palm center extraction`
6. `add fusion state generation and filtering`
7. `implement state machine and command policy tests`
8. `add audio command playback and debounce control`
9. `add evaluation metrics and replay scripts`
10. `polish live demo overlay and configuration tuning`

## 17. Immediate Execution Checklist

This is the direct start list for the team.

### Day 1

- create repo structure
- install and pin dependencies
- verify camera capture works
- verify ASR and parser return `CommandSpec`

### Day 2

- integrate Grounding DINO on saved fixtures
- integrate Depth Anything V2 on saved fixtures
- integrate MediaPipe Hands on saved fixtures

### Day 3

- implement `FusionState`
- implement target lock
- implement depth sampling
- implement palm center extraction

### Day 4

- implement state machine
- implement command debounce
- implement audio playback

### Day 5

- run screenshot-loop replay tests
- tune thresholds in YAML
- fix transition edge cases

### Day 6

- run live trials
- collect metrics
- capture screenshots and overlay images for report

### Day 7

- finalize demo script
- finalize report figures
- freeze configuration for presentation

## 18. Definition of Done

The project is considered ready for demo when all conditions below are met:

- Voice command is parsed into a valid structured target request.
- Initialization burst capture acquires a usable frame.
- The requested target is detected in the initialization phase.
- The screenshot guidance loop produces three visual modalities on every valid session step.
- The fusion layer emits a valid `FusionState` with no crashes on missing-hand or missing-target frames.
- The state machine passes all unit transition tests.
- The command player respects debounce and does not chatter on stable scenes.
- The live system succeeds in at least 3 out of 5 full tabletop trials.
- Evaluation metrics can be exported for report writing.
