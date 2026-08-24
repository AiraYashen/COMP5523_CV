# SmartReach-HSM

SmartReach-HSM is a multimodal assistive grasping prototype for visually impaired users. The system listens to a spoken query, captures the current camera frame, extracts three same-source visual modalities from the same RGB scene, builds a multimodal prompt, sends the prompt to a vision-language model, and finally speaks the answer back to the user.

The current project supports:

- live webcam input
- fixed 8-second Whisper-based speech capture
- Grounding DINO object detection
- Depth Anything monocular depth estimation
- MediaPipe hand pose estimation
- GPT-5.4 or GLM-based multimodal reasoning
- training-free GRPO style prompt optimization
- a real-time multimodal dashboard and a separate camera preview window

## Pipeline

The default runtime pipeline is:

1. Listen to user speech.
2. Capture the current RGB frame from the camera.
3. Run object detection, depth estimation, and hand pose estimation on the same scene.
4. Package RGB image, depth image, detection results, hand pose results, and fusion metadata into one prompt.
5. Send the prompt to the multimodal model.
6. Convert the model answer to speech and play it back.

## Repository Structure

- `src/main.py`: CLI entry point.
- `src/app/orchestrator.py`: main runtime loop and pipeline orchestration.
- `src/audio/`: ASR, TTS, narration, and command audio logic.
- `src/camera/`: webcam streaming and frame selection.
- `src/perception/`: Grounding DINO, Depth Anything, MediaPipe, and mock backends.
- `src/fusion/`: multimodal state fusion utilities.
- `src/vlm/`: prompt building and multimodal API/model calls.
- `src/training_free_grpo/`: training-free GRPO dataset export, reward model, trainer, and pipeline.
- `configs/`: runtime, threshold, prompt, and secret configuration.
- `models/`: local model checkpoints and MediaPipe assets.
- `outputs/`: logged VLM inputs, overlays, and training-free GRPO outputs.
- `tests/`: unit tests for core modules.
- `report/`: project report material and paper references.

## Environment Requirements

- Python 3.9 or newer
- macOS is the easiest environment because the default TTS backend uses `say`
- a working webcam
- microphone permission granted to Terminal / Python
- enough disk space for local model weights under `models/`

If you run on Apple Silicon, CPU mode is the safest default. You can later switch to `auto` or `mps` in `configs/app.yaml` after validation.

## Installation

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. Prepare local model folders under `models/`.

Expected paths in the current config:

- `models/whisper-large-v3-turbo`
- `models/grounding-dino-tiny`
- `models/depth-anything-small-hf`
- `models/mediapipe/hand_landmarker.task`

Notes:

- Whisper, Grounding DINO, and Depth Anything are loaded from local model directories configured in `configs/app.yaml`.
- MediaPipe hand landmarker will auto-download its `.task` file if it is missing.

4. Create `configs/local.secrets.yaml` for API keys.

Example:

```yaml
vlm:
  api_key: sk-xxxxxxxxxxxxxxxx
```

## Main Configuration

The primary runtime configuration lives in `configs/app.yaml`.

Key sections:

- `app`: ASR, TTS, overlay, preview, and runtime behavior
- `vlm`: multimodal model backend, endpoint, model name, and request options
- `perception`: detector, depth, and hand backend selection
- `models`: local model folder paths
- `capture`: camera id, stream FPS, session steps, and mirror correction

The current default VLM backend is:

- backend: `responses_api`
- model: `gpt-5.4`
- endpoint: `https://code.rayinai.com/v1/responses`

If that endpoint is temporarily unavailable, you can switch back to the original GLM settings already left in comments inside `configs/app.yaml`.

## Running the Project

### Live camera mode

```bash
python3 -m src.main --camera-id 0
```

This is the main demo mode. It opens:

- a live camera preview window
- a multimodal dashboard window
- the full speech -> perception -> VLM -> speech loop

### Text-only query with a static image

```bash
python3 -m src.main --text-command "告诉我现在桌面上有什么" --image-path sample_test.jpg
```

### Audio file input with a static image

```bash
python3 -m src.main --audio-path sample_command.wav --image-path sample_test.jpg
```

## Overlay and Preview

The dashboard shows four panels:

- RGB camera plus runtime state
- Grounding DINO object detection
- Depth Anything depth map
- MediaPipe hand pose

Useful switches in `configs/app.yaml`:

- `app.enable_overlay`
- `app.enable_camera_preview`
- `app.camera_preview_title`
- `app.camera_preview_fps`
- `app.save_overlay`
- `capture.mirror_correction`

## Speech Behavior

The current ASR behavior is:

- record a full fixed 8 seconds per query
- do not stop early on short pauses
- trim leading and trailing silence only after recording

Important config keys:

- `app.asr_record_seconds`
- `app.asr_language`
- `app.asr_silence_threshold`
- `app.asr_min_confidence`

## Training-Free GRPO

This repository includes a training-free GRPO style prompt optimization pipeline. It does not finetune model weights. Instead, it improves the prompt profile injected before multimodal reasoning.

### Export a seed dataset from logged VLM inputs

```bash
python3 -m src.training_free_grpo.prepare_dataset \
  --log-dir outputs/vlm_inputs \
  --output-path outputs/training_free_grpo/train_dataset.jsonl
```

### Run training

```bash
python3 -m src.training_free_grpo.train \
  --dataset outputs/training_free_grpo/train_dataset.jsonl \
  --output-dir outputs/training_free_grpo
```

### Run export and training in one command

```bash
python3 -m src.training_free_grpo.pipeline \
  --log-dir outputs/vlm_inputs \
  --dataset-path outputs/training_free_grpo/train_dataset.jsonl \
  --output-dir outputs/training_free_grpo
```

Training outputs are written to:

- `outputs/training_free_grpo/latest_profile.json`

Runtime injection behavior is configured by:

- `configs/training_free_grpo.yaml`

## Tests and Basic Verification

Run unit tests:

```bash
python3 -m pytest tests
```

Run a quick syntax check:

```bash
python3 -m compileall src tests
```

## Common Troubleshooting

### 1. The system says it cannot answer the question

Possible causes:

- the multimodal API endpoint is unreachable
- the API key is missing or invalid
- the VLM request timed out

Check:

- `configs/local.secrets.yaml`
- `configs/app.yaml`
- terminal logs for `Responses API request failed` or `GLM request failed`

### 2. Left and right feel reversed

Check:

- `capture.mirror_correction` in `configs/app.yaml`

If your webcam feed is not mirrored, set it to `false`.

### 3. No camera image appears

Check:

- webcam permission
- correct `--camera-id`
- whether another app is already using the camera

### 4. The dashboard text is too small

This project now uses enlarged panel titles, subtitles, runtime labels, and detection labels in `src/ui/overlay.py`. If you still need a larger display for demo day, increase the panel width and height constants in that file.

### 5. Whisper or local perception models fail to load

Check:

- the corresponding folder exists under `models/`
- the checkpoint files are complete
- the model path in `configs/app.yaml` matches the actual folder

## Team Handoff Checklist

For a new teammate, the fastest setup path is:

1. Clone the repository.
2. Create a virtual environment and install `requirements.txt`.
3. Copy required model folders into `models/`.
4. Add the API key to `configs/local.secrets.yaml`.
5. Run `python3 -m compileall src tests`.
6. Run `python3 -m src.main --camera-id 0`.

## Current Status

The codebase already supports:

- real-time camera preview
- multimodal dashboard visualization
- same-source RGB, depth, and hand-pose reasoning inputs
- training-free prompt profile injection
- GPT-5.4 multimodal API integration

The project is best understood as a local assistive interaction prototype for tabletop object understanding and grasp guidance, not as a finished production system.
