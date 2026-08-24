# VLM Integration Plan

## Goal
- Add an event-driven local VLM narration path to SmartReach-HSM.
- Preserve the existing rule-based control loop as the only authority for movement commands.

## Scope
- Build a dashboard-to-VLM packaging path.
- Build a prompt contract using dashboard image plus structured state.
- Build a local VLM inference service.
- Trigger VLM narration only on selected events.
- Add narration priority rules so command speech remains dominant.

## Trigger Events
- Session start after target acquisition
- Target lost
- Hand lost
- Explicit user question about the scene
- Optional low-frequency periodic summary with cooldown

## Output Contract
- VLM returns JSON only.
- Fields: scene_status, scene_description, uncertainty, speak.
- VLM must not emit motion commands.

## Safety Rules
- Rule-based guidance remains authoritative.
- VLM narration is skipped if a fresh directional command is pending.
- VLM narration is rate-limited with cooldown.

## Implementation Sections
- `src/vlm/prompt_builder.py`
- `src/vlm/vlm_service.py`
- `src/audio/narration_player.py`
- orchestrator integration with event triggers and cooldown
- config entries for model path, enable flag, cooldown, and event toggles
- tests for prompt packaging, event gating, and narration priority

## Validation
- static checks pass
- unit tests pass
- VLM service loads locally
- event-triggered narration runs on a sample dashboard image
