# Project Jerico 2026 Frontier Migration Roadmap

This roadmap converts Jerico from a detection-first stack into a reasoning-first, spatially aware 2026 system. It is aligned to the five Frontier Validation Suite cases and includes measurable exit criteria for each phase.

## Phase 1 - Edge and Detection Optimization

Goal: Remove density-sensitive bottlenecks and stabilize edge latency.

Actions:
- Replace YOLOv8 inference path with YOLO26 deployment artifacts.
- Ensure detector pipeline is end-to-end NMS-free in production runtime.
- Keep existing API contracts stable while swapping internals in the detection module.

Code touchpoints:
- src/detect.py
- frontend/api.py
- src/worker_pool.py

Exit criteria:
- Frontier Case T3 passes.
- Mean latency meets target and variance ratio is under threshold for high-density scenes.
- CI publishes latency report artifacts.

## Phase 2 - Open Vocabulary Semantic Layer

Goal: Remove rigid class dependencies and support text-defined concept detection.

Actions:
- Add SAM 3 semantic segmentation interface behind a concept-prompt API.
- Replace per-class model fanout logic with concept-driven segmentation and tracking outputs.
- Add SAM 3.1 object multiplex mode for shared-memory temporal tracking.

Code touchpoints:
- src/detect.py (or new src/semantic_detect.py)
- frontend/api.py
- src/tracker.py (or new multiplex tracking wrapper)

Exit criteria:
- Frontier Case T2 passes with target recall.
- Frontier Case T5 tracking continuity improves and no identity breaks in occlusion tests.
- Harness report includes concept-level confusion and recall metrics.

## Phase 3 - Cognitive Reasoning and Causal Analysis

Goal: Replace keyword scoring with explicit intent reasoning.

Actions:
- Introduce Qwen3-VL-Thinking reasoning adapter for scene-level causal interpretation.
- Add structured reasoning output schema:
  - context_cues
  - intent_hypothesis
  - severity
  - rationale
- Gate heavy reasoning by policy and cache per-track context windows.

Code touchpoints:
- src/scene_understanding.py
- src/hybrid_stack.py
- src/threat_logic.py

Exit criteria:
- Frontier Case T1 passes.
- Long-tail reliability gain is measured against baseline in harness output.
- Severity assignments are traceable through structured rationale.

## Phase 4 - Spatial Awareness and Depth Gating

Goal: Eliminate 2D-overlap false positives with depth-aware proximity logic.

Actions:
- Integrate Depth Anything V2 pre-processor for relative depth map generation.
- Update armed-person logic to require 2D proximity plus depth-plane consistency.
- Add depth-threshold calibration profile per camera type.

Code touchpoints:
- src/threat_logic.py
- frontend/api.py
- optional new src/depth_gate.py

Exit criteria:
- Frontier Case T4 passes.
- False-positive rate for overlap-only cases drops to target.
- Depth gating metrics are exported in report JSON.

## Immediate Execution Step - Frontier Benchmark Harness

A baseline harness is included at:
- benchmarks/frontier_validation_harness.py
- benchmarks/frontier_cases.json

Run baseline suite:

```bash
python benchmarks/frontier_validation_harness.py
```

Write a custom result path:

```bash
python benchmarks/frontier_validation_harness.py --out benchmarks/results/frontier_results_baseline.json
```

Current adapter modes:
- baseline: mirrors current Jerico architecture assumptions.
- placeholder2026: contract stub for future integrated stack.

Recommended CI gate policy:
- Fail build if baseline regression occurs.
- Require monotonic improvement of pass_count for upgraded adapter.
- Persist JSON reports for trend tracking.
