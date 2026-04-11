#!/usr/bin/env python3
"""
Frontier Validation Harness for Project Jerico.

Purpose:
- Establish a reproducible baseline for the 5-case 2026 Frontier Validation Suite.
- Compare current Jerico behavior against future upgraded adapters.
- Produce machine-readable JSON summaries for CI/CD quality gates.

This harness is intentionally lightweight and model-agnostic:
- Baseline adapter reflects current rule/keyword-centric architecture.
- Future adapters can be plugged in without changing the suite contract.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


@dataclass
class FrontierCase:
    case_id: str
    title: str
    category: str
    payload: Dict[str, Any]
    expected: Dict[str, Any]


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    score: float
    observed: Dict[str, Any]
    expected: Dict[str, Any]
    notes: str


@dataclass
class AdapterResult:
    adapter: str
    pass_count: int
    total_cases: int
    pass_rate: float
    case_results: List[CaseResult]
    aggregate: Dict[str, Any]


class BaseAdapter:
    name = "base"

    def run_case(self, case: FrontierCase) -> CaseResult:
        raise NotImplementedError


class JericoBaselineAdapter(BaseAdapter):
    """
    Mirrors current baseline behavior at a benchmark-contract level.
    It does not run heavyweight model inference; it evaluates the same logic shape:
    - keyword-centric scene interpretation
    - fixed class schema assumptions
    - 2D proximity heuristics
    - centroid-style temporal continuity limits
    """

    name = "jerico-baseline-2024-style"

    def run_case(self, case: FrontierCase) -> CaseResult:
        if case.case_id == "T1_INTENT_IDENTITY":
            return self._case_1(case)
        if case.case_id == "T2_ZERO_SHOT_OPEN_VOCAB":
            return self._case_2(case)
        if case.case_id == "T3_NMS_FREE_LATENCY":
            return self._case_3(case)
        if case.case_id == "T4_3D_PROXIMITY":
            return self._case_4(case)
        if case.case_id == "T5_TEMPORAL_IDENTITY":
            return self._case_5(case)

        return CaseResult(
            case_id=case.case_id,
            passed=False,
            score=0.0,
            observed={"error": "unknown_case"},
            expected=case.expected,
            notes="Case not implemented in baseline adapter.",
        )

    def _case_1(self, case: FrontierCase) -> CaseResult:
        samples = case.payload.get("samples", [])
        out = []
        keyword_set = {
            "weapon",
            "metallic",
            "crowbar",
            "restricted",
            "intruder",
            "machete",
            "mask",
            "balaclava",
            "hoodie",
        }

        for item in samples:
            scene = item.get("scene_text", "").lower()
            hits = [k for k in keyword_set if k in scene]
            severity = "CRITICAL" if len(hits) >= 2 else "LOW"
            out.append({"sample": item.get("name"), "severity": severity, "hits": hits})

        expected_map = case.expected.get("per_sample_severity", {})
        correct = 0
        for row in out:
            if expected_map.get(row["sample"]) == row["severity"]:
                correct += 1
        score = correct / max(1, len(out))
        passed = score >= case.expected.get("min_accuracy", 1.0)

        return CaseResult(
            case_id=case.case_id,
            passed=passed,
            score=score,
            observed={"predictions": out, "accuracy": score},
            expected=case.expected,
            notes="Baseline keyword scoring tends to over-flag authorized scenarios.",
        )

    def _case_2(self, case: FrontierCase) -> CaseResult:
        concepts = case.payload.get("concept_prompts", [])
        known_schema = {"weapon", "person", "fire", "road anomaly", "vehicle"}
        detected = []
        for concept in concepts:
            c = concept.lower()
            matched = any(k in c for k in known_schema)
            detected.append({"concept": concept, "detected": matched})

        recall = sum(1 for d in detected if d["detected"]) / max(1, len(detected))
        passed = recall >= case.expected.get("min_recall", 0.75)

        return CaseResult(
            case_id=case.case_id,
            passed=passed,
            score=recall,
            observed={"detections": detected, "recall": recall},
            expected=case.expected,
            notes="Baseline fixed-class design misses novel text-defined concepts.",
        )

    def _case_3(self, case: FrontierCase) -> CaseResult:
        random.seed(26)
        counts = case.payload.get("object_counts", [10, 20, 40, 60])
        runs_per_count = int(case.payload.get("runs_per_count", 30))

        all_runs_ms: List[float] = []
        per_count_mean = {}

        for n in counts:
            bucket = []
            for _ in range(runs_per_count):
                # Simulated baseline latency model:
                # constant detector + variable post-processing growth
                nms_cost = 0.03 * math.log(max(2, n), 2) + (n * 0.002)
                jitter = random.uniform(-0.08, 0.08)
                latency_ms = 2.1 + nms_cost + jitter
                bucket.append(round(latency_ms, 4))
                all_runs_ms.append(latency_ms)
            per_count_mean[str(n)] = round(mean(bucket), 4)

        mean_ms = mean(all_runs_ms) if all_runs_ms else 0.0
        std_ms = pstdev(all_runs_ms) if len(all_runs_ms) > 1 else 0.0
        variance_ratio = (std_ms / mean_ms) if mean_ms else 0.0

        target_latency_ms = float(case.expected.get("target_mean_latency_ms", 1.5))
        target_var_ratio = float(case.expected.get("max_variance_ratio", 0.03))
        passed = mean_ms <= target_latency_ms and variance_ratio <= target_var_ratio

        return CaseResult(
            case_id=case.case_id,
            passed=passed,
            score=max(0.0, 1.0 - min(1.0, variance_ratio / max(1e-9, target_var_ratio))),
            observed={
                "mean_latency_ms": round(mean_ms, 4),
                "std_latency_ms": round(std_ms, 4),
                "variance_ratio": round(variance_ratio, 4),
                "per_count_mean_ms": per_count_mean,
            },
            expected=case.expected,
            notes="Baseline simulation includes density-sensitive post-processing cost.",
        )

    def _case_4(self, case: FrontierCase) -> CaseResult:
        p = case.payload
        person_box = p.get("person_box", [100, 100, 220, 380])
        weapon_box = p.get("weapon_box", [210, 180, 270, 300])
        person_depth = float(p.get("person_depth", 8.0))
        weapon_depth = float(p.get("weapon_depth", 2.5))
        threshold_px = float(p.get("pixel_threshold", 120.0))

        pcx = (person_box[0] + person_box[2]) / 2.0
        pcy = (person_box[1] + person_box[3]) / 2.0
        wcx = (weapon_box[0] + weapon_box[2]) / 2.0
        wcy = (weapon_box[1] + weapon_box[3]) / 2.0
        dist_2d = math.hypot(pcx - wcx, pcy - wcy)

        armed_flag_2d = dist_2d < threshold_px
        # Baseline does not use depth gating.
        predicted = "ARMED_PERSON" if armed_flag_2d else "NO_ARMED_LINK"
        expected_label = case.expected.get("label", "NO_ARMED_LINK")
        passed = predicted == expected_label

        return CaseResult(
            case_id=case.case_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            observed={
                "predicted_label": predicted,
                "distance_2d_px": round(dist_2d, 2),
                "person_depth": person_depth,
                "weapon_depth": weapon_depth,
                "depth_delta": round(abs(person_depth - weapon_depth), 3),
            },
            expected=case.expected,
            notes="2D proximity logic can fire despite large Z-axis separation.",
        )

    def _case_5(self, case: FrontierCase) -> CaseResult:
        occlusion_s = float(case.payload.get("occlusion_seconds", 3.0))
        max_missed_frames = int(case.payload.get("max_missed_frames", 12))
        fps = float(case.payload.get("fps", 30.0))

        tolerated_s = max_missed_frames / max(1.0, fps)
        identity_preserved = occlusion_s <= tolerated_s
        passed = identity_preserved and bool(case.expected.get("require_identity_lock", True))

        return CaseResult(
            case_id=case.case_id,
            passed=passed,
            score=1.0 if identity_preserved else 0.0,
            observed={
                "occlusion_seconds": occlusion_s,
                "tolerated_seconds": round(tolerated_s, 3),
                "identity_preserved": identity_preserved,
            },
            expected=case.expected,
            notes="Centroid-style tracking likely drops identity for long full occlusions.",
        )


class Placeholder2026Adapter(BaseAdapter):
    """
    Contract placeholder for future implementations.
    Replace this class with real calls into:
    - YOLO26 runtime
    - SAM 3 / SAM 3.1 semantic + multiplex tracking
    - Qwen3-VL-Thinking reasoning layer
    - Depth Anything V2 proximity gating
    """

    name = "frontier-2026-placeholder"

    def run_case(self, case: FrontierCase) -> CaseResult:
        return CaseResult(
            case_id=case.case_id,
            passed=False,
            score=0.0,
            observed={"status": "not_implemented"},
            expected=case.expected,
            notes="Implement this adapter to benchmark upgraded modules.",
        )


def _load_cases(path: Path) -> List[FrontierCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[FrontierCase] = []
    for item in data.get("cases", []):
        out.append(
            FrontierCase(
                case_id=item["case_id"],
                title=item["title"],
                category=item["category"],
                payload=item.get("payload", {}),
                expected=item.get("expected", {}),
            )
        )
    return out


def _run_adapter(adapter: BaseAdapter, cases: List[FrontierCase]) -> AdapterResult:
    case_results = [adapter.run_case(c) for c in cases]
    pass_count = sum(1 for r in case_results if r.passed)
    total = len(case_results)
    pass_rate = (pass_count / total) if total else 0.0

    by_category: Dict[str, List[float]] = {}
    for c, r in zip(cases, case_results):
        by_category.setdefault(c.category, []).append(r.score)
    category_scores = {k: round(mean(v), 4) for k, v in by_category.items()}

    aggregate = {
        "category_scores": category_scores,
    }

    return AdapterResult(
        adapter=adapter.name,
        pass_count=pass_count,
        total_cases=total,
        pass_rate=pass_rate,
        case_results=case_results,
        aggregate=aggregate,
    )


def _to_jsonable(results: List[AdapterResult], suite_name: str) -> Dict[str, Any]:
    return {
        "suite": suite_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "adapter": r.adapter,
                "pass_count": r.pass_count,
                "total_cases": r.total_cases,
                "pass_rate": round(r.pass_rate, 4),
                "aggregate": r.aggregate,
                "cases": [asdict(c) for c in r.case_results],
            }
            for r in results
        ],
    }


def _print_table(results: List[AdapterResult]) -> None:
    print("\nFrontier Validation Summary")
    print("=" * 80)
    for r in results:
        print(
            f"- {r.adapter:30s} | pass {r.pass_count:>2d}/{r.total_cases:<2d} "
            f"| pass_rate={r.pass_rate:.2%}"
        )
    print("=" * 80)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_cases = repo_root / "benchmarks" / "frontier_cases.json"
    default_out = repo_root / "benchmarks" / "results" / "frontier_results.json"

    parser = argparse.ArgumentParser(description="Run Jerico Frontier Validation Suite")
    parser.add_argument("--cases", type=Path, default=default_cases, help="Path to frontier cases JSON")
    parser.add_argument("--out", type=Path, default=default_out, help="Path to output JSON report")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["baseline", "placeholder2026"],
        choices=["baseline", "placeholder2026"],
        help="Adapters to execute",
    )
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    adapters: List[BaseAdapter] = []

    for key in args.adapters:
        if key == "baseline":
            adapters.append(JericoBaselineAdapter())
        elif key == "placeholder2026":
            adapters.append(Placeholder2026Adapter())

    results = [_run_adapter(adp, cases) for adp in adapters]
    _print_table(results)

    payload = _to_jsonable(results, suite_name="Jerico Frontier Validation Suite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved JSON report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
