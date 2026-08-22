"""
Evaluation metrics computation.

Computes aggregate metrics from the event log for:
- Diagnosis correctness (vs ground truth)
- Retrieval relevance
- Confidence calibration
- Follow-up decision quality
- Latency breakdown
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_metrics(events: list[dict], test_cases: list[dict] | None = None) -> dict:
    """
    Compute aggregate metrics from logged events.

    Args:
        events: List of event dicts from DiagnosticEventLogger.
        test_cases: Optional ground truth test cases for correctness evaluation.

    Returns:
        Dict with computed metrics.
    """
    if not events:
        return {"total_events": 0}

    diagnosis_events = [e for e in events if e.get("event") == "diagnosis_complete"]
    error_events = [e for e in events if e.get("event") == "error"]
    followup_events = [e for e in events if e.get("event") == "followup_asked"]
    escalation_events = [e for e in events if e.get("event") == "escalated"]

    metrics: dict = {
        "total_events": len(events),
        "total_diagnoses": len(diagnosis_events),
        "total_errors": len(error_events),
        "total_followups": len(followup_events),
        "total_escalations": len(escalation_events),
    }

    # ── Confidence metrics ────────────────────────────────────────
    if diagnosis_events:
        confidences = [
            e.get("metrics", {}).get("confidence", 0) for e in diagnosis_events
        ]
        metrics["confidence"] = {
            "mean": round(sum(confidences) / len(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "high_confidence_pct": round(
                sum(1 for c in confidences if c >= 0.7) / len(confidences) * 100, 1
            ),
        }

    # ── Latency metrics ───────────────────────────────────────────
    all_latencies = [
        e.get("metrics", {}).get("latency_ms", 0)
        for e in events
        if e.get("metrics", {}).get("latency_ms", 0) > 0
    ]
    if all_latencies:
        metrics["latency_ms"] = {
            "mean": round(sum(all_latencies) / len(all_latencies), 1),
            "min": round(min(all_latencies), 1),
            "max": round(max(all_latencies), 1),
            "p50": round(sorted(all_latencies)[len(all_latencies) // 2], 1),
        }

    # ── Tool usage metrics ────────────────────────────────────────
    tool_counts: dict[str, int] = defaultdict(int)
    total_tool_calls = 0
    for e in events:
        for tool in e.get("metrics", {}).get("tools_called", []):
            tool_counts[tool] += 1
            total_tool_calls += 1

    metrics["tool_usage"] = {
        "total_calls": total_tool_calls,
        "by_tool": dict(tool_counts),
        "vision_usage_pct": round(
            sum(1 for e in events if e.get("metrics", {}).get("vision_used"))
            / max(len(events), 1) * 100,
            1,
        ),
        "voice_usage_pct": round(
            sum(1 for e in events if e.get("metrics", {}).get("voice_used"))
            / max(len(events), 1) * 100,
            1,
        ),
    }

    # ── Correctness (if test cases provided) ──────────────────────
    if test_cases and diagnosis_events:
        correct = 0
        evaluated = 0
        for tc in test_cases:
            tc_disease = tc.get("expected_disease", "").lower()
            matching_events = [
                e
                for e in diagnosis_events
                if e.get("extra", {}).get("test_case_id") == tc.get("id")
            ]
            for event in matching_events:
                predicted = event.get("metrics", {}).get("predicted_disease", "").lower()
                if tc_disease and predicted and tc_disease in predicted:
                    correct += 1
                evaluated += 1

        if evaluated > 0:
            metrics["correctness"] = {
                "evaluated": evaluated,
                "correct": correct,
                "accuracy_pct": round(correct / evaluated * 100, 1),
            }

    # ── Escalation rate ───────────────────────────────────────────
    total_outcomes = len(diagnosis_events) + len(escalation_events)
    if total_outcomes > 0:
        metrics["escalation_rate_pct"] = round(
            len(escalation_events) / total_outcomes * 100, 1
        )

    return metrics


def print_metrics_report(metrics: dict) -> None:
    """Print a formatted metrics report to stdout."""
    print("\n" + "=" * 60)
    print("AI CROP DOCTOR — EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal Events:      {metrics.get('total_events', 0)}")
    print(f"Diagnoses:         {metrics.get('total_diagnoses', 0)}")
    print(f"Follow-ups Asked:  {metrics.get('total_followups', 0)}")
    print(f"Escalations:       {metrics.get('total_escalations', 0)}")
    print(f"Errors:            {metrics.get('total_errors', 0)}")

    if "confidence" in metrics:
        c = metrics["confidence"]
        print(f"\nConfidence:")
        print(f"  Mean:            {c['mean']:.2%}")
        print(f"  Range:           {c['min']:.2%} — {c['max']:.2%}")
        print(f"  High (≥70%):     {c['high_confidence_pct']:.1f}%")

    if "latency_ms" in metrics:
        l = metrics["latency_ms"]
        print(f"\nLatency:")
        print(f"  Mean:            {l['mean']:.0f}ms")
        print(f"  Median (p50):    {l['p50']:.0f}ms")
        print(f"  Range:           {l['min']:.0f}ms — {l['max']:.0f}ms")

    if "tool_usage" in metrics:
        t = metrics["tool_usage"]
        print(f"\nTool Usage:")
        print(f"  Total Calls:     {t['total_calls']}")
        for tool, count in t.get("by_tool", {}).items():
            print(f"  {tool}: {count}")
        print(f"  Vision Usage:    {t['vision_usage_pct']:.1f}%")
        print(f"  Voice Usage:     {t['voice_usage_pct']:.1f}%")

    if "correctness" in metrics:
        c = metrics["correctness"]
        print(f"\nCorrectness:")
        print(f"  Evaluated:       {c['evaluated']}")
        print(f"  Correct:         {c['correct']}")
        print(f"  Accuracy:        {c['accuracy_pct']:.1f}%")

    if "escalation_rate_pct" in metrics:
        print(f"\nEscalation Rate:   {metrics['escalation_rate_pct']:.1f}%")

    print("\n" + "=" * 60)
