"""
Evaluation runner.

Runs test cases against the AI Crop Doctor API and reports metrics.

Usage:
    python evaluation/evaluate.py [--api-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def load_test_cases(path: str = "evaluation/test_cases.json") -> list[dict]:
    """Load test cases from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_test_case(client: httpx.Client, base_url: str, test_case: dict) -> dict:
    """Run a single test case against the API."""
    tc_id = test_case["id"]
    print(f"\n{'-' * 40}")
    print(f"Running: {tc_id} - {test_case['description']}")

    start_time = time.time()

    try:
        # Send diagnosis request (text-only for automated testing)
        response = client.post(
            f"{base_url}/api/diagnosis",
            data={"text": test_case["farmer_text"]},
            timeout=60.0,
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}: {response.text[:200]}")
            return {
                "test_case_id": tc_id,
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "latency_ms": latency_ms,
            }

        result = response.json()

        # Evaluate results
        diagnosis = result.get("diagnosis")
        confidence = result.get("confidence", 0)
        needs_followup = result.get("needs_followup", False)
        response_text = result.get("response_text", "")

        predicted_disease = diagnosis.get("disease", "") if diagnosis else ""
        predicted_plant = diagnosis.get("plant_name", "") if diagnosis else ""

        expected_disease = test_case.get("expected_disease", "")
        expected_plant = test_case.get("expected_plant", "")
        expected_min_conf = test_case.get("expected_min_confidence", 0)
        expect_followup = test_case.get("expect_followup", False)

        # Check correctness
        disease_match = (
            expected_disease.lower() in predicted_disease.lower()
            if expected_disease and predicted_disease
            else False
        )
        confidence_met = confidence >= expected_min_conf
        followup_correct = needs_followup == expect_followup if expect_followup else True

        status = "pass" if (disease_match or expect_followup) and confidence_met else "fail"

        # Print result
        icon = "[PASS]" if status == "pass" else "[FAIL]"
        print(f"  {icon} Disease: '{predicted_disease}' (expected: '{expected_disease}')")
        print(f"     Confidence: {confidence:.2%} (min: {expected_min_conf:.0%})")
        print(f"     Follow-up: {needs_followup} (expected: {expect_followup})")
        print(f"     Latency: {latency_ms:.0f}ms")
        print(f"     Response: {response_text[:100]}...")

        return {
            "test_case_id": tc_id,
            "status": status,
            "predicted_disease": predicted_disease,
            "predicted_plant": predicted_plant,
            "expected_disease": expected_disease,
            "confidence": confidence,
            "disease_match": disease_match,
            "confidence_met": confidence_met,
            "followup_correct": followup_correct,
            "needs_followup": needs_followup,
            "latency_ms": latency_ms,
            "response_text": response_text[:500],
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        print(f"  [ERROR] Error: {e}")
        return {
            "test_case_id": tc_id,
            "status": "error",
            "error": str(e),
            "latency_ms": latency_ms,
        }


def main():
    parser = argparse.ArgumentParser(description="AI Crop Doctor Evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test-file", default="evaluation/test_cases.json", help="Test cases file")
    args = parser.parse_args()

    # Load test cases
    test_cases = load_test_cases(args.test_file)
    print(f"Loaded {len(test_cases)} test cases")

    # Check API health
    client = httpx.Client()
    try:
        health = client.get(f"{args.api_url}/api/health", timeout=10)
        print(f"API Health: {health.json()}")
    except Exception as e:
        print(f"[WARN] Cannot reach API at {args.api_url}: {e}")
        print("Make sure the server is running: python -m uvicorn app.main:app")
        sys.exit(1)

    # Run tests
    results = []
    for tc in test_cases:
        result = run_test_case(client, args.api_url, tc)
        results.append(result)
        time.sleep(2.5)  # Pace requests to avoid API rate limits

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] == "error")
    total = len(results)

    print(f"\nTotal:   {total}")
    print(f"Passed:  {passed} ({passed/total*100:.0f}%)")
    print(f"Failed:  {failed} ({failed/total*100:.0f}%)")
    print(f"Errors:  {errors} ({errors/total*100:.0f}%)")

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms", 0) > 0]
    if latencies:
        print(f"\nMean Latency: {sum(latencies)/len(latencies):.0f}ms")
        print(f"Max Latency:  {max(latencies):.0f}ms")

    # Save results
    output_path = "evaluation/results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    print("=" * 60)
    client.close()


if __name__ == "__main__":
    main()
