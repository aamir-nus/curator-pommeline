"""
Guardrail classification routes for content filtering.
"""

import time
from typing import List
from fastapi import APIRouter, HTTPException

from ..models import GuardrailRequest, GuardrailResponse, ErrorResponse
from src.guardrails.classifier import get_guardrail_classifier, GuardrailLabel
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger("guardrail_routes")
router = APIRouter(prefix="/guardrail", tags=["guardrails"])
classifier = get_guardrail_classifier()


@router.post("/classify", response_model=GuardrailResponse)
async def classify_text(request: GuardrailRequest):
    """
    Classify text using guardrail classifier.

    Args:
        request: GuardrailRequest with text to classify

    Returns:
        GuardrailResponse with classification result
    """
    start_time = time.time()

    try:
        logger.info(f"Classifying text: '{request.text[:50]}...'")

        # Perform classification
        result = classifier.classify(request.text)

        # Determine if content is appropriate
        is_appropriate = result.label == GuardrailLabel.APPROPRIATE

        response = GuardrailResponse(
            label=result.label,
            confidence=result.confidence,
            reasoning=result.reasoning,
            latency_ms=result.latency_ms,
            is_appropriate=is_appropriate
        )

        # Log metrics
        total_latency = (time.time() - start_time) * 1000
        logger.log_response("/guardrail/classify", 200, total_latency)
        metrics.add_metric("guardrail_classification", 1)

        return response

    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error(f"Error during guardrail classification: {e}")
        logger.log_response("/guardrail/classify", 500, total_latency)

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ClassificationError",
                message="Failed to classify text",
                details={
                    "text_preview": request.text[:100] if request.text else "",
                    "error": str(e)
                },
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.post("/batch-classify")
async def batch_classify_texts(texts: List[str]):
    """
    Classify multiple texts in batch.

    Args:
        texts: List of texts to classify

    Returns:
        List of classification results
    """
    start_time = time.time()

    try:
        if not texts:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="EmptyInput",
                    message="No texts provided for classification",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        logger.info(f"Batch classifying {len(texts)} texts")

        # Perform batch classification
        results = classifier.batch_classify(texts)

        # Convert to response format
        response_results = []
        for i, result in enumerate(results):
            is_appropriate = result.label == GuardrailLabel.APPROPRIATE
            response_results.append({
                "text_index": i,
                "text_preview": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "label": result.label,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "is_appropriate": is_appropriate,
                "latency_ms": result.latency_ms
            })

        total_latency = (time.time() - start_time) * 1000

        response = {
            "results": response_results,
            "total_texts": len(texts),
            "total_latency_ms": total_latency,
            "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0.0
        }

        logger.info(f"Batch classification completed in {total_latency:.2f}ms")
        metrics.add_metric("guardrail_batch_classification", len(texts))

        return response

    except HTTPException:
        raise
    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error(f"Error during batch guardrail classification: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="BatchClassificationError",
                message="Failed to classify texts in batch",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/labels")
async def get_available_labels():
    """Get available guardrail classification labels."""
    try:
        labels = [
            {
                "label": "appropriate",
                "description": "Content that is safe and within scope",
                "action": "allow"
            },
            {
                "label": "out_of_scope",
                "description": "Content that is outside the system's intended scope",
                "action": "redirect_or_decline"
            },
            {
                "label": "inappropriate",
                "description": "Content that is harmful, offensive, or inappropriate",
                "action": "block"
            },
            {
                "label": "prompt_injection",
                "description": "Attempts to manipulate or override system instructions",
                "action": "block"
            }
        ]

        return {
            "labels": labels,
            "total_labels": len(labels)
        }

    except Exception as e:
        logger.error(f"Error getting guardrail labels: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="LabelsError",
                message="Failed to retrieve guardrail labels",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/stats")
async def get_guardrail_stats():
    """Get guardrail classifier statistics and configuration."""
    try:
        stats = classifier.get_classification_stats()

        if stats["type"] == "ml_based":
            return {
                "classifier_type": stats["type"],
                "configuration": {
                    "model_type": stats["model_type"],
                    "vectorizer_type": stats["vectorizer_type"],
                    "feature_count": stats["feature_count"],
                    "support_vector_count": stats["support_vector_count"]
                },
                "performance": {
                    "classification_speed_ms": "Fast - typically 1-5ms per classification",
                    "memory_usage": "Medium - loaded models in memory",
                    "scalability": "High - optimized for batch processing"
                }
            }
        else:
            # Rule-based classifier stats
            return {
                "classifier_type": stats["type"],
                "configuration": {
                    "injection_patterns": stats["injection_patterns"],
                    "inappropriate_patterns": stats["inappropriate_patterns"],
                    "out_of_scope_patterns": stats["out_of_scope_patterns"]
                },
                "rule_categories": {
                    "shopping_keywords_count": stats["shopping_keywords"],
                    "policy_keywords_count": stats["policy_keywords"],
                    "product_categories_count": stats["product_categories"]
                },
                "performance": {
                    "classification_speed_ms": "Variable - depends on text length and patterns",
                    "memory_usage": "Low - rule-based approach",
                    "scalability": "High - simple pattern matching"
                }
            }

    except Exception as e:
        logger.error(f"Error getting guardrail stats: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="StatsError",
                message="Failed to retrieve guardrail statistics",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.post("/test")
async def test_guardrail_rules(test_cases: List[dict]):
    """
    Test guardrail rules against custom test cases.

    Args:
        test_cases: List of test cases with expected results

    Returns:
        Test results with pass/fail status
    """
    try:
        if not test_cases:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="EmptyTestCases",
                    message="No test cases provided",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        logger.info(f"Testing guardrail rules with {len(test_cases)} test cases")

        results = []
        passed = 0

        for i, test_case in enumerate(test_cases):
            if "text" not in test_case:
                continue

            text = test_case["text"]
            expected_label = test_case.get("expected_label")
            tolerance = test_case.get("tolerance", 0.1)  # Confidence tolerance

            # Classify the text
            result = classifier.classify(text)

            # Check if result matches expectations
            is_correct = True
            if expected_label:
                is_correct = (
                    result.label == expected_label and
                    (tolerance >= 1.0 or abs(result.confidence - test_case.get("expected_confidence", 1.0)) <= tolerance)
                )

            if is_correct:
                passed += 1

            results.append({
                "test_case_index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "expected": {
                    "label": expected_label,
                    "confidence": test_case.get("expected_confidence")
                },
                "actual": {
                    "label": result.label.value,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning
                },
                "passed": is_correct
            })

        test_summary = {
            "total_tests": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results) if results else 0.0,
            "results": results
        }

        logger.info(f"Guardrail testing completed: {passed}/{len(results)} tests passed")

        return test_summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during guardrail testing: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="TestError",
                message="Failed to test guardrail rules",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )