"""
Simple rule-based classifier for guardrails (PoC implementation).
"""

import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..utils.prompts import get_json_prompt

logger = get_logger("guardrail_classifier")


class GuardrailLabel(Enum):
    """Guardrail classification labels."""
    APPROPRIATE = "appropriate"
    OUT_OF_SCOPE = "out_of_scope"
    INAPPROPRIATE = "inappropriate"
    PROMPT_INJECTION = "prompt_injection"


@dataclass
class ClassificationResult:
    """Result of guardrail classification."""
    label: GuardrailLabel
    confidence: float
    reasoning: str
    latency_ms: float

class RuleBasedClassifier:
    """Simple rule-based classifier for PoC guardrails."""

    def __init__(self):
        self.rules = self._initialize_rules()
        self.injection_patterns = self._initialize_injection_patterns()
        self.inappropriate_patterns = self._initialize_inappropriate_patterns()
        self.out_of_scope_patterns = self._initialize_out_of_scope_patterns()

    def _initialize_rules(self) -> Dict[str, List[str]]:
        """Initialize classification rules."""
        return {
            "shopping_keywords": [
                "buy", "purchase", "price", "cost", "cheap", "expensive",
                "product", "item", "deal", "discount", "sale", "offer",
                "store", "shop", "order", "shipping", "delivery", "available",
                "stock", "inventory", "brand", "model", "specifications"
            ],
            "policy_keywords": [
                "return", "refund", "warranty", "guarantee", "policy",
                "terms", "conditions", "rules", "guidelines", "support",
                "help", "customer service", "contact", "complaint", "feedback"
            ],
            "product_categories": [
                "phone", "laptop", "computer", "tablet", "headphone", "earphone",
                "camera", "tv", "monitor", "speaker", "watch", "gaming",
                "apple", "samsung", "dell", "sony", "microsoft", "google"
            ],
            "inappropriate_content": [
                "hack", "illegal", "steal", "theft", "weapon", "violence",
                "hate", "discrimination", "harassment", "abuse", "harmful"
            ],
            "injection_attempts": [
                "ignore all previous instructions",
                "disregard the above",
                "system prompt", "developer mode", "jailbreak",
                "act as", "pretend to be", "roleplay as", "simulate",
                "forget everything", "new instructions", "override"
            ]
        }

    def _initialize_injection_patterns(self) -> List[re.Pattern]:
        """Initialize prompt injection detection patterns."""
        patterns = [
            re.compile(r"ignore\s+all\s+previous\s+instructions", re.IGNORECASE),
            re.compile(r"disregard\s+(the\s+)?above", re.IGNORECASE),
            re.compile(r"system\s+(prompt|message|instructions)", re.IGNORECASE),
            re.compile(r"(jailbreak|developer\s+mode|admin\s+mode)", re.IGNORECASE),
            re.compile(r"(act\s+as|pretend\s+to\s+be|simulate)", re.IGNORECASE),
            re.compile(r"forget\s+everything\s+(above|previous)", re.IGNORECASE),
            re.compile(r"new\s+instructions", re.IGNORECASE),
            re.compile(r"override\s+(your\s+)?programming", re.IGNORECASE),
            re.compile(r"roleplay\s+as", re.IGNORECASE),
        ]
        return patterns

    def _initialize_inappropriate_patterns(self) -> List[re.Pattern]:
        """Initialize inappropriate content detection patterns."""
        patterns = [
            re.compile(r"\b(hack|crack|steal|theft|rob)\b", re.IGNORECASE),
            re.compile(r"\b(weapon|gun|knife|bomb|explosive)\b", re.IGNORECASE),
            re.compile(r"\b(violence|kill|harm|hurt|abuse)\b", re.IGNORECASE),
            re.compile(r"\b(hate|discriminat|harass|bully)\b", re.IGNORECASE),
            re.compile(r"\b(illegal|criminal|fraud|scam)\b", re.IGNORECASE),
        ]
        return patterns

    def _initialize_out_of_scope_patterns(self) -> List[re.Pattern]:
        """Initialize out-of-scope detection patterns."""
        patterns = [
            re.compile(r"\b(weather|temperature|forecast|rain|sun|snow)\b", re.IGNORECASE),
            re.compile(r"\b(stock\s+market|finance|investment|crypto|bitcoin)\b", re.IGNORECASE),
            re.compile(r"\b(sports|game|match|score|team|player)\b", re.IGNORECASE),
            re.compile(r"\b(politics|election|government|president)\b", re.IGNORECASE),
            re.compile(r"\b(celebrity|gossip|entertainment|movie)\b", re.IGNORECASE),
            re.compile(r"\b(recipe|cook|food|meal|restaurant)\b", re.IGNORECASE),
            re.compile(r"\b(travel|vacation|hotel|flight|trip)\b", re.IGNORECASE),
        ]
        return patterns

    @track_latency("guardrail_classify")
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text using rule-based approach.

        Args:
            text: Input text to classify

        Returns:
            ClassificationResult with label, confidence, and reasoning
        """
        import time
        start_time = time.time()

        text_lower = text.lower().strip()

        # Check for prompt injection first (highest priority)
        injection_result = self._check_prompt_injection(text)
        if injection_result:
            latency_ms = (time.time() - start_time) * 1000
            return ClassificationResult(
                label=GuardrailLabel.PROMPT_INJECTION,
                confidence=0.9,
                reasoning=injection_result,
                latency_ms=latency_ms
            )

        # Check for inappropriate content
        inappropriate_result = self._check_inappropriate_content(text)
        if inappropriate_result:
            latency_ms = (time.time() - start_time) * 1000
            return ClassificationResult(
                label=GuardrailLabel.INAPPROPRIATE,
                confidence=0.8,
                reasoning=inappropriate_result,
                latency_ms=latency_ms
            )

        # Check if out of scope
        out_of_scope_result = self._check_out_of_scope(text)
        if out_of_scope_result:
            latency_ms = (time.time() - start_time) * 1000
            return ClassificationResult(
                label=GuardrailLabel.OUT_OF_SCOPE,
                confidence=0.7,
                reasoning=out_of_scope_result,
                latency_ms=latency_ms
            )

        # Check if appropriate (shopping/policy related)
        appropriate_result = self._check_appropriate_content(text)
        if appropriate_result:
            latency_ms = (time.time() - start_time) * 1000
            return ClassificationResult(
                label=GuardrailLabel.APPROPRIATE,
                confidence=0.8,
                reasoning=appropriate_result,
                latency_ms=latency_ms
            )

        # Default to appropriate if no specific patterns match
        latency_ms = (time.time() - start_time) * 1000
        return ClassificationResult(
            label=GuardrailLabel.APPROPRIATE,
            confidence=0.5,
            reasoning="No specific patterns detected, defaulting to appropriate",
            latency_ms=latency_ms
        )

    def _check_prompt_injection(self, text: str) -> Optional[str]:
        """Check for prompt injection attempts."""
        for pattern in self.injection_patterns:
            if pattern.search(text):
                return f"Prompt injection pattern detected: {pattern.pattern}"

        # Check for role-playing attempts
        if re.search(r"act as|pretend to be|roleplay as", text, re.IGNORECASE):
            return "Role-playing attempt detected"

        return None

    def _check_inappropriate_content(self, text: str) -> Optional[str]:
        """Check for inappropriate content."""
        for pattern in self.inappropriate_patterns:
            if pattern.search(text):
                return f"Inappropriate content pattern detected: {pattern.pattern}"

        return None

    def _check_out_of_scope(self, text: str) -> Optional[str]:
        """Check if content is out of scope."""
        text_lower = text.lower()

        # Count out-of-scope keywords
        out_of_scope_count = 0
        matched_patterns = []

        for pattern in self.out_of_scope_patterns:
            if pattern.search(text):
                out_of_scope_count += 1
                matched_patterns.append(pattern.pattern)

        # Check if mostly out-of-scope content
        if out_of_scope_count >= 2:
            return f"Multiple out-of-scope topics detected: {', '.join(matched_patterns)}"

        # Check if text is mostly about out-of-scope topics
        if out_of_scope_count == 1 and len(text.split()) > 10:
            return f"Out-of-scope topic detected: {matched_patterns[0]}"

        return None

    def _check_appropriate_content(self, text: str) -> Optional[str]:
        """Check if content is appropriate (shopping/policy related)."""
        text_lower = text.lower()

        # Count appropriate keywords
        shopping_count = sum(1 for keyword in self.rules["shopping_keywords"]
                           if keyword in text_lower)
        policy_count = sum(1 for keyword in self.rules["policy_keywords"]
                         if keyword in text_lower)
        product_count = sum(1 for keyword in self.rules["product_categories"]
                          if keyword in text_lower)

        total_appropriate = shopping_count + policy_count + product_count

        if total_appropriate >= 2:
            categories = []
            if shopping_count > 0:
                categories.append("shopping")
            if policy_count > 0:
                categories.append("policy")
            if product_count > 0:
                categories.append("products")

            return f"Appropriate content detected: {', '.join(categories)} related"

        elif total_appropriate == 1 and len(text.split()) <= 10:
            return "Appropriate short query detected"

        return None

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        results = []
        for text in texts:
            result = self.classify(text)
            results.append(result)

        metrics.add_metric("guardrail_batch_size", len(texts))
        return results

    def get_classification_stats(self) -> Dict[str, any]:
        """Get classifier statistics."""
        return {
            "type": "rule_based",
            "injection_patterns": len(self.injection_patterns),
            "inappropriate_patterns": len(self.inappropriate_patterns),
            "out_of_scope_patterns": len(self.out_of_scope_patterns),
            "shopping_keywords": len(self.rules["shopping_keywords"]),
            "policy_keywords": len(self.rules["policy_keywords"]),
            "product_categories": len(self.rules["product_categories"]),
        }
    
class MLClassifier:
    """
    Simple ML-based classifier for guardrails.
    """

    def __init__(self):
        import joblib
        self.classifier = joblib.load("./data/models/guardrail_svc_classifier.joblib")
        self.vectorizer = joblib.load("./data/models/guardrail_tfidf_vectorizer.joblib")

    def _get_guardrail_label(self, prediction: str) -> GuardrailLabel:

        """Map prediction string to GuardrailLabel."""

        #possible predictions = ['shopping_keywords', 'policy_keywords','product_categories','inappropriate_content','injection_attempts','out_of_scope']

        if prediction == "inappropriate_content":
            return GuardrailLabel.INAPPROPRIATE
        elif prediction == "out_of_scope":
            return GuardrailLabel.OUT_OF_SCOPE
        elif prediction == "injection_attempts":
            return GuardrailLabel.PROMPT_INJECTION
        else:
            return GuardrailLabel.APPROPRIATE

    @track_latency("guardrail_ml_classify")
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text using ML-based approach.

        Args:
            text: Input text to classify

        Returns:
            ClassificationResult with label, confidence, and reasoning
        """
        start_time = time.time()

        X_vectorized = self.vectorizer.transform([text])
        prediction = self.classifier.predict(X_vectorized)[0]
        confidence = max(self.classifier.decision_function(X_vectorized)[0])

        label = self._get_guardrail_label(prediction)
        latency_ms = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            label=label,
            confidence=float(confidence),
            reasoning=f"ML-based classification predicted: {prediction}",
            latency_ms=latency_ms
        )
    
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        results = []
        X_vectorized = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X_vectorized)
        confidences = self.classifier.decision_function(X_vectorized)

        for i, text in enumerate(texts):
            prediction = predictions[i]
            confidence = max(confidences[i])

            label = self._get_guardrail_label(prediction)

            results.append(ClassificationResult(
                label=label,
                confidence=float(confidence),
                reasoning=f"ML-based classification predicted: {prediction}",
                latency_ms=0.0  # Latency not tracked per item in batch
            ))

        metrics.add_metric("guardrail_ml_batch_size", len(texts))
        return results

    def get_classification_stats(self) -> Dict[str, any]:
        """Get classifier statistics."""
        try:
            support_vector_count = int(self.classifier.n_support_.sum()) if hasattr(self.classifier, 'n_support_') else "unknown"
        except:
            support_vector_count = "unknown"

        return {
            "type": "ml_based",
            "model_type": "SVC",
            "vectorizer_type": "TF-IDF",
            "feature_count": len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else "unknown",
            "support_vector_count": support_vector_count,
        }


# Global classifier instance
guardrail_classifier = RuleBasedClassifier()
# guardrail_classifier = MLClassifier()

def get_guardrail_classifier() -> RuleBasedClassifier | MLClassifier:
    """Get the global guardrail classifier instance."""
    return guardrail_classifier


def classify_text(text: str) -> ClassificationResult:
    """Classify text using global classifier."""
    return guardrail_classifier.classify(text)


def is_appropriate(text: str, threshold: float = 0.6) -> bool:
    """Check if text is appropriate based on classification."""
    result = guardrail_classifier.classify(text)
    return (result.label == GuardrailLabel.APPROPRIATE and
            result.confidence >= threshold)