"""
Advanced Decision Engine

Applies the strict analysis pipeline decision rules:
- Confidence-based decisions (≥90%, 70-89%, <70%)
- Multi-factor analysis
- Uncertainty handling
- Reasoning generation

Decision Rules:
- Confidence ≥90%: High confidence
- Confidence 70-89%: Moderate confidence
- Confidence <70%: Uncertain result
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""
    HIGH = "High"  # ≥ 90%
    MODERATE = "Moderate"  # 70-89%
    LOW = "Low"  # 50-69%
    UNCERTAIN = "Uncertain"  # < 50%


class DecisionEngine:
    """Advanced decision engine for deepfake classification."""

    # Decision thresholds
    HIGH_CONFIDENCE_THRESHOLD = 90.0
    MODERATE_CONFIDENCE_THRESHOLD = 70.0
    LOW_CONFIDENCE_THRESHOLD = 50.0

    # Weight factors for multi-factor analysis
    MODEL_PREDICTION_WEIGHT = 0.40
    INPUT_QUALITY_WEIGHT = 0.20
    FACIAL_CONSISTENCY_WEIGHT = 0.20
    ARTIFACT_DETECTION_WEIGHT = 0.20

    def __init__(self):
        """Initialize decision engine."""
        self.decision_history = []

    def make_decision(
        self,
        model_prediction: Dict[str, Any],
        input_quality: Dict[str, Any],
        facial_consistency: Dict[str, Any],
        artifact_detection: Dict[str, Any],
        ensemble_info: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Make final deepfake classification decision using multi-factor analysis.

        Args:
            model_prediction: Primary model prediction result
            input_quality: Input quality validation results
            facial_consistency: Facial consistency analysis results
            artifact_detection: Artifact detection results
            ensemble_info: Optional ensemble prediction information

        Returns:
            Final decision with confidence and reasoning
        """

        # Step 1: Extract base prediction
        base_label = model_prediction.get("label", "Unknown")
        base_confidence = model_prediction.get("confidence", 50.0)
        base_fake_confidence = model_prediction.get("fake_confidence", 50.0)
        base_real_confidence = model_prediction.get("real_confidence", 50.0)

        # Step 2: Calculate quality adjustment factor
        quality_penalty = input_quality.get("confidence_penalty", 0.0)
        quality_level = input_quality.get("quality_level", "Fair")
        quality_factor = 1.0 - (quality_penalty / 100.0)

        # Step 3: Calculate facial consistency score (0-1)
        facial_consistency_score = facial_consistency.get("consistency_score", 0.5)
        facial_issues_count = len(facial_consistency.get("issues", []))
        facial_artifacts_count = len(facial_consistency.get("facial_artifacts", []))

        # Step 4: Extract artifact information
        artifact_score = artifact_detection.get("artifact_score", 0.5)
        has_artifacts = artifact_detection.get("has_artifacts", False)
        artifacts_found = len(artifact_detection.get("artifacts", []))

        # Step 5: Consider ensemble information
        ensemble_disagreement = 0.0
        ensemble_label = None
        if ensemble_info:
            ensemble_disagreement = ensemble_info.get("model_disagreement", {}).get("disagreement_score", 0.0)
            ensemble_pred = ensemble_info.get("ensemble_prediction", {})
            ensemble_label = ensemble_pred.get("label")

        # Step 6: Synthesize all factors
        final_decision = self._synthesize_decision(
            base_label=base_label,
            base_confidence=base_confidence,
            base_fake_confidence=base_fake_confidence,
            base_real_confidence=base_real_confidence,
            quality_level=quality_level,
            quality_factor=quality_factor,
            facial_consistency_score=facial_consistency_score,
            facial_issues_count=facial_issues_count,
            facial_artifacts_count=facial_artifacts_count,
            artifact_score=artifact_score,
            has_artifacts=has_artifacts,
            artifacts_found=artifacts_found,
            ensemble_disagreement=ensemble_disagreement,
            ensemble_label=ensemble_label,
            input_quality=input_quality,
            facial_consistency=facial_consistency,
            artifact_detection=artifact_detection
        )

        return final_decision

    def _synthesize_decision(self, **kwargs) -> Dict[str, Any]:
        """Synthesize all analysis factors into final decision."""

        # Extract parameters
        base_label = kwargs["base_label"]
        base_confidence = kwargs["base_confidence"]
        base_fake_confidence = kwargs["base_fake_confidence"]
        base_real_confidence = kwargs["base_real_confidence"]
        quality_level = kwargs["quality_level"]
        quality_factor = kwargs["quality_factor"]
        facial_consistency_score = kwargs["facial_consistency_score"]
        facial_issues_count = kwargs["facial_issues_count"]
        artifact_score = kwargs["artifact_score"]
        has_artifacts = kwargs["has_artifacts"]
        ensemble_disagreement = kwargs["ensemble_disagreement"]

        # Calculate adjusted confidence
        # Step 1: Start with model prediction
        adjusted_fake_conf = base_fake_confidence

        # Step 2: Apply quality adjustment
        adjusted_fake_conf *= quality_factor

        # Step 3: Apply facial consistency adjustment
        # High consistency suggests real, low consistency suggests fake
        consistency_factor = 1.0 - (facial_consistency_score * 0.2)
        adjusted_fake_conf *= consistency_factor

        # Step 4: Apply artifact detection adjustment
        # More artifacts suggest higher likelihood of fake
        artifact_factor = 1.0 + (artifact_score * 0.3)
        adjusted_fake_conf *= artifact_factor

        # Step 5: Consider ensemble disagreement
        # High disagreement reduces confidence
        if ensemble_disagreement > 0.5:
            adjusted_fake_conf *= (1.0 - ensemble_disagreement * 0.2)

        # Normalize to 0-100 range
        adjusted_fake_conf = np.clip(adjusted_fake_conf, 0, 100)
        adjusted_real_conf = 100.0 - adjusted_fake_conf
        final_confidence = max(adjusted_fake_conf, adjusted_real_conf)

        # Determine confidence level
        confidence_level = self._determine_confidence_level(final_confidence)

        # Determine final label
        final_label = "Fake" if adjusted_fake_conf > adjusted_real_conf else "Real"

        # Generate reasoning
        reasoning = self._generate_reasoning(
            final_label=final_label,
            final_confidence=final_confidence,
            base_label=base_label,
            base_confidence=base_confidence,
            quality_level=quality_level,
            quality_factor=quality_factor,
            facial_consistency_score=facial_consistency_score,
            facial_issues_count=facial_issues_count,
            artifact_score=artifact_score,
            has_artifacts=has_artifacts,
            ensemble_disagreement=ensemble_disagreement,
            confidence_level=confidence_level,
            **kwargs
        )

        decision = {
            "final_label": final_label,
            "final_confidence": final_confidence,
            "fake_confidence": adjusted_fake_conf,
            "real_confidence": adjusted_real_conf,
            "confidence_level": confidence_level.value,
            "reasoning": reasoning,
            "decision_factors": self._get_decision_factors(
                quality_factor=quality_factor,
                facial_consistency_score=facial_consistency_score,
                artifact_score=artifact_score,
                ensemble_disagreement=ensemble_disagreement
            ),
            "recommendation": self._get_recommendation(confidence_level, final_label)
        }

        # Store in history
        self.decision_history.append(decision)

        return decision

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on percentage."""
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif confidence >= self.MODERATE_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MODERATE
        elif confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _generate_reasoning(self, **kwargs) -> List[str]:
        """Generate human-readable reasoning for the decision."""
        reasoning = []

        final_label = kwargs["final_label"]
        final_confidence = kwargs["final_confidence"]
        base_label = kwargs["base_label"]
        quality_level = kwargs["quality_level"]
        quality_factor = kwargs["quality_factor"]
        facial_consistency_score = kwargs["facial_consistency_score"]
        facial_issues_count = kwargs["facial_issues_count"]
        facial_artifacts_count = kwargs["facial_artifacts_count"]
        artifact_score = kwargs["artifact_score"]
        has_artifacts = kwargs["has_artifacts"]
        ensemble_disagreement = kwargs["ensemble_disagreement"]
        confidence_level = kwargs["confidence_level"]
        input_quality = kwargs["input_quality"]
        facial_consistency = kwargs["facial_consistency"]
        artifact_detection = kwargs["artifact_detection"]

        # Primary classification reason
        if final_label == "Fake":
            reasoning.append(
                f"The model classifies this content as DEEPFAKE with {final_confidence:.1f}% "
                f"confidence ({confidence_level.value.lower()})."
            )
        else:
            reasoning.append(
                f"The model classifies this content as AUTHENTIC with {final_confidence:.1f}% "
                f"confidence ({confidence_level.value.lower()})."
            )

        # Input quality reasons
        if quality_level == "High":
            reasoning.append("Input quality is excellent - all quality checks passed.")
        elif quality_level == "Good":
            reasoning.append("Input quality is good with minimal quality issues.")
        elif quality_level == "Fair":
            reasoning.append("Input quality is fair with some quality concerns that may affect confidence.")
        else:
            reasoning.append("Input quality is poor. Confidence has been reduced due to quality issues.")
            for issue in input_quality.get("issues", [])[:3]:
                reasoning.append(f"  - {issue}")

        # Facial consistency reasons
        if facial_consistency_score > 0.7:
            reasoning.append("Facial analysis shows high consistency with authentic characteristics.")
        elif facial_consistency_score > 0.4:
            reasoning.append("Facial analysis shows moderate consistency.")
        else:
            reasoning.append("Facial analysis reveals inconsistencies suggesting synthetic origin.")

        if facial_issues_count > 0:
            top_issues = facial_consistency.get("issues", [])[:2]
            for issue in top_issues:
                reasoning.append(f"  - Facial issue detected: {issue}")

        if facial_artifacts_count > 0:
            top_artifacts = facial_consistency.get("facial_artifacts", [])[:2]
            for artifact in top_artifacts:
                reasoning.append(f"  - Artifact found: {artifact}")

        # Artifact detection reasons
        if has_artifacts:
            reasoning.append(
                f"Multiple artifacts detected (score: {artifact_score:.2f}), "
                f"which is consistent with deepfake generation methods."
            )
            artifacts_list = artifact_detection.get("artifacts", [])[:3]
            for artifact in artifacts_list:
                reasoning.append(f"  - {artifact}")
        else:
            if artifact_score < 0.3:
                reasoning.append("Minimal artifacts detected - content appears naturally generated.")
            else:
                reasoning.append("Some artifacts detected but not strongly conclusive.")

        # Ensemble disagreement
        if ensemble_disagreement > 0.6:
            reasoning.append(
                "Models show significant disagreement on this classification. "
                "The result should be considered uncertain."
            )
        elif ensemble_disagreement > 0.3:
            reasoning.append("Some disagreement among detection models increases uncertainty.")

        # Confidence-based advice
        if confidence_level == ConfidenceLevel.HIGH:
            reasoning.append(
                "HIGH CONFIDENCE - This prediction is highly reliable based on multiple "
                "analysis factors."
            )
        elif confidence_level == ConfidenceLevel.MODERATE:
            reasoning.append(
                "MODERATE CONFIDENCE - This prediction is reasonably reliable but should be "
                "verified if critical decision is needed."
            )
        elif confidence_level == ConfidenceLevel.LOW:
            reasoning.append(
                "LOW CONFIDENCE - Multiple factors suggest uncertainty. Manual review is recommended."
            )
        else:
            reasoning.append(
                "UNCERTAIN - The classification is inconclusive. Multiple analysis factors "
                "contradict each other. Manual verification is highly recommended."
            )

        return reasoning

    def _get_decision_factors(self, **kwargs) -> Dict[str, float]:
        """Get numeric breakdown of decision factors."""
        return {
            "quality_factor": float(kwargs.get("quality_factor", 1.0)),
            "facial_consistency_factor": float(kwargs.get("facial_consistency_score", 0.5)),
            "artifact_factor": float(kwargs.get("artifact_score", 0.5)),
            "ensemble_agreement_factor": 1.0 - float(kwargs.get("ensemble_disagreement", 0.0))
        }

    def _get_recommendation(self, confidence_level: ConfidenceLevel, label: str) -> str:
        """Get recommendation based on confidence level."""
        if confidence_level == ConfidenceLevel.HIGH:
            return (
                f"✅ High confidence in {label} classification. "
                "This result is suitable for decision-making."
            )
        elif confidence_level == ConfidenceLevel.MODERATE:
            return (
                f"⚠️ Moderate confidence in {label} classification. "
                "Verify with additional checks if critical decisions depend on this result."
            )
        elif confidence_level == ConfidenceLevel.LOW:
            return (
                f"⚠️ Low confidence in classification. "
                "Manual review strongly recommended before making any decisions."
            )
        else:
            return (
                "❌ Uncertain classification. Cannot reliably determine authenticity. "
                "This content requires manual expert review."
            )

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of all decisions made."""
        return self.decision_history

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics from decision history."""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "statistics": "No decisions made yet"
            }

        labels = [d["final_label"] for d in self.decision_history]
        confidence_levels = [d["confidence_level"] for d in self.decision_history]
        confidences = [d["final_confidence"] for d in self.decision_history]

        return {
            "total_decisions": len(self.decision_history),
            "fake_count": labels.count("Fake"),
            "real_count": labels.count("Real"),
            "avg_confidence": np.mean(confidences),
            "confidence_level_distribution": {
                level: confidence_levels.count(level) for level in ["High", "Moderate", "Low", "Uncertain"]
            }
        }
