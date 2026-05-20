"""
Multi-Model Ensemble System

Combines predictions from multiple deepfake detectors for:
- Increased reliability
- Ensemble confidence scoring
- Detection of model disagreement
- Weighted voting based on model performance
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Tuple
import streamlit as st


class EnsembleDetector:
    """Manages multiple deepfake detection models and combines predictions."""

    def __init__(self):
        """Initialize ensemble with model registry."""
        self.models = {}  # Dict of model_name -> model
        self.model_weights = {}  # Dict of model_name -> weight (0-1)
        self.model_performance = {}  # Historical performance metrics

    def register_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        description: str = ""
    ) -> None:
        """
        Register a model in the ensemble.

        Args:
            name: Model identifier
            model: Model with predict() method
            weight: Importance weight (higher = more trusted)
            description: Model description
        """
        self.models[name] = model
        self.model_weights[name] = weight
        self.model_performance[name] = {
            "description": description,
            "predictions_count": 0,
            "avg_confidence": 0.0
        }

    def predict_ensemble(self, batch: np.ndarray) -> Dict[str, Any]:
        """
        Generate ensemble prediction from all registered models.

        Args:
            batch: Batch of images (format varies by model)

        Returns:
            Ensemble prediction with voting results and confidence
        """
        if not self.models:
            return {
                "error": "No models registered in ensemble",
                "label": "Unknown",
                "confidence": 0.0
            }

        predictions = {}
        probabilities = []
        model_confidences = []

        # Collect predictions from all models
        for model_name, model in self.models.items():
            try:
                pred = model.predict(batch, verbose=0)

                # Normalize to 2-class probabilities
                pred_normalized = self._normalize_prediction(pred)

                predictions[model_name] = {
                    "fake_prob": float(pred_normalized[0, 0]),
                    "real_prob": float(pred_normalized[0, 1]),
                    "confidence": float(max(pred_normalized[0]))
                }

                probabilities.append(pred_normalized[0])
                model_confidences.append(predictions[model_name]["confidence"])

            except Exception as e:
                predictions[model_name] = {
                    "error": str(e),
                    "fake_prob": 0.5,
                    "real_prob": 0.5,
                    "confidence": 0.0
                }

        # Ensemble voting with weights
        ensemble_result = self._ensemble_voting(predictions, probabilities)

        return {
            "ensemble_prediction": ensemble_result,
            "individual_predictions": predictions,
            "model_count": len([p for p in predictions.values() if "error" not in p]),
            "voting_details": self._get_voting_details(predictions),
            "model_disagreement": self._calculate_disagreement(predictions)
        }

    def _normalize_prediction(self, prediction: Any) -> np.ndarray:
        """Normalize model output to 2-class probability array."""
        probs = np.asarray(prediction, dtype=np.float32)

        # Handle different output shapes
        if probs.ndim == 0:
            fake_prob = float(probs)
            probs = np.array([[fake_prob, 1.0 - fake_prob]], dtype=np.float32)
        elif probs.ndim == 1:
            if probs.size >= 2:
                probs = probs[:2][np.newaxis, :]
            else:
                fake_prob = float(probs[0])
                probs = np.array([[fake_prob, 1.0 - fake_prob]], dtype=np.float32)
        elif probs.ndim == 2:
            if probs.shape[-1] != 2:
                if probs.shape[-1] == 1:
                    fake_prob = probs[:, 0]
                    probs = np.stack([fake_prob, 1.0 - fake_prob], axis=-1)
                else:
                    probs = probs[:, :2]

        # Normalize probabilities
        probs = np.clip(probs, 1e-6, None)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        return probs

    def _ensemble_voting(
        self,
        predictions: Dict[str, Dict],
        probabilities: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform weighted ensemble voting."""

        if not probabilities:
            return {
                "label": "Unknown",
                "fake_confidence": 0.0,
                "real_confidence": 0.0,
                "ensemble_confidence": 0.0,
                "voting_method": "weighted_average"
            }

        # Weighted average voting
        weights = []
        valid_probs = []

        for model_name in self.models.keys():
            if model_name in predictions and "error" not in predictions[model_name]:
                weight = self.model_weights.get(model_name, 1.0)
                weights.append(weight)
                # Find index of this model's prediction
                idx = list([p for p in predictions.keys()]).index(model_name)
                if idx < len(probabilities):
                    valid_probs.append(probabilities[idx])

        if not valid_probs:
            return {
                "label": "Unknown",
                "fake_confidence": 0.0,
                "real_confidence": 0.0,
                "ensemble_confidence": 0.0
            }

        # Normalize weights
        weights = np.array(weights) / sum(weights)
        valid_probs = np.array(valid_probs)

        # Weighted average
        ensemble_probs = np.average(valid_probs, axis=0, weights=weights)

        fake_confidence = float(ensemble_probs[0]) * 100.0
        real_confidence = float(ensemble_probs[1]) * 100.0
        ensemble_confidence = max(fake_confidence, real_confidence)

        label = "Fake" if fake_confidence >= real_confidence else "Real"

        return {
            "label": label,
            "fake_confidence": fake_confidence,
            "real_confidence": real_confidence,
            "ensemble_confidence": ensemble_confidence,
            "voting_method": "weighted_average",
            "models_used": len(valid_probs)
        }

    def _get_voting_details(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Get detailed voting breakdown."""
        fake_votes = 0
        real_votes = 0
        uncertain_votes = 0
        valid_models = 0

        for model_name, pred in predictions.items():
            if "error" in pred:
                continue

            valid_models += 1
            fake_conf = pred.get("fake_prob", 0.5)
            real_conf = pred.get("real_prob", 0.5)

            if fake_conf > 0.6:
                fake_votes += 1
            elif real_conf > 0.6:
                real_votes += 1
            else:
                uncertain_votes += 1

        return {
            "total_valid_models": valid_models,
            "fake_votes": fake_votes,
            "real_votes": real_votes,
            "uncertain_votes": uncertain_votes,
            "majority_verdict": "Fake" if fake_votes > real_votes else "Real" if real_votes > fake_votes else "Uncertain"
        }

    def _calculate_disagreement(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate degree of disagreement between models."""
        valid_predictions = [
            (name, pred) for name, pred in predictions.items()
            if "error" not in pred
        ]

        if len(valid_predictions) < 2:
            return {
                "disagreement_score": 0.0,
                "disagreement_level": "N/A (fewer than 2 models)",
                "models_agreeing": 1,
                "models_disagreeing": 0
            }

        # Get verdict for each model
        verdicts = []
        for name, pred in valid_predictions:
            fake_conf = pred.get("fake_prob", 0.5)
            if fake_conf > 0.5:
                verdicts.append("Fake")
            else:
                verdicts.append("Real")

        # Calculate agreement
        majority_verdict = max(set(verdicts), key=verdicts.count)
        agreeing = verdicts.count(majority_verdict)
        disagreeing = len(verdicts) - agreeing

        disagreement_ratio = disagreeing / len(verdicts) if verdicts else 0

        # Determine disagreement level
        if disagreement_ratio < 0.2:
            disagreement_level = "Very High Agreement"
            disagreement_score = 0.1
        elif disagreement_ratio < 0.4:
            disagreement_level = "High Agreement"
            disagreement_score = 0.25
        elif disagreement_ratio < 0.6:
            disagreement_level = "Moderate Agreement"
            disagreement_score = 0.5
        else:
            disagreement_level = "Low Agreement (Uncertain)"
            disagreement_score = 0.75

        return {
            "disagreement_score": float(disagreement_score),
            "disagreement_level": disagreement_level,
            "models_agreeing": agreeing,
            "models_disagreeing": disagreeing,
            "disagreement_ratio": float(disagreement_ratio),
            "majority_verdict": majority_verdict,
            "verdicts": verdicts
        }

    def update_model_performance(
        self,
        model_name: str,
        confidence: float
    ) -> None:
        """Update performance metrics for a model."""
        if model_name not in self.model_performance:
            return

        perf = self.model_performance[model_name]
        count = perf["predictions_count"]
        old_avg = perf["avg_confidence"]

        # Update running average
        new_avg = (old_avg * count + confidence) / (count + 1)
        perf["avg_confidence"] = new_avg
        perf["predictions_count"] = count + 1

    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "total_models": len(self.models),
            "model_registry": {
                name: {
                    "weight": self.model_weights.get(name, 1.0),
                    "predictions_count": self.model_performance[name]["predictions_count"],
                    "avg_confidence": self.model_performance[name]["avg_confidence"],
                    "description": self.model_performance[name]["description"]
                }
                for name in self.models.keys()
            }
        }

    def adjust_model_weights(self, model_performance_scores: Dict[str, float]) -> None:
        """
        Adjust model weights based on recent performance.

        Args:
            model_performance_scores: Dict of model_name -> performance_score (0-1)
        """
        for model_name, score in model_performance_scores.items():
            if model_name in self.models:
                # Update weight (higher performance = higher weight)
                self.model_weights[model_name] = max(0.1, score)

    def get_model_disagreement_report(self) -> str:
        """Generate a human-readable report of model disagreement."""
        stats = self.get_ensemble_statistics()
        report = "=== Ensemble Model Agreement Report ===\n"

        for model_name, info in stats["model_registry"].items():
            report += f"\n{model_name}:\n"
            report += f"  Weight: {info['weight']:.2f}\n"
            report += f"  Predictions: {info['predictions_count']}\n"
            report += f"  Avg Confidence: {info['avg_confidence']:.2f}%\n"

        return report
