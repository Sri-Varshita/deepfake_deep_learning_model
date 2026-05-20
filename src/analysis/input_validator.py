"""
Input Validation Module - Quality checks for images and videos

Validates input media for:
- Corruption
- Blur detection
- Resolution quality
- Brightness/darkness
- Noise levels
- Completeness

Returns quality assessment and confidence adjustments.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
import streamlit as st


class InputValidator:
    """Validates input quality and returns quality metrics."""

    # Quality thresholds
    MIN_RESOLUTION = 100
    BLUR_THRESHOLD = 100.0  # Laplacian variance threshold
    DARKNESS_THRESHOLD = 30  # Mean pixel intensity below this is dark
    BRIGHTNESS_THRESHOLD = 220  # Mean pixel intensity above this is too bright
    NOISE_THRESHOLD = 0.15  # Noise level threshold

    def __init__(self):
        """Initialize validator with default thresholds."""
        self.quality_issues = []
        self.confidence_penalty = 0.0  # Percentage points to reduce confidence

    def validate_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Validate a single image for quality issues.

        Args:
            image: PIL Image to validate

        Returns:
            Dictionary with quality metrics and issues
        """
        self.quality_issues = []
        self.confidence_penalty = 0.0

        # Convert to numpy array
        img_array = np.array(image.convert("RGB"))

        # Check resolution
        self._check_resolution(image.size)

        # Check blur
        self._check_blur(img_array)

        # Check brightness/contrast
        self._check_brightness_contrast(img_array)

        # Check noise
        self._check_noise(img_array)

        # Check corruption (check image can be processed)
        self._check_corruption(img_array)

        return {
            "is_valid": len(self.quality_issues) == 0,
            "quality_level": self._determine_quality_level(),
            "issues": self.quality_issues,
            "confidence_penalty": self.confidence_penalty,
            "metrics": {
                "resolution": image.size,
                "blur_score": self._calculate_blur_score(img_array),
                "brightness_mean": float(img_array.mean()),
                "contrast_std": float(img_array.std()),
            }
        }

    def validate_video(self, video_path: str, sample_frames: int = 5) -> Dict[str, Any]:
        """
        Validate video quality by sampling frames.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample

        Returns:
            Dictionary with video quality metrics
        """
        self.quality_issues = []
        self.confidence_penalty = 0.0

        try:
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                self.quality_issues.append("Video file cannot be opened")
                return self._get_video_validation_result(video_path, capture, [])

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)

            # Check video properties
            if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
                self.quality_issues.append(
                    f"Low resolution: {width}x{height} (minimum: {self.MIN_RESOLUTION})"
                )
                self.confidence_penalty += 20

            if frame_count < 10:
                self.quality_issues.append(f"Video too short: {frame_count} frames")
                self.confidence_penalty += 15

            # Sample frames for quality check
            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            else:
                frame_indices = list(range(sample_frames))

            sampled_frames = []
            frame_blur_scores = []

            for frame_idx in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = capture.read()
                if ret:
                    sampled_frames.append(frame)
                    blur_score = self._calculate_blur_score(frame)
                    frame_blur_scores.append(blur_score)

            capture.release()

            # Analyze sampled frames
            for frame in sampled_frames:
                self._check_blur(frame)
                self._check_brightness_contrast(frame)
                self._check_noise(frame)

            return self._get_video_validation_result(
                video_path, None, frame_blur_scores,
                metadata={
                    "total_frames": total_frames,
                    "resolution": (width, height),
                    "fps": fps,
                    "sampled_frames": len(sampled_frames)
                }
            )

        except Exception as e:
            self.quality_issues.append(f"Video validation error: {str(e)}")
            self.confidence_penalty += 30
            return self._get_video_validation_result(video_path, None, [])

    def _check_resolution(self, size: Tuple[int, int]):
        """Check if image resolution is acceptable."""
        width, height = size
        if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
            self.quality_issues.append(
                f"Low resolution: {width}x{height} (minimum: {self.MIN_RESOLUTION})"
            )
            self.confidence_penalty += 25

    def _check_blur(self, img_array: np.ndarray):
        """Detect if image is blurry using Laplacian variance."""
        blur_score = self._calculate_blur_score(img_array)
        if blur_score < self.BLUR_THRESHOLD:
            self.quality_issues.append(f"Image is blurry (blur score: {blur_score:.2f})")
            self.confidence_penalty += 20

    def _check_brightness_contrast(self, img_array: np.ndarray):
        """Check brightness and contrast levels."""
        mean_brightness = img_array.mean()
        contrast_std = img_array.std()

        if mean_brightness < self.DARKNESS_THRESHOLD:
            self.quality_issues.append(f"Image too dark (mean: {mean_brightness:.2f})")
            self.confidence_penalty += 20

        if mean_brightness > self.BRIGHTNESS_THRESHOLD:
            self.quality_issues.append(f"Image too bright (mean: {mean_brightness:.2f})")
            self.confidence_penalty += 15

        if contrast_std < 15:
            self.quality_issues.append(f"Low contrast (std: {contrast_std:.2f})")
            self.confidence_penalty += 10

    def _check_noise(self, img_array: np.ndarray):
        """Detect high noise levels using texture analysis."""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.astype(np.uint8)

        # Estimate noise using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var() / (gray.var() + 1e-5)

        if noise_level > self.NOISE_THRESHOLD:
            self.quality_issues.append(f"High noise detected (level: {noise_level:.3f})")
            self.confidence_penalty += 15

    def _check_corruption(self, img_array: np.ndarray):
        """Check if image is corrupted or incomplete."""
        # Check for all-zero or all-max regions (indicates corruption)
        if np.all(img_array == 0) or np.all(img_array == 255):
            self.quality_issues.append("Image appears corrupted (uniform pixel values)")
            self.confidence_penalty += 30

        # Check for NaN or Inf values
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            self.quality_issues.append("Image contains invalid values (NaN/Inf)")
            self.confidence_penalty += 30

    def _calculate_blur_score(self, img_array: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.astype(np.uint8)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        return float(blur_score)

    def _determine_quality_level(self) -> str:
        """Determine overall quality level based on penalties."""
        if self.confidence_penalty == 0:
            return "High"
        elif self.confidence_penalty <= 15:
            return "Good"
        elif self.confidence_penalty <= 30:
            return "Fair"
        else:
            return "Low"

    def _get_video_validation_result(
        self,
        video_path: str,
        capture: Any,
        blur_scores: list,
        metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Build video validation result dictionary."""
        if capture:
            capture.release()

        return {
            "is_valid": len(self.quality_issues) == 0,
            "quality_level": self._determine_quality_level(),
            "issues": self.quality_issues,
            "confidence_penalty": self.confidence_penalty,
            "metrics": {
                "video_path": video_path,
                "avg_blur_score": float(np.mean(blur_scores)) if blur_scores else 0,
                "blur_scores": blur_scores,
                **(metadata or {})
            }
        }
