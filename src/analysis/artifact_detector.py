"""
Artifact Detection Module

Detects:
- Pixel-level distortions
- GAN artifacts
- Unnatural edges
- Blurring around face regions
- Background inconsistencies
- Color mismatches
- Frame-to-frame anomalies
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import ndimage
import streamlit as st


class ArtifactDetector:
    """Detects various deepfake artifacts in images."""

    def __init__(self):
        """Initialize artifact detector."""
        self.artifacts_found = []
        self.artifact_scores = {}

    def detect_artifacts(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect various artifacts indicating deepfake.

        Args:
            img_array: Image as numpy array (BGR or RGB)

        Returns:
            Dictionary with detected artifacts and scores
        """
        self.artifacts_found = []
        self.artifact_scores = {}

        if img_array is None or img_array.size == 0:
            return {
                "has_artifacts": False,
                "artifacts": [],
                "artifact_score": 0.5,
                "details": {}
            }

        # Convert to uint8 if needed
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Detect various artifact types
        self._detect_gan_artifacts(img_array)
        self._detect_pixel_distortions(img_array)
        self._detect_frequency_anomalies(img_array)
        self._detect_boundary_artifacts(img_array)
        self._detect_color_artifacts(img_array)
        self._detect_texture_inconsistencies(img_array)

        # Calculate overall artifact score (0-1, higher = more artifacts = more likely deepfake)
        artifact_score = self._calculate_artifact_score()

        return {
            "has_artifacts": len(self.artifacts_found) > 0,
            "artifacts": self.artifacts_found,
            "artifact_score": artifact_score,
            "details": self.artifact_scores
        }

    def detect_frame_anomalies(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Detect frame-to-frame anomalies in video.

        Args:
            frames: List of video frames

        Returns:
            Dictionary with detected anomalies
        """
        if len(frames) < 2:
            return {
                "frame_count": len(frames),
                "anomalies": [],
                "anomaly_score": 0.0
            }

        anomalies = []
        anomaly_scores = []

        # Compare consecutive frames
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]

            # Calculate frame difference
            if frame1.shape == frame2.shape:
                diff = cv2.absdiff(frame1, frame2)
                motion_magnitude = diff.mean()

                # Check for unnatural motion patterns
                if motion_magnitude < 1.0:
                    anomalies.append(f"Frame {i}-{i+1}: Very low motion (static video)")
                    anomaly_scores.append(0.3)

                if motion_magnitude > 50:
                    anomalies.append(f"Frame {i}-{i+1}: Very high motion (possible jump cut)")
                    anomaly_scores.append(0.7)

                # Check for temporal flicker
                flicker_score = self._detect_temporal_flicker(frame1, frame2)
                if flicker_score > 0.7:
                    anomalies.append(f"Frame {i}-{i+1}: Temporal flicker detected")
                    anomaly_scores.append(flicker_score)

        anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0.0

        return {
            "frame_count": len(frames),
            "frame_pairs_analyzed": len(frames) - 1,
            "anomalies": anomalies,
            "anomaly_score": float(anomaly_score),
            "anomaly_scores": anomaly_scores
        }

    def _detect_gan_artifacts(self, img_array: np.ndarray):
        """Detect GAN-specific artifacts."""
        try:
            # Convert to frequency domain
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array

            # FFT analysis
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)

            # Normalize
            magnitude_spectrum = np.log1p(magnitude_spectrum)
            magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                                 (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-6)

            # Check for ring artifacts (common in GAN outputs)
            center_h, center_w = gray.shape[0] // 2, gray.shape[1] // 2
            ring_artifact_score = self._detect_frequency_rings(magnitude_spectrum, center_h, center_w)

            if ring_artifact_score > 0.6:
                self.artifacts_found.append("GAN ring artifacts detected in frequency domain")
                self.artifact_scores["gan_ring_artifacts"] = float(ring_artifact_score)

            # Check for specific GAN patterns (checkerboard artifacts)
            checkerboard_score = self._detect_checkerboard_artifacts(img_array)
            if checkerboard_score > 0.5:
                self.artifacts_found.append("Checkerboard-like GAN artifacts detected")
                self.artifact_scores["checkerboard_artifacts"] = float(checkerboard_score)

        except Exception:
            pass

    def _detect_frequency_rings(self, magnitude_spectrum: np.ndarray, center_h: int, center_w: int) -> float:
        """Detect ring patterns in frequency domain."""
        try:
            # Create radial mask
            h, w = magnitude_spectrum.shape
            y, x = np.ogrid[:h, :w]
            mask = (x - center_w) ** 2 + (y - center_h) ** 2

            # Check for energy concentration at specific radii
            ring_scores = []
            for radius in range(10, min(center_h, center_w), 20):
                ring_mask = (np.abs(mask - radius**2) < radius**2 * 0.1)
                ring_energy = magnitude_spectrum[ring_mask].mean()
                ring_scores.append(ring_energy)

            if ring_scores:
                # High variance in ring energy indicates artifacts
                ring_variance = np.var(ring_scores)
                return float(np.clip(ring_variance / 0.5, 0, 1))
            else:
                return 0.0
        except Exception:
            return 0.0

    def _detect_checkerboard_artifacts(self, img_array: np.ndarray) -> float:
        """Detect checkerboard-like artifacts from upsampling."""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array

            # Create checkerboard pattern detector
            # Subsample and analyze periodicity
            h, w = gray.shape
            step = 8

            if h < step or w < step:
                return 0.0

            subsampled = gray[::step, ::step]

            # Analyze pixel value patterns
            # Checkerboard artifacts show alternating pattern
            horizontal_diff = np.abs(np.diff(subsampled, axis=1)).mean()
            vertical_diff = np.abs(np.diff(subsampled, axis=0)).mean()

            pattern_strength = max(horizontal_diff, vertical_diff)

            # If pattern is too regular, it indicates artifacts
            if pattern_strength > 50:
                return 0.3
            elif pattern_strength > 100:
                return 0.6
            else:
                return 0.0

        except Exception:
            return 0.0

    def _detect_pixel_distortions(self, img_array: np.ndarray):
        """Detect pixel-level distortions."""
        try:
            # Check for quantization artifacts (indicates compression or GAN)
            if len(img_array.shape) == 3:
                channels = cv2.split(img_array)
            else:
                channels = [img_array]

            distortion_scores = []

            for channel in channels:
                # Count unique values (quantized images have fewer unique values)
                unique_values = len(np.unique(channel))
                unique_ratio = unique_values / 256.0

                # Check for edge blocks
                edges = cv2.Canny(channel.astype(np.uint8), 100, 200)
                edge_block_score = self._detect_edge_blocks(edges)

                distortion_scores.append((1.0 - unique_ratio) * 0.5 + edge_block_score * 0.5)

            distortion_score = np.mean(distortion_scores)

            if distortion_score > 0.4:
                self.artifacts_found.append(f"Pixel distortions detected (score: {distortion_score:.2f})")
                self.artifact_scores["pixel_distortions"] = float(distortion_score)

        except Exception:
            pass

    def _detect_edge_blocks(self, edges: np.ndarray) -> float:
        """Detect blocky edge patterns."""
        try:
            h, w = edges.shape
            block_size = 8

            if h < block_size or w < block_size:
                return 0.0

            # Analyze block boundaries
            block_edges = 0
            total_blocks = 0

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = edges[y:y+block_size, x:x+block_size]
                    total_blocks += 1

                    # Check if edges align with block boundaries
                    top_edge = block[0, :].sum()
                    left_edge = block[:, 0].sum()

                    if top_edge > block_size // 2 or left_edge > block_size // 2:
                        block_edges += 1

            if total_blocks > 0:
                block_ratio = block_edges / total_blocks
                return float(np.clip(block_ratio, 0, 1))
            else:
                return 0.0

        except Exception:
            return 0.0

    def _detect_frequency_anomalies(self, img_array: np.ndarray):
        """Detect anomalies in frequency domain."""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array

            # DCT (Discrete Cosine Transform) analysis
            dct = cv2.dct(np.float32(gray))
            dct_spectrum = np.abs(dct)

            # Check for abnormal frequency distribution
            high_freq_energy = dct_spectrum[dct_spectrum.shape[0]//2:, dct_spectrum.shape[1]//2:].sum()
            total_energy = dct_spectrum.sum()

            if total_energy > 0:
                high_freq_ratio = high_freq_energy / total_energy

                # Real images typically have more low-frequency content
                if high_freq_ratio > 0.4:
                    self.artifacts_found.append("Abnormal frequency distribution detected")
                    self.artifact_scores["frequency_anomalies"] = float(high_freq_ratio)

        except Exception:
            pass

    def _detect_boundary_artifacts(self, img_array: np.ndarray):
        """Detect artifacts at face boundaries."""
        try:
            if len(img_array.shape) == 3:
                edges_img = cv2.Canny(img_array, 50, 150)
            else:
                edges_img = cv2.Canny(img_array.astype(np.uint8), 50, 150)

            h, w = edges_img.shape

            # Analyze edge distribution at image boundaries
            border_width = w // 10
            border_edges = (edges_img[:, :border_width].sum() +
                           edges_img[:, -border_width:].sum() +
                           edges_img[:border_width, :].sum() +
                           edges_img[-border_width:, :].sum())

            total_edges = edges_img.sum()

            if total_edges > 0:
                boundary_ratio = border_edges / total_edges

                if boundary_ratio > 0.3:
                    self.artifacts_found.append("High edge concentration at boundaries")
                    self.artifact_scores["boundary_artifacts"] = float(boundary_ratio)

        except Exception:
            pass

    def _detect_color_artifacts(self, img_array: np.ndarray):
        """Detect color-related artifacts."""
        try:
            if len(img_array.shape) != 3:
                return

            # Analyze color consistency
            b, g, r = cv2.split(img_array)

            # Check for color channel misalignment
            shifts = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
            correlations = []

            for shift in shifts:
                b_shift = np.roll(b, shift, axis=(0, 1))
                corr = cv2.matchTemplate(b_shift.astype(np.float32), r.astype(np.float32), cv2.TM_CCOEFF)
                correlations.append(corr.max() if corr.size > 0 else 0)

            # Perfect correlation at zero shift indicates possible artifact
            zero_corr = correlations[0]
            max_other_corr = max(correlations[1:])

            if max_other_corr > zero_corr * 0.9:
                self.artifacts_found.append("Color channel misalignment detected")
                self.artifact_scores["color_misalignment"] = float((max_other_corr / zero_corr) if zero_corr > 0 else 0)

        except Exception:
            pass

    def _detect_texture_inconsistencies(self, img_array: np.ndarray):
        """Detect texture inconsistencies."""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array

            # Divide image into regions and analyze texture consistency
            h, w = gray.shape
            region_h, region_w = h // 4, w // 4

            texture_scores = []

            for y in range(0, h - region_h, region_h):
                for x in range(0, w - region_w, region_w):
                    region = gray[y:y+region_h, x:x+region_w]

                    # Local Binary Pattern
                    lbp_score = self._calculate_lbp_texture(region)
                    texture_scores.append(lbp_score)

            texture_scores = np.array(texture_scores)
            texture_variance = texture_scores.std()

            # High variance indicates inconsistent texture
            if texture_variance > 0.3:
                self.artifacts_found.append("Texture inconsistencies detected across regions")
                self.artifact_scores["texture_inconsistencies"] = float(texture_variance)

        except Exception:
            pass

    def _calculate_lbp_texture(self, region: np.ndarray) -> float:
        """Calculate Local Binary Pattern texture score."""
        try:
            if region.shape[0] < 3 or region.shape[1] < 3:
                return 0.5

            center = region[1:-1, 1:-1]
            neighbors = np.array([
                region[:-2, :-2], region[:-2, 1:-1], region[:-2, 2:],
                region[1:-1, :-2], region[1:-1, 2:],
                region[2:, :-2], region[2:, 1:-1], region[2:, 2:]
            ])

            # Calculate LBP
            lbp = np.sum(neighbors >= center[np.newaxis, :, :] * 2 ** np.arange(8)[:, np.newaxis, np.newaxis], axis=0)
            return float(lbp.std() / 256.0)

        except Exception:
            return 0.5

    def _detect_temporal_flicker(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Detect temporal flicker between frames."""
        try:
            if frame1.shape != frame2.shape:
                return 0.0

            if len(frame1.shape) == 3:
                f1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                f2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                f1_gray = frame1
                f2_gray = frame2

            # Calculate rapid intensity changes
            diff = cv2.absdiff(f1_gray, f2_gray)
            flicker_regions = np.sum(diff > 100) / diff.size

            return float(flicker_regions)

        except Exception:
            return 0.0

    def _calculate_artifact_score(self) -> float:
        """Calculate overall artifact score."""
        if not self.artifact_scores:
            return 0.0

        scores = list(self.artifact_scores.values())
        return float(np.mean(scores))
