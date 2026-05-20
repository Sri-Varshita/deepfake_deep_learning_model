"""
Facial Consistency Analysis Module

Analyzes:
- Eye blinking patterns
- Lip movement synchronization
- Head movement consistency
- Facial expression transitions
- Skin texture consistency
- Facial boundary artifacts
- Lighting and shadow consistency
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import streamlit as st

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FacialConsistencyAnalyzer:
    """Analyzes facial features for consistency and authenticity."""

    def __init__(self):
        """Initialize facial analyzer with MediaPipe detection."""
        self.face_detector = None
        self.landmarks_detector = None
        self.eye_aspect_ratio_history = []
        self.head_pose_history = []
        self.lip_distance_history = []

        if MEDIAPIPE_AVAILABLE:
            try:
                self._initialize_mediapipe()
            except Exception as e:
                st.warning(f"MediaPipe initialization: {str(e)}")

    def _initialize_mediapipe(self):
        """Initialize MediaPipe face detection and landmark detection."""
        try:
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh

            self.face_detector = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            self.landmarks_detector = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except Exception as e:
            st.warning(f"Could not initialize MediaPipe: {str(e)}")
            self.face_detector = None
            self.landmarks_detector = None

    def analyze_image(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze facial consistency in a single image.

        Args:
            img_array: Image as numpy array (BGR or RGB)

        Returns:
            Dictionary with facial consistency metrics
        """
        analysis = {
            "has_faces": False,
            "face_count": 0,
            "facial_artifacts": [],
            "consistency_score": 0.5,  # 0-1, higher = more authentic
            "issues": [],
            "details": {}
        }

        if img_array is None or img_array.size == 0:
            analysis["issues"].append("Image is empty or invalid")
            return analysis

        # Convert to RGB if BGR
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            try:
                rgb_image = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2RGB)
            except:
                rgb_image = img_array.astype(np.uint8)
        else:
            rgb_image = img_array.astype(np.uint8)

        # Detect faces
        faces = self._detect_faces(rgb_image)
        analysis["has_faces"] = len(faces) > 0
        analysis["face_count"] = len(faces)

        if not faces:
            analysis["issues"].append("No faces detected in image")
            return analysis

        # Analyze each face
        for face_idx, face_box in enumerate(faces):
            face_analysis = self._analyze_face(rgb_image, face_box)
            analysis["details"][f"face_{face_idx}"] = face_analysis

            # Aggregate artifacts and issues
            analysis["facial_artifacts"].extend(face_analysis.get("artifacts", []))
            analysis["issues"].extend(face_analysis.get("issues", []))

        # Calculate overall consistency score
        analysis["consistency_score"] = self._calculate_consistency_score(analysis)

        return analysis

    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze facial consistency across video frames.

        Args:
            frames: List of video frames

        Returns:
            Dictionary with temporal facial analysis
        """
        if not frames:
            return {
                "is_valid": False,
                "issues": ["No frames to analyze"],
                "temporal_consistency": 0.5
            }

        frame_analyses = []
        eye_aspect_ratios = []
        head_poses = []
        lip_distances = []

        for frame in frames:
            analysis = self.analyze_image(frame)
            frame_analyses.append(analysis)

            # Extract metrics if available
            if "details" in analysis:
                for face_data in analysis["details"].values():
                    if "eye_aspect_ratio" in face_data:
                        eye_aspect_ratios.append(face_data["eye_aspect_ratio"])
                    if "head_pose" in face_data:
                        head_poses.append(face_data["head_pose"])
                    if "lip_distance" in face_data:
                        lip_distances.append(face_data["lip_distance"])

        # Analyze temporal consistency
        temporal_analysis = self._analyze_temporal_consistency(
            frame_analyses,
            eye_aspect_ratios,
            head_poses,
            lip_distances
        )

        return temporal_analysis

    def _detect_faces(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe."""
        faces = []

        if not self.face_detector:
            # Fallback: use simple face detection with Haar Cascade
            return self._fallback_face_detection(rgb_image)

        try:
            results = self.face_detector.process(rgb_image)
            if results.detections:
                h, w, c = rgb_image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    faces.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": detection.score[0] if detection.score else 0.5
                    })
        except Exception as e:
            # Fallback to Haar Cascade
            faces = self._fallback_face_detection(rgb_image)

        return faces

    def _fallback_face_detection(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback face detection using Haar Cascade."""
        faces = []
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            detected = cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in detected:
                faces.append({
                    "box": (x, y, x + w, y + h),
                    "confidence": 0.7
                })
        except Exception:
            pass

        return faces

    def _analyze_face(self, rgb_image: np.ndarray, face_box: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single detected face."""
        face_analysis = {
            "box": face_box["box"],
            "confidence": face_box.get("confidence", 0.5),
            "artifacts": [],
            "issues": [],
            "metrics": {}
        }

        x1, y1, x2, y2 = face_box["box"]

        # Ensure valid box
        if x1 >= x2 or y1 >= y2:
            face_analysis["issues"].append("Invalid face bounding box")
            return face_analysis

        # Crop face region
        face_roi = rgb_image[y1:y2, x1:x2]

        if face_roi.size == 0:
            face_analysis["issues"].append("Face region is empty")
            return face_analysis

        # Detect facial landmarks
        landmarks = self._detect_landmarks(rgb_image)

        if landmarks:
            # Analyze facial features
            eye_ratio = self._calculate_eye_aspect_ratio(landmarks)
            face_analysis["eye_aspect_ratio"] = eye_ratio

            lip_distance = self._calculate_lip_distance(landmarks)
            face_analysis["lip_distance"] = lip_distance

            head_pose = self._estimate_head_pose(landmarks)
            face_analysis["head_pose"] = head_pose

            # Check for facial boundary artifacts
            boundary_artifacts = self._detect_boundary_artifacts(face_roi, landmarks)
            face_analysis["artifacts"].extend(boundary_artifacts)

        # Analyze skin texture
        texture_artifacts = self._analyze_skin_texture(face_roi)
        face_analysis["artifacts"].extend(texture_artifacts)

        # Check lighting consistency
        lighting_issues = self._check_lighting_consistency(face_roi)
        face_analysis["issues"].extend(lighting_issues)

        # Analyze facial color consistency
        color_issues = self._analyze_color_consistency(face_roi)
        face_analysis["issues"].extend(color_issues)

        return face_analysis

    def _detect_landmarks(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks using MediaPipe."""
        if not self.landmarks_detector:
            return None

        try:
            results = self.landmarks_detector.process(rgb_image)
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                landmarks = results.multi_face_landmarks[0]
                h, w, c = rgb_image.shape

                landmarks_array = np.array([
                    [lm.x * w, lm.y * h, lm.z] for lm in landmarks.landmark
                ])
                return landmarks_array
        except Exception:
            pass

        return None

    def _calculate_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate eye aspect ratio (blink indicator)."""
        try:
            # Simplified eye aspect ratio calculation
            # Using MediaPipe landmark indices for left eye
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 398, 388, 374]

            if landmarks.shape[0] > max(left_eye_indices + right_eye_indices):
                left_eye = landmarks[left_eye_indices]
                right_eye = landmarks[right_eye_indices]

                # Calculate vertical distances
                left_ear = self._calculate_eye_distance(left_eye)
                right_ear = self._calculate_eye_distance(right_eye)

                return (left_ear + right_ear) / 2.0
            else:
                return 0.5
        except Exception:
            return 0.5

    def _calculate_eye_distance(self, eye_points: np.ndarray) -> float:
        """Calculate distance metric for eye points."""
        try:
            # Distance between vertical points
            vertical_dist = np.linalg.norm(eye_points[1] - eye_points[4])
            # Distance between horizontal points
            horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])

            return vertical_dist / (horizontal_dist + 1e-6)
        except Exception:
            return 0.5

    def _calculate_lip_distance(self, landmarks: np.ndarray) -> float:
        """Calculate lip opening distance."""
        try:
            # MediaPipe lip indices
            upper_lip_indices = [13, 312]
            lower_lip_indices = [14, 87]

            if landmarks.shape[0] > max(upper_lip_indices + lower_lip_indices):
                upper_lip = landmarks[upper_lip_indices].mean(axis=0)
                lower_lip = landmarks[lower_lip_indices].mean(axis=0)

                lip_distance = np.linalg.norm(upper_lip - lower_lip)
                return float(lip_distance)
            else:
                return 0.5
        except Exception:
            return 0.5

    def _estimate_head_pose(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Estimate head pose angles."""
        try:
            # Use nose and eye positions to estimate head pose
            nose_idx = 0
            left_eye_idx = 33
            right_eye_idx = 263

            if landmarks.shape[0] > right_eye_idx:
                nose = landmarks[nose_idx]
                left_eye = landmarks[left_eye_idx]
                right_eye = landmarks[right_eye_idx]

                # Calculate yaw (left-right rotation)
                yaw = np.arctan2(left_eye[0] - right_eye[0], left_eye[2] - right_eye[2])

                # Calculate pitch (up-down rotation)
                eye_center = (left_eye + right_eye) / 2
                pitch = np.arctan2(nose[1] - eye_center[1], nose[2] - eye_center[2])

                return {
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                }
            else:
                return {"yaw": 0.0, "pitch": 0.0}
        except Exception:
            return {"yaw": 0.0, "pitch": 0.0}

    def _detect_boundary_artifacts(self, face_roi: np.ndarray, landmarks: np.ndarray) -> List[str]:
        """Detect artifacts around facial boundaries."""
        artifacts = []

        try:
            # Check for unnatural edges using edge detection
            gray = cv2.cvtColor(face_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Calculate edge density at boundaries
            h, w = edges.shape
            boundary_edges_left = edges[:, :w//10].sum()
            boundary_edges_right = edges[:, 9*w//10:].sum()
            boundary_edges_top = edges[:h//10, :].sum()
            boundary_edges_bottom = edges[9*h//10:, :].sum()

            total_edges = edges.sum()
            if total_edges > 0:
                boundary_edge_ratio = (boundary_edges_left + boundary_edges_right +
                                       boundary_edges_top + boundary_edges_bottom) / total_edges

                if boundary_edge_ratio > 0.3:
                    artifacts.append("High edge density at facial boundaries")

            # Check for blur around face edges
            edge_blur = self._check_edge_blur(face_roi)
            if edge_blur > 100:
                artifacts.append("Localized blurring around face edges")

        except Exception:
            pass

        return artifacts

    def _check_edge_blur(self, face_roi: np.ndarray) -> float:
        """Check for blur specifically around edges."""
        try:
            gray = cv2.cvtColor(face_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            # Create edge mask
            edge_width = w // 5
            edge_region = np.concatenate([
                gray[:, :edge_width],
                gray[:, -edge_width:],
                gray[:edge_width//2, :],
                gray[-edge_width//2:, :]
            ])

            # Calculate Laplacian variance of edge regions
            laplacian = cv2.Laplacian(edge_region, cv2.CV_64F)
            return float(laplacian.var())
        except Exception:
            return 100.0

    def _analyze_skin_texture(self, face_roi: np.ndarray) -> List[str]:
        """Analyze skin texture consistency."""
        artifacts = []

        try:
            gray = cv2.cvtColor(face_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Detect unnatural smoothing using high-frequency analysis
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = laplacian.var()

            # Check for regions with too-smooth texture (indicate deepfake)
            if texture_variance < 50:
                artifacts.append("Unnaturally smooth skin texture detected")

            # Check for inconsistent skin tone using color variance
            hsv = cv2.cvtColor(face_roi.astype(np.uint8), cv2.COLOR_RGB2HSV)
            s_channel = hsv[:, :, 1].astype(float)
            saturation_std = s_channel.std()

            if saturation_std < 10:
                artifacts.append("Low saturation variance in skin tone")

        except Exception:
            pass

        return artifacts

    def _check_lighting_consistency(self, face_roi: np.ndarray) -> List[str]:
        """Check lighting and shadow consistency."""
        issues = []

        try:
            # Analyze light distribution
            gray = cv2.cvtColor(face_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Check for unnatural lighting using histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()

            # Check for bimodal or unnatural distributions
            peaks = np.where(hist > 0.01)[0]
            if len(peaks) > 5:
                issues.append("Unnatural lighting distribution detected")

            # Check shadow consistency
            h, w = gray.shape
            left_brightness = gray[:, :w//3].mean()
            right_brightness = gray[:, 2*w//3:].mean()
            brightness_diff = abs(left_brightness - right_brightness)

            if brightness_diff > 50:
                issues.append("Inconsistent shadows and lighting across face")

        except Exception:
            pass

        return issues

    def _analyze_color_consistency(self, face_roi: np.ndarray) -> List[str]:
        """Analyze color consistency."""
        issues = []

        try:
            # Check color channels for unnaturalness
            b, g, r = cv2.split(face_roi.astype(np.uint8))

            b_mean, b_std = b.mean(), b.std()
            g_mean, g_std = g.mean(), g.std()
            r_mean, r_std = r.mean(), r.std()

            # Check for unnatural color channel ratios
            if r_mean > 0 and b_mean > 0:
                rb_ratio = r_mean / b_mean
                if rb_ratio < 0.5 or rb_ratio > 2.0:
                    issues.append("Unnatural color channel ratio detected")

            # Check for color banding
            color_var = np.array([b_std, g_std, r_std]).var()
            if color_var > 30:
                issues.append("Color channel inconsistency detected")

        except Exception:
            pass

        return issues

    def _analyze_temporal_consistency(
        self,
        frame_analyses: List[Dict],
        eye_ratios: List[float],
        head_poses: List[Dict],
        lip_distances: List[float]
    ) -> Dict[str, Any]:
        """Analyze temporal consistency across frames."""
        temporal_analysis = {
            "frame_count": len(frame_analyses),
            "temporal_consistency": 1.0,
            "issues": [],
            "details": {}
        }

        # Analyze eye blink patterns
        if eye_ratios and len(eye_ratios) > 2:
            blink_consistency = self._analyze_blink_pattern(eye_ratios)
            temporal_analysis["details"]["eye_blink_consistency"] = blink_consistency
            if blink_consistency < 0.3:
                temporal_analysis["issues"].append("Unnatural eye blink patterns")

        # Analyze head movement
        if head_poses and len(head_poses) > 2:
            movement_smoothness = self._analyze_head_movement(head_poses)
            temporal_analysis["details"]["head_movement_smoothness"] = movement_smoothness
            if movement_smoothness < 0.3:
                temporal_analysis["issues"].append("Jerky or unnatural head movements")

        # Analyze lip movement
        if lip_distances and len(lip_distances) > 2:
            lip_sync_quality = self._analyze_lip_sync(lip_distances)
            temporal_analysis["details"]["lip_sync_quality"] = lip_sync_quality
            if lip_sync_quality < 0.3:
                temporal_analysis["issues"].append("Poor lip synchronization")

        # Calculate overall temporal consistency
        consistency_scores = []
        if "eye_blink_consistency" in temporal_analysis["details"]:
            consistency_scores.append(temporal_analysis["details"]["eye_blink_consistency"])
        if "head_movement_smoothness" in temporal_analysis["details"]:
            consistency_scores.append(temporal_analysis["details"]["head_movement_smoothness"])
        if "lip_sync_quality" in temporal_analysis["details"]:
            consistency_scores.append(temporal_analysis["details"]["lip_sync_quality"])

        if consistency_scores:
            temporal_analysis["temporal_consistency"] = np.mean(consistency_scores)

        return temporal_analysis

    def _analyze_blink_pattern(self, eye_ratios: List[float]) -> float:
        """Analyze eye blink pattern consistency."""
        try:
            if len(eye_ratios) < 3:
                return 0.5

            # Eye closed when ratio < threshold
            eye_closed_threshold = 0.15
            blinks = np.array(eye_ratios) < eye_closed_threshold

            # Check for realistic blink frequency
            blink_count = np.sum(np.diff(blinks.astype(int)) != 0) // 2
            expected_blinks = max(1, len(eye_ratios) // 30)  # ~1 blink per 2 seconds (30fps)

            if blink_count == 0:
                return 0.2  # No blinking is unnatural

            blink_ratio = min(blink_count, expected_blinks) / max(blink_count, expected_blinks)
            return float(blink_ratio)
        except Exception:
            return 0.5

    def _analyze_head_movement(self, head_poses: List[Dict]) -> float:
        """Analyze smoothness of head movement."""
        try:
            if len(head_poses) < 3:
                return 0.5

            yaws = [pose.get("yaw", 0) for pose in head_poses]
            pitches = [pose.get("pitch", 0) for pose in head_poses]

            # Calculate acceleration (second derivative)
            yaw_accel = np.diff(np.diff(yaws))
            pitch_accel = np.diff(np.diff(pitches))

            # High acceleration indicates jerky movement
            total_accel = np.abs(yaw_accel).mean() + np.abs(pitch_accel).mean()
            smoothness = 1.0 / (1.0 + total_accel)

            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

    def _analyze_lip_sync(self, lip_distances: List[float]) -> float:
        """Analyze lip synchronization quality."""
        try:
            if len(lip_distances) < 3:
                return 0.5

            lip_distances = np.array(lip_distances)

            # Analyze variability and smoothness
            lip_changes = np.abs(np.diff(lip_distances))
            smoothness = 1.0 - np.mean(lip_changes) / (np.max(lip_distances) + 1e-6)

            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

    def _calculate_consistency_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall consistency score."""
        score = 1.0

        # Penalize for artifacts
        score -= len(analysis.get("facial_artifacts", [])) * 0.1

        # Penalize for issues
        score -= len(analysis.get("issues", [])) * 0.05

        # Penalize if no faces detected
        if analysis.get("face_count", 0) == 0:
            score -= 0.3

        return float(np.clip(score, 0, 1))
