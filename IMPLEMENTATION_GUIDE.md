# Implementation Guide: Strict Analysis Pipeline Integration

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**New dependencies added:**
- `scipy>=1.10.0` - Signal processing and statistics
- `mediapipe>=0.10.0` - Facial landmark detection

### 2. Update Streamlit App

Replace the import in `src/app.py`:

```python
# OLD (Simple inference)
from model import load_model, predict_image, predict_video

# NEW (Advanced pipeline)
from model import load_model
from model.enhanced_inference import analyze_image_complete, analyze_video_complete
```

### 3. Update Analysis Functions

In `src/app.py`, replace the prediction calls:

#### Before (Simple):
```python
def _render_image_tab(model, preprocess_method: str, show_debug: bool):
    # ... upload code ...
    if st.button("🔍 Analyze Image"):
        result = predict_image(model, image, preprocess_method)
        _render_prediction(result, "Image", show_debug=show_debug)
```

#### After (Advanced Pipeline):
```python
def _render_image_tab(model, preprocess_method: str, show_debug: bool):
    # ... upload code ...
    if st.button("🔍 Analyze Image"):
        with st.spinner("🔄 Running complete analysis pipeline..."):
            result = analyze_image_complete(model, image, preprocess_method)
            _render_advanced_prediction(result, "Image", show_debug=show_debug)
```

### 4. Implement Advanced Result Rendering

Add new rendering function in `src/app.py`:

```python
def _render_advanced_prediction(result: Dict[str, Any], source_name: str, show_debug: bool = False):
    """Render comprehensive analysis result."""
    
    if "error" in result:
        st.error(f"❌ Analysis error: {result['error']}")
        return
    
    final_pred = result.get("final_prediction", {})
    final_label = final_pred.get("final_label", "Unknown")
    confidence = final_pred.get("final_confidence", 0.0)
    confidence_level = final_pred.get("confidence_level", "Unknown")
    
    # Display main result
    if final_label == "Fake":
        st.error(f"### 🚨 DETECTION RESULT: DEEPFAKE - {source_name}")
    elif final_label == "Real":
        st.success(f"### ✅ DETECTION RESULT: AUTHENTIC - {source_name}")
    else:
        st.warning(f"### ⚠️ DETECTION RESULT: UNCERTAIN - {source_name}")
    
    # Display confidence
    st.markdown(f"**Confidence Level:** {confidence_level} ({confidence:.1f}%)")
    
    # Display confidence bar
    if final_label == "Fake":
        st.progress(confidence / 100)
    else:
        st.progress(confidence / 100)
    
    # Display reasoning
    st.markdown("**Analysis Reasoning:**")
    reasoning = final_pred.get("reasoning", [])
    for reason in reasoning:
        st.write(f"• {reason}")
    
    # Display recommendation
    recommendation = final_pred.get("recommendation", "")
    if recommendation:
        st.info(recommendation)
    
    # Detailed analysis breakdown
    if show_debug:
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis Breakdown")
        
        # Input quality
        quality = result.get("input_quality", {})
        st.markdown("#### Input Quality:")
        st.write(f"- Quality Level: {quality.get('quality_level', 'N/A')}")
        st.write(f"- Issues: {len(quality.get('issues', []))} detected")
        if quality.get("issues"):
            for issue in quality.get("issues", []):
                st.write(f"  - {issue}")
        
        # Facial analysis
        facial = result.get("facial_analysis", {})
        st.markdown("#### Facial Consistency:")
        st.write(f"- Consistency Score: {facial.get('consistency_score', 0):.2f}")
        st.write(f"- Faces Detected: {facial.get('face_count', 0)}")
        if facial.get("facial_artifacts"):
            st.write("- Artifacts:")
            for artifact in facial.get("facial_artifacts", []):
                st.write(f"  - {artifact}")
        
        # Artifact detection
        artifacts = result.get("artifact_analysis", {})
        st.markdown("#### Artifact Detection:")
        st.write(f"- Artifact Score: {artifacts.get('artifact_score', 0):.2f}")
        st.write(f"- Artifacts Found: {len(artifacts.get('artifacts', []))}")
        if artifacts.get("artifacts"):
            st.write("- Detected Artifacts:")
            for artifact in artifacts.get("artifacts", [])[:5]:
                st.write(f"  - {artifact}")
        
        # Decision factors
        factors = final_pred.get("decision_factors", {})
        st.markdown("#### Decision Factors:")
        st.write(f"- Quality Factor: {factors.get('quality_factor', 0):.2f}")
        st.write(f"- Facial Consistency: {factors.get('facial_consistency_factor', 0):.2f}")
        st.write(f"- Artifact Factor: {factors.get('artifact_factor', 0):.2f}")
        st.write(f"- Ensemble Agreement: {factors.get('ensemble_agreement_factor', 0):.2f}")
        
        # Pipeline info
        st.markdown(f"- Analysis Pipeline: {result.get('analysis_pipeline', 'N/A')}")
```

### 5. Update UI Components

Update the header to show pipeline status:

```python
st.markdown("""
    ### 🔒 DeepShield AI - Strict Analysis Pipeline
    **High-Accuracy Deepfake Detection System**
    
    This system uses a **7-step analysis pipeline** for maximum reliability:
    1. ✅ Input Validation
    2. 📐 Preprocessing & Face Detection
    3. 👤 Facial Consistency Analysis
    4. 🔍 Artifact Detection
    5. ⏱️ Temporal Analysis (Videos)
    6. 🤖 Multi-Model Verification
    7. ⚖️ Advanced Decision Engine
""")
```

---

## Module Structure

### New Modules in `src/analysis/`:

```
src/analysis/
├── __init__.py                 # Package exports
├── input_validator.py          # Quality validation
├── facial_consistency.py       # Facial feature analysis
├── artifact_detector.py        # Artifact detection
├── ensemble_detector.py        # Multi-model voting
└── decision_engine.py          # Final decision making
```

### Enhanced Inference in `src/model/`:

```
src/model/
├── enhanced_inference.py       # New: Complete pipeline
├── inference.py                # Existing: Basic predictions
├── architecture.py             # Model architecture
└── loader.py                   # Model loading
```

---

## Configuration Options

### Adjust Decision Thresholds

In `src/analysis/decision_engine.py`:

```python
class DecisionEngine:
    # Change confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 90.0      # Default: 90%
    MODERATE_CONFIDENCE_THRESHOLD = 70.0  # Default: 70%
    
    # Change weight factors
    MODEL_PREDICTION_WEIGHT = 0.40        # Default: 40%
    INPUT_QUALITY_WEIGHT = 0.20           # Default: 20%
    FACIAL_CONSISTENCY_WEIGHT = 0.20      # Default: 20%
    ARTIFACT_DETECTION_WEIGHT = 0.20      # Default: 20%
```

### Adjust Input Quality Thresholds

In `src/analysis/input_validator.py`:

```python
class InputValidator:
    MIN_RESOLUTION = 100            # Minimum resolution
    BLUR_THRESHOLD = 100.0          # Blur detection threshold
    DARKNESS_THRESHOLD = 30         # Too dark threshold
    BRIGHTNESS_THRESHOLD = 220      # Too bright threshold
    NOISE_THRESHOLD = 0.15          # Noise detection threshold
```

### Adjust Face Analysis Thresholds

In `src/analysis/artifact_detector.py`:

```python
# Modify detection thresholds for different artifacts
# GAN artifacts, checkerboard, frequency anomalies, etc.
```

---

## Performance Considerations

### Memory Usage
- Video analysis: ~500MB per video (12 sampled frames)
- Image analysis: ~100MB per image
- Ensemble with 3 models: ~150% base model memory

### Processing Time (Approximate)
- Image analysis: 2-5 seconds
- Video analysis (12 frames): 5-15 seconds
- Ensemble (3 models): +1-2 seconds overhead

### Optimization Tips

1. **Reduce Frame Sampling**
```python
result = analyze_video_complete(model, video, max_frames=6)  # Faster, less accurate
```

2. **Disable Debug Output**
```python
show_debug = False  # Skips detailed rendering
```

3. **Cache Analysis Pipeline**
```python
# Already cached with @st.cache_resource in enhanced_inference.py
```

---

## Testing

### Unit Tests

Create `tests/test_analysis_pipeline.py`:

```python
import numpy as np
from PIL import Image
from src.analysis import InputValidator, ArtifactDetector, DecisionEngine

def test_input_validator():
    validator = InputValidator()
    # Create test image
    img = Image.new('RGB', (200, 200), color='white')
    result = validator.validate_image(img)
    assert result["is_valid"] == True
    assert result["quality_level"] in ["High", "Good", "Fair", "Low"]

def test_artifact_detector():
    detector = ArtifactDetector()
    img_array = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.detect_artifacts(img_array)
    assert "artifact_score" in result
    assert 0 <= result["artifact_score"] <= 1

def test_decision_engine():
    engine = DecisionEngine()
    model_pred = {"label": "Fake", "confidence": 85, "fake_confidence": 85, "real_confidence": 15}
    quality = {"confidence_penalty": 0, "quality_level": "High", "issues": []}
    facial = {"consistency_score": 0.8, "issues": [], "facial_artifacts": []}
    artifact = {"artifact_score": 0.3, "has_artifacts": False, "artifacts": []}
    
    decision = engine.make_decision(model_pred, quality, facial, artifact)
    assert decision["final_label"] in ["Real", "Fake"]
    assert decision["confidence_level"] in ["High", "Moderate", "Low", "Uncertain"]
```

### Integration Tests

Test with sample images and videos:

```bash
# Create test images with different characteristics
python tests/generate_test_images.py

# Run analysis pipeline
python tests/test_integration.py
```

---

## Troubleshooting

### ImportError: No module named 'mediapipe'

**Solution:**
```bash
pip install mediapipe
# If that fails, try:
pip install --upgrade mediapipe
```

### CUDA Out of Memory

**Solution:**
```python
# Use CPU-only analysis
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Slow Analysis

**Solution:**
1. Reduce max_frames: `max_frames=6` instead of 12
2. Use lower resolution input
3. Disable debug output
4. Run on GPU machine

### Missing Face Detection

**Solution:**
1. Check if face is clearly visible
2. Ensure minimum resolution (100x100)
3. Check lighting and contrast
4. Face detection fallback uses Haar Cascade if MediaPipe unavailable

---

## Integration Checklist

- [ ] Install new dependencies (scipy, mediapipe)
- [ ] Create `src/analysis/` directory
- [ ] Add all analysis modules to `src/analysis/`
- [ ] Add `enhanced_inference.py` to `src/model/`
- [ ] Update `src/app.py` imports
- [ ] Add `_render_advanced_prediction()` function to UI
- [ ] Update UI header with pipeline description
- [ ] Test image analysis with sample image
- [ ] Test video analysis with sample video
- [ ] Verify all confidence levels and reasoning
- [ ] Test debug output display
- [ ] Performance testing on target machine
- [ ] Document any customizations made

---

## Next Steps

1. **Deploy Updated App**
   - Test in development environment
   - Deploy to Hugging Face Spaces (if applicable)
   - Monitor performance metrics

2. **Collect Feedback**
   - Gather user feedback on confidence levels
   - Adjust thresholds based on real-world usage
   - Track prediction accuracy

3. **Improve Models**
   - Add more detector models to ensemble
   - Fine-tune weights based on performance
   - Implement online learning if applicable

4. **Extend Analysis**
   - Add more artifact detectors
   - Implement additional facial features
   - Add speech-sync analysis for videos
   - Implement behavioral pattern analysis

5. **Optimize Performance**
   - GPU acceleration for batch processing
   - Async processing for better UX
   - Caching of frequently analyzed content
