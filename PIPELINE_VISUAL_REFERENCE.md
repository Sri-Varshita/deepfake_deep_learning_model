# DeepShield AI - Strict Analysis Pipeline Visual Reference

## Pipeline Architecture Diagram

```
                          INPUT MEDIA
                        (Image/Video)
                              │
                              ▼
                   ┌──────────────────────┐
                   │  STEP 1: VALIDATION  │
                   │   Input Quality Check │
                   │  ✓ Resolution       │
                   │  ✓ Blur Detection   │
                   │  ✓ Brightness       │
                   │  ✓ Noise            │
                   │  ✓ Corruption       │
                   └──────────┬───────────┘
                              │ [Quality Result]
                              ▼
                   ┌──────────────────────┐
                   │  STEP 2: PREPROCESS  │
                   │  Normalize & Resize  │
                   │  Brightness/Contrast │
                   │  Face Detection      │
                   │  128x128 Format      │
                   └──────────┬───────────┘
                              │
                   FOR VIDEOS │  FOR IMAGES
              ┌────────────────┴────────────────┐
              ▼                                  ▼
     ┌────────────────────┐      ┌──────────────────────┐
     │ Sample Frames      │      │ Facial Consistency   │
     │ (Uniform)          │      │ Single Frame:        │
     │                    │      │ • Eye blinking       │
     │                    │      │ • Lip sync           │
     │                    │      │ • Head pose          │
     │                    │      │ • Skin texture       │
     │                    │      │ • Lighting           │
     └────────┬───────────┘      └──────────┬───────────┘
              │                              │
              ▼                              ▼
     ┌────────────────────┐      ┌──────────────────────┐
     │ STEP 3: FACIAL     │      │ STEP 4: ARTIFACTS    │
     │ CONSISTENCY        │      │ DETECTION            │
     │ Temporal Analysis: │      │ • GAN artifacts      │
     │ • Blink patterns   │      │ • Pixel distortions  │
     │ • Head movement    │      │ • Frequency anomaly  │
     │ • Lip sync         │      │ • Boundary artifacts │
     │ • Smoothness       │      │ • Color misalignment │
     └────────┬───────────┘      │ • Texture issues     │
              │                  └──────────┬───────────┘
              │                             │
              │         STEP 5: TEMPORAL    │
              │         FRAME-TO-FRAME      │
              │         ANALYSIS            │
              │                             │
              └────────────┬────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  STEP 5: MODEL       │
                │  PREDICTION          │
                │  Primary Classifier  │
                │  → Base Confidence   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ STEP 6: ENSEMBLE     │
                │ VERIFICATION         │
                │ (Optional)           │
                │ • Multiple models    │
                │ • Weighted voting    │
                │ • Disagreement check │
                └──────────┬───────────┘
                           │
              ┌────────────┴───────────┐
              │                        │
              ▼                        ▼
    ┌─────────────────────┐  ┌──────────────────┐
    │ Quality Factor      │  │ Facial Const.    │
    │ (Weight: 20%)       │  │ (Weight: 20%)    │
    │                     │  │                  │
    │ Quality × 0.20      │  │ Consistency × 0.2│
    └─────────────────────┘  └──────────────────┘
              │                        │
              │         ┌──────────────┘
              │         │
              └─────────┼─────────────┐
                        │             │
                        ▼             ▼
              ┌─────────────────────────┐
              │ STEP 7: DECISION ENGINE │
              │                         │
              │ Confidence Synthesis:   │
              │ = Base × Quality        │
              │   × Facial             │
              │   × Artifact           │
              │   × Ensemble           │
              │                         │
              │ Confidence Level:       │
              │ • HIGH    (≥90%)        │
              │ • MODERATE (70-89%)    │
              │ • LOW     (50-69%)     │
              │ • UNCERTAIN (<50%)     │
              │                         │
              │ Output:                 │
              │ ✓ Final Label           │
              │ ✓ Final Confidence      │
              │ ✓ Reasoning (Multi)     │
              │ ✓ Recommendation        │
              │ ✓ Decision Factors      │
              └────────────┬────────────┘
                           │
                           ▼
                    FINAL DECISION
                    ✅ REAL or 🚨 FAKE
                    + Confidence Level
                    + Detailed Reasoning
                    + Recommendation
```

---

## Decision Flow Chart

```
PREDICTION RESULT
       │
       ├─► Confidence ≥ 90%?
       │       YES ──► HIGH CONFIDENCE
       │       NO  ┐
       │           │
       │           ├─► Confidence ≥ 70%?
       │           │       YES ──► MODERATE CONFIDENCE
       │           │       NO  ┐
       │           │           │
       │           │           ├─► Confidence ≥ 50%?
       │           │           │       YES ──► LOW CONFIDENCE
       │           │           │       NO  ──► UNCERTAIN
       │           │           │
       │           └───────────┘
       │
       ▼
CONFIDENCE LEVEL APPLIED
       │
       ├─► Quality Issues Detected?
       │    YES ──► Apply Quality Penalty
       │    NO  ──► No Penalty
       │
       ├─► Facial Inconsistencies?
       │    YES ──► Reduce Consistency Score
       │    NO  ──► Full Score
       │
       ├─► Artifacts Detected?
       │    MANY ──► Increase Artifact Score
       │    FEW  ──► Lower Artifact Score
       │
       ├─► Multi-Model Agreement?
       │    YES ──► Increase Ensemble Factor
       │    NO  ──► Decrease Ensemble Factor
       │
       ▼
GENERATE REASONING
       │
       ├─► Primary Classification Reason
       ├─► Quality Assessment
       ├─► Facial Feature Results
       ├─► Artifact Detection Results
       ├─► Confidence Level Assessment
       └─► Actionable Recommendation
       │
       ▼
FINAL DECISION WITH FULL CONTEXT
```

---

## Confidence Level Decision Tree

```
                           START
                            │
                            ▼
              Is input quality good?
                    │         │
              YES ┌─┴──┐  NO ┌─┴──────────┐
                  │    │     │            │
                  ▼    │     │      Quality Penalty
          No major     │     │      10-30% reduction
          issues       │     │            │
                       │     │            ▼
                       │     └─── Input → Moderate/Low
                       │           Quality
                       │
                       ▼
        Are facial features consistent?
              │                    │
         YES ┌─┴──┐ NO/SOME ┌──┴──────────┐
             │    │         │             │
             │    │         Inconsistency│
             │    │         Detected     │
             │    │         Suspicious   │
             │    │         Score ↓      │
             │    │                      │
             ▼    │                      │
        Natural   │                      │
        Patterns  │                      ▼
                  │            Possible Deepfake
                  │            Indicators Found
                  │
                  ▼
        Are artifacts detected?
              │              │
         LOW ┌─┴──┐ HIGH ┌───┴─────┐
             │    │      │         │
             │    │      Artifact  │
             │    │      Score ↑   │
             │    │      Confidence│
             │    │      → FAKE    │
             ▼    │                ▼
       Authentic  │           Likely DEEPFAKE
       Indicators │
             +────┘
             │
             ▼
   Do models agree (ensemble)?
              │           │
         YES ┌─┴──┐  NO  ┌─┴──┐
             │    │      │    │
        High │    │      Low  │
        Agree│    │      Agree│
             │    │      │    │
             ▼    │      │    ▼
        Increase  │      Uncertainty
        Confidence│      + Review
             │    │      Recommended
             │    │            │
             └────┼────────────┘
                  │
                  ▼
          CALCULATE FINAL
          CONFIDENCE %
                  │
        ┌─────────┼──────────┐
        │         │          │
    ≥90% │     70-89% │   <50% │
        │         │          │
        ▼         ▼          ▼
      HIGH   MODERATE    UNCERTAIN
    CONFIDENCE CONFIDENCE  RESULT
        │         │          │
        ├─► ✅    ├─► ⚠️     └─► ❌
        │   REAL  │   VERIFY   EXPERT
        │   or    │   NEEDED   REVIEW
        │   🚨    └─► 🚨
        │   FAKE     DEEPFAKE?
        │
        ▼
    FINAL DECISION
    WITH RECOMMENDATION
```

---

## Data Flow: Analysis Modules

```
INPUT IMAGE/VIDEO
        │
        ├─────────────────────┬──────────────────┬──────────────┐
        │                     │                  │              │
        ▼                     ▼                  ▼              ▼
   ┌─────────┐      ┌─────────────┐    ┌──────────────┐  ┌─────────┐
   │InputVali│      │Facial Consis│    │ Artifact     │  │Ensemble │
   │dator    │      │ tency       │    │ Detector     │  │Detector │
   │         │      │ Analyzer    │    │              │  │         │
   └────┬────┘      └─────┬───────┘    └──────┬───────┘  └────┬────┘
        │                 │                   │               │
        ├─► Quality      ├─► Facial       ├─► Artifacts  ├─► Model
        │   Level        │   Score        │   Found       │   Agreement
        │                │   + Issues     │   + Score     │   Level
        │                │                │               │
        │                └─────────────────────────────────┼────┬───┐
        │                                                   │    │   │
        └───────────────────────────────────────┬──────────┴────┴─┬─┘
                                                │                  │
                                                ▼                  ▼
                                          ┌──────────────────────────┐
                                          │  DECISION ENGINE         │
                                          │                          │
                                          │ Synthesize:              │
                                          │ • Quality (20%)          │
                                          │ • Facial (20%)           │
                                          │ • Artifacts (20%)        │
                                          │ • Model (40%)            │
                                          │ • Ensemble Agr.          │
                                          │                          │
                                          └────────┬─────────────────┘
                                                   │
                                    ┌──────────────┴───────────────┐
                                    │                              │
                                    ▼                              ▼
                            ┌──────────────┐          ┌──────────────────┐
                            │Final Label   │          │Confidence Level  │
                            │(Real/Fake)   │          │(High/Mod/Low/Unc)│
                            │              │          │                  │
                            │Final Confid. │          │Reasoning         │
                            │(0-100%)      │          │(Multi-line)      │
                            │              │          │                  │
                            │Decision      │          │Recommendation    │
                            │Factors (%)   │          │(Actionable)      │
                            └──────────────┘          └──────────────────┘
                                    │                              │
                                    └──────────┬───────────────────┘
                                               │
                                               ▼
                                    FINAL COMPREHENSIVE RESULT
```

---

## Confidence Adjustment Factors

```
BASE CONFIDENCE (from model)
        │
        ├─► ×  QUALITY FACTOR (0.0 - 1.0)
        │        - Excellent: 1.0x
        │        - Good:      0.95x
        │        - Fair:      0.80x
        │        - Poor:      0.50x
        │
        ├─► ×  FACIAL CONSISTENCY FACTOR (0.8 - 1.0)
        │        - High (>0.7):  1.0x
        │        - Medium:       0.9x
        │        - Low (<0.3):   0.8x
        │
        ├─► ×  ARTIFACT FACTOR (1.0 - 1.3)
        │        - None:         1.0x
        │        - Some:         1.1x
        │        - Many:         1.3x
        │
        ├─► ×  ENSEMBLE AGREEMENT FACTOR (0.8 - 1.0)
        │        - Full agreement:    1.0x
        │        - Moderate agree:    0.9x
        │        - Disagreement:      0.8x
        │
        ▼
    FINAL CONFIDENCE
    (Clipped to 0-100%)
        │
        ▼
    CONFIDENCE LEVEL ASSIGNMENT
    ├─► ≥90%  → HIGH
    ├─► 70-89% → MODERATE
    ├─► 50-69% → LOW
    └─► <50%  → UNCERTAIN
```

---

## Quality Penalty Calculation

```
QUALITY CHECKS
        │
        ├─► Resolution < 100px?
        │   YES: +25% penalty, "Low resolution"
        │   NO:  No penalty
        │
        ├─► Blur score < 100?
        │   YES: +20% penalty, "Image is blurry"
        │   NO:  No penalty
        │
        ├─► Brightness < 30 or > 220?
        │   YES: +20% penalty, "Too dark/bright"
        │   NO:  No penalty
        │
        ├─► Noise level > 0.15?
        │   YES: +15% penalty, "High noise"
        │   NO:  No penalty
        │
        ├─► Corruption detected?
        │   YES: +30% penalty, "Image corrupted"
        │   NO:  No penalty
        │
        ▼
    TOTAL CONFIDENCE PENALTY
    (Sum of individual penalties, max 100%)
        │
        ├─► 0% penalty:    Quality Level = "High"
        ├─► 1-15% penalty: Quality Level = "Good"
        ├─► 16-30% penalty: Quality Level = "Fair"
        └─► >30% penalty:  Quality Level = "Low"
```

---

## Artifact Detection Pipeline

```
IMAGE ANALYSIS
        │
        ├─► FREQUENCY ANALYSIS
        │   ├─► FFT Magnitude Spectrum
        │   │   └─► Detect Ring Artifacts
        │   │       (GAN fingerprint)
        │   │
        │   └─► DCT Analysis
        │       └─► High-frequency anomalies
        │
        ├─► PIXEL-LEVEL ANALYSIS
        │   ├─► Quantization artifacts
        │   ├─► Edge blocking patterns
        │   └─► Checkerboard detection
        │
        ├─► BOUNDARY ANALYSIS
        │   ├─► Edge concentration
        │   └─► Border blur patterns
        │
        ├─► COLOR ANALYSIS
        │   ├─► Channel misalignment
        │   └─► Unnatural ratios
        │
        └─► TEXTURE ANALYSIS
            ├─► Local Binary Patterns
            └─► Smoothness detection
        │
        ▼
    ARTIFACT SCORES AGGREGATION
    (Average of all detections)
        │
        ├─► 0-0.3:  Few/No artifacts → Likely Authentic
        ├─► 0.3-0.6: Some artifacts → Moderate suspicion
        └─► 0.6-1.0: Many artifacts → Likely Deepfake
```

---

## Video Temporal Analysis

```
VIDEO FRAMES
        │
        ├─► Frame 1
        ├─► Frame 2
        ├─► Frame 3
        └─► Frame 4 ... N (sampled uniformly)
        │
        ▼
    FACIAL CONSISTENCY ACROSS FRAMES
        │
        ├─► Eye Blink Pattern
        │   • Frequency check (~1 per 2 sec)
        │   • Natural transitions
        │   └─► Blink Consistency Score
        │
        ├─► Head Movement
        │   • Yaw/Pitch continuity
        │   • Acceleration analysis
        │   └─► Movement Smoothness Score
        │
        └─► Lip Movement
            • Opening/closing patterns
            • Sync with blinks
            └─► Lip Sync Quality Score
        │
        ▼
    FRAME-TO-FRAME ANOMALIES
        │
        ├─► Motion Magnitude Check
        │   └─► Detect unnatural jumps
        │
        ├─► Temporal Flicker
        │   └─► Detect rapid intensity changes
        │
        └─► Consistency Verification
            └─► Check for sudden changes
        │
        ▼
    TEMPORAL CONSISTENCY SCORE
    (Average of all metrics)
        │
        ├─► 0-0.3:  Unnatural temporal patterns
        ├─► 0.3-0.7: Mixed signals
        └─► 0.7-1.0: Natural temporal progression
```

---

## Legend

```
✅  Feature Complete
🚨  Deepfake Indicator
⚠️   Warning/Review Needed
❌  Uncertain/Requires Expert
🟢  Good/High Confidence
🟡  Moderate/Caution
🔴  Bad/Uncertain
───  Data Flow
```
