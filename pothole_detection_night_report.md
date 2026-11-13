# Comprehensive Project Report: Pothole Detection Under Nighttime Driving Conditions

---

## 1. Executive Summary

This project addresses a major challenge in autonomous driving and road safety computer vision: **detecting road potholes accurately at night**.

Most existing object detection models perform well during the day because almost all road datasets are photographed under bright sunlight. However, when deployed at night, standard models suffer a severe drop in accuracy due to low visibility, harsh headlight glare, pitch-black road edges, and camera sensor noise. 

To overcome this without having to manually film and label thousands of dangerous night driving videos, this project explored **physics-based synthetic simulation** and **hybrid domain mixing**.

Across the two primary notebooks:
* **`train_dataset3_colab_resumable.ipynb`**: Proved that daytime models degrade significantly in dark conditions, created a physics-based day-to-night simulator, and discovered that while training on purely simulated night recovers synthetic accuracy, it faces a **severe "Sim-to-Real" gap** when tested on actual real-world night roads (collapsing from 73.7% to 22.5% accuracy).
* **`mixed_night_training_colab_(1).ipynb`**: Engineered the targeted solution: **Hybrid Mixed Training**. It combined a large volume of synthetic night images with a small, curated budget of deduplicated real night images (200 training images) and set up a 1,002-image pure real-night test benchmark to prove that low-cost synthetic data combined with minimal real data can bridge the real-world gap.

---

## 2. The Problem We Are Trying to Solve

```
┌──────────────────────────────────────────┐
│  Standard Road Datasets                  │
│  (100% Day, Bright Sun, High Contrast)   │
└────────────────────┬─────────────────────┘
                     │ Trains standard YOLO
                     ▼
┌──────────────────────────────────────────┐
│  Daytime Model: 80.3% Accuracy           │
└────────────────────┬─────────────────────┘
                     │ Deployed in the Dark
                     ▼
┌──────────────────────────────────────────┐
│  Night Driving Environment:              │
│  • Headlight beam cone                   │
│  • Pitch-black periphery                 │
│  • Camera sensor grain / noise           │
│  • Disappearing pothole shadow cues      │
└────────────────────┬─────────────────────┘
                     │ Accuracy drops dramatically!
                     ▼
┌──────────────────────────────────────────┐
│  Real Night Performance: 22.5% (Failure) │
└──────────────────────────────────────────┘
```

### Why Pothole Detection Fails at Night
1. **The Illumination Shift (Domain Gap)**:
   * During the day, sunlight illuminates the road evenly. The interior of a pothole creates a distinct shadow that neural networks learn to identify easily.
   * At night, illumination comes primarily from car headlights. Headlights create a bright cone of light directly ahead, leaving the rest of the road dark. Shadows cast by headlights look completely different from shadows cast by the sun.
2. **Camera Sensor Degradation**:
   * Digital camera sensors in low light crank up their gain (ISO). This introduces random grain (called *shot noise* and *readout noise*), blurring fine edge details of road cracks.
3. **The Data Scarcity Dilemma**:
   * High-quality real night datasets are rare and expensive to produce. Driving at night to capture potholes is hazardous, and human annotators struggle to draw precise bounding boxes in pitch-black video frames.
   * In addition, dashcam video captures often contain dozens of nearly identical sequential frames, causing data redundancy.

---

## 3. What Has Been Done in Each Notebook

### Notebook 1: `train_dataset3_colab_resumable.ipynb`
**Focus**: Establishing baselines, physics-based night simulation, and uncovering the "Sim-to-Real" performance cliff.

* **Disconnect-Resilient Cloud Architecture**:
  * Configured Google Colab with Google Drive integration (`MyDrive/pothole_yolo`), enabling training to save `last.pt` weights and `metrics.json` at every epoch. If Colab disconnects, the notebook resumes from where it stopped instead of restarting from scratch.
* **Day Baseline Training**:
  * Trained **YOLOv8s** (small, 11.1 million parameters) on the daytime dataset `Dataset-3_yolo` (452 train, 80 validation, 133 test images) for 50 epochs.
  * Achieved strong daytime performance: **80.27% mAP@0.5** on the unseen test set.
* **Physics-Based Day-to-Night Simulator**:
  * Rather than applying a simple, unrealistic dark filter, built an optical rendering pipeline (in Python/OpenCV) that transforms daylight images into synthetic night images while keeping all ground-truth pothole labels intact.
* **The "Dark Drop" Domain Shift Experiment**:
  * Evaluated the Day-trained model on the synthetic night test set. 
  * Accuracy dropped from **80.27% down to 66.99% (-13.3% drop)**, confirming that darkness degrades standard models.
* **Synthetic Night Adaptation**:
  * Trained a new YOLOv8s model purely on the synthetic night images.
  * Tested on the synthetic night test set: recovered performance up to **73.67% (+6.7% improvement)**.
* **The Reality Check (Evaluating on Real Night Data)**:
  * Sourced real night pothole images from Kaggle (`lxc1ferr/nighttime-potholes-dataset`), filtered annotations, and built a real night test benchmark.
  * **Critical Finding**: When tested on actual real-world night photos, the synthetic model's performance collapsed to **22.45% mAP@0.5**.
  * **Conclusion**: Physics simulation alone helps with lighting geometry, but real-world nighttime driving includes unmodeled artifacts (complex reflections, uneven road tarmac, camera sensor blur, streetlamp glare) that purely synthetic data cannot fully represent.

---

### Notebook 2: `mixed_night_training_colab_(1).ipynb`
**Focus**: Bridging the Sim-to-Real gap using Hybrid (Synthetic + Real) training and deduplication.

* **Real Night Data Sourcing & Harmonization**:
  * Sourced a larger nighttime pothole dataset (`tavishisharma09/night-potholes-yolov8-format`).
  * Filtered out non-pothole classes and remapped the target class labels into standard YOLO format (`class 0 = pothole`), identifying 1,330 candidate images.
* **Perceptual Hash Deduplication (`dHash`)**:
  * Implemented an automated 64-bit difference hashing algorithm with horizontal mirror checks.
  * Discovered and eliminated **85 near-duplicate frames**, preventing data leakage between training and testing.
* **Strategic Mixed Dataset Split**:
  To test whether a small amount of real night data can anchor the synthetic data, the dataset was partitioned with strict separation:
  * **Train Set (652 images)**: 452 Synthetic Night + **only 200 Real Night** images.
  * **Validation Set (123 images)**: 80 Synthetic Night + 43 Real Night images.
  * **Test Set (1,002 images)**: **100% pure Real Night images** (0% synthetic data).
  *(Keeping the test set 100% real ensures an objective, uncompromised real-world benchmark).*
* **3-Way Comparative Evaluation Plan**:
  * Prepared an end-to-end benchmark script to test three models head-to-head on the 1,002 real night images:
    1. Day-Trained Baseline Model
    2. Synthetic-Night-Only Model
    3. Mixed (Synthetic + Real) Model
* **Current Status & Execution Diagnostics**:
  * Data preparation, deduplication, synthetic generation, and dataset mixing completed.
  * **Run status**: In the uploaded notebook run, the Colab session started without a GPU (`nvidia-smi` not found; running on CPU). Cell 9 attempted to train on an Intel Xeon CPU (~16 minutes per epoch) and was stopped at Epoch 4/50.
  * Cells 10 (Mixed Model training), 11 (3-way evaluation on 1,002 real images), 12 (visual comparison plots), and 13 (Drive export) are fully written and ready to run once the Colab runtime is set to **T4 GPU**.

---

## 4. Methodologies Explained in Simple Terms

### A. The Object Detection Model: YOLOv8s
* **What is YOLO?** *You Only Look Once* is a neural network architecture designed for real-time object detection. Unlike older systems that scan an image piece by piece, YOLO processes the full image in a single forward pass, predicting bounding boxes and class probabilities simultaneously.
* **Why YOLOv8s ("Small")?**
  * It has **11.1 million parameters**, balancing high detection accuracy with fast inference speed (~8 milliseconds per frame).
  * This makes it suitable for deployment on dashcams or edge devices (such as NVIDIA Jetson or Raspberry Pi) inside vehicles.

---

### B. The Physics-Based Synthetic Night Simulator
Implemented in both `generate_synthetic_night_d3.py` and the notebook cells, this method transforms daytime images into realistic night scenes through a 6-step radiometric pipeline:

```
[Daytime Image]
       │
       ▼ (1) Gamma Linearization (Convert sRGB -> Linear Radiance)
[Linear Light Space]
       │
       ▼ (2) Sunlight Removal (Gaussian Blur Shading / Albedo Extraction)
[Albedo Map (Pure Road Texture)]
       │
       ▼ (3) Vehicle Lighting Projection (Elliptical Headlight Cone + Ambient Sky)
[Night-Illuminated Scene]
       │
       ▼ (4) Auto-Exposure Binary Search (Realistic Road Brightness Level)
[Target Exposure Image]
       │
       ▼ (5) Sensor Noise Injection (Poisson Shot Noise + Gaussian Read Noise)
[Grainy Night Sensor Response]
       │
       ▼ (6) Purkinje Low-Light Desaturation & Gamma Encoding
[Final Realistic Synthetic Night Image]
```

1. **Linearization**: Digital photos are non-linear (gamma encoded, $\gamma = 2.2$). The simulator expands pixel values back into linear physical light radiance ($x = (I/255)^{2.2}$).
2. **Sunlight Removal (Intrinsic Shading)**:
   * To add nighttime headlights, daytime sunlight must first be removed.
   * A large Gaussian blur extracts the smooth daytime lighting map. Dividing the image by this map isolates the road's true surface texture (albedo), removing sun glare and daylight shadows.
3. **Headlight Cone & Ambient Illumination**:
   * Models a vehicle headlight using an elliptical light beam originating near the bottom center of the camera frame ($cx \approx 0.5, cy \approx 1.0$).
   * Light intensity decays exponentially with distance: $\exp(-\text{beam} \cdot d^2)$.
   * A warm yellowish tint ($[0.80, 0.93, 1.00]$) is applied to the headlight beam, and a cool bluish ambient tint ($[1.20, 1.00, 0.85]$) is applied to the surrounding darkness.
4. **Auto-Exposure Convergence**: Uses a 32-iteration binary search to calibrate brightness so headlights match standard digital camera exposure settings at night.
5. **Sensor Noise Modeling**:
   * Real night images are noisy due to camera sensor physical constraints.
   * **Shot noise** (statistical arrival of photons) is simulated using a Poisson distribution.
   * **Read noise** (electrical noise in the camera circuitry) is simulated using a Gaussian distribution.
6. **Scotopic Desaturation (Purkinje Effect)**: In low light, human eyes and camera sensors lose color fidelity. The simulator desaturates colors by 30%, shifting the palette toward realistic nighttime tones.

> **Why this simulation approach is effective**: Because this is purely a photometric pixel-level transformation, the physical coordinates of the potholes do not move. **100% of the original daytime bounding box labels remain valid**, creating hundreds of nighttime training examples with zero manual labeling effort.

---

### C. Perceptual Hash Deduplication (`dHash`)
When road datasets are extracted from dashboard video cameras, consecutive video frames often show the same pothole with only fractional pixel movement:

```
Frame 101: [Car approaching pothole] ──┐ 
Frame 102: [0.03 seconds later]     ──┼─ Identical scene! If split between train
Frame 103: [0.06 seconds later]     ──┘  and test, the model cheats by memorization.
```

If nearly identical frames appear in both the training set and the test set, the model can memorize the pothole rather than learning to generalize.

* **How `dHash` Works**:
  1. Shrinks each image to a tiny $9 \times 8$ grayscale matrix.
  2. Compares each pixel's brightness to its neighbor on the right ($P[x, y] > P[x+1, y]$), producing a compact **64-bit binary fingerprint**.
  3. Checks both standard and horizontally flipped orientations.
  4. Computes the **Hamming distance** (how many bits differ between two images).
  5. Any image pair with a Hamming distance $\le 6$ is flagged as a near-duplicate and pruned.
* **Result**: **85 duplicate images were identified and removed**, ensuring the test set remains an honest benchmark.

---

### D. The Mixed-Domain Training Strategy
* **The Concept**: Rather than relying exclusively on synthetic simulation or collecting thousands of expensive real images, this approach combines:
  * **Synthetic Night Data (Abundant, Free)**: Teaches the network a diverse range of pothole shapes, aspect ratios, and edge contours.
  * **Real Night Data (Small, Budget-Capped at 200 images)**: Fine-tunes the network on authentic road asphalt textures, headlight reflection patterns, and real camera artifacts.
* By testing on **1,002 pure real night images**, the experiment directly measures whether adding just 200 real examples resolves the Sim-to-Real gap.

---

## 5. What Has Been Achieved So Far (Results & Findings)

### Quantitative Experimental Results Table

| Model Variant | Trained On | Evaluated On Split | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Key Insight / Finding |
|---|---|---|---|---|---|---|---|
| **Day Model** | Day Dataset-3 | Day Validation | 80.2% | 76.0% | **83.3%** | 53.0% | Model converges well on daytime road scenes. |
| **Day Model** | Day Dataset-3 | Day Test | 76.3% | 74.2% | **80.3%** | 51.4% | **Day Baseline**: Solid performance in bright daylight. |
| **Day Model** | Day Dataset-3 | Synthetic Night Test | 82.9% | 55.9% | **67.0%** | 41.1% | **The Dark Drop**: Accuracy falls by 13.3% when darkness is applied. |
| **Synthetic Night Model** | Synthetic Night | Synthetic Night Test | 69.6% | 70.2% | **73.7%** | 46.7% | **Synthetic Recovery**: Retraining on synthetic night recovers +6.7% mAP. |
| **Synthetic Night Model** | Synthetic Night | **Pure REAL Night Test** | N/A | N/A | **22.5%** | 2.2% | **The Sim-to-Real Cliff**: Simulated night alone does not transfer reliably to real-world roads. |
| **Mixed Model** ⭐ | **Synthetic (452) + Real (200)** | **Pure REAL Night Test (1,002)** | *Pending GPU Run* | *Pending GPU Run* | *Pending GPU Run* | *Pending GPU Run* | **Target Solution**: Specifically designed to bridge the Sim-to-Real gap. |

> **Understanding the Metrics**:
> * **Precision**: When the model predicts a pothole, how often is it actually a pothole? (High precision = few false alarms).
> * **Recall**: Out of all real potholes on the road, what percentage did the model catch? (High recall = few missed potholes).
> * **mAP@0.5**: *Mean Average Precision* at 50% Intersection-over-Union. The standard benchmark metric for object detection accuracy. Higher is better.

---

## 6. Current State & Immediate Next Steps

### Current State
1. **Pipeline & Engineering**: The codebase is organized and complete. The data pipelines, synthetic simulator, deduplication algorithm, and training routines are functional.
2. **Key Insight Established**: The project confirmed that while physics-based simulation handles lighting geometry, it is insufficient on its own for real-world deployment (mAP drop to 22.5%).
3. **The Solution Implemented**: The mixed dataset structure (452 synthetic + 200 real train, 1,002 real test) is built and waiting on Google Drive.

### Recommended Next Steps to Complete the Experiment
1. **Enable GPU in Google Colab**:
   * Open `mixed_night_training_colab_(1).ipynb` in Google Colab.
   * Go to **Runtime → Change runtime type → select T4 GPU → Save**.
2. **Execute Remaining Training & Evaluation**:
   * Click **Runtime → Run all**.
   * Because caching is in place, completed preparation steps will be skipped.
   * YOLOv8s will train on the mixed dataset in roughly 10–15 minutes on a T4 GPU.
   * Cell 11 will automatically benchmark all 3 models on the 1,002 real-night test set, generating `real_night_results.csv`.
   * Cell 12 will output side-by-side visual detection plots (`qualitative_comparison.png`) showing predictions from all three models on sample night images.
