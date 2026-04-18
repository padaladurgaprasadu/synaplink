# 🧠 SynapLink — Brain-to-Code Interface

<div align="center">

**Decode EEG brain signals → Classify cognitive intent → Generate executable code**

*No keyboard. No voice. Just your brain.*

</div>

---

## 📖 What is SynapLink?

**SynapLink** is a non-invasive Brain-Computer Interface (BCI) system that:

1. 🧠 **Captures** EEG brain signals through 8 scalp electrodes
2. 🤖 **Classifies** cognitive intent across **42 programming and content tasks**
3. ⚡ **Generates** executable Python code, text, or image prompts via **Google Gemini AI**
4. 🖥️ **Displays** the output on screen in **1.21 seconds** — end to end

> Built as an independent research project by **Padala Durga Prasadu** (B.Tech CSE AI/ML, KARE 2025)  
> Under the guidance of **Dr. P. Deepalakshmi**, Dean, School of Computing, KARE  
> Research paper submitted to **IEEE ICIRCA-2025**

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| L1 Domain Accuracy (CODE/TEXT/IMAGE) | **99.99%** |
| L2 Category Accuracy (14 classes) | **73.54%** |
| L3 Intent Accuracy (42 classes) | **61.06%** |
| End-to-End Latency | **1.21 seconds** |
| Training Dataset | **37,800 EEG samples** |
| Cross-Session Feature Stability | **60.7× better** than raw features |
| Headset Build Cost | **₹11,750** |

---

## 🔬 Scientific Novelty

### Session-Invariant Ratio Features

The biggest problem in EEG-based BCIs is **signal variability across sessions** — the same thought produces different signals on different days due to fatigue, electrode placement, and arousal.

SynapLink introduces **5 novel band-power ratio features** that algebraically cancel session drift:

```
R₁ = θ/α    R₂ = β/γ    R₃ = θ/β    R₄ = θ phase coupling    R₅ = HAI
```

**When session drift multiplies all bands by k:**
```
(β·k)/(α·k) = β/α  ✅  k cancels! Session drift eliminated algebraically.
```

| Feature Type | Cross-Session CV | Stability |
|-------------|-----------------|-----------|
| Raw β power | 18.2% | Baseline |
| Ratio features (mean) | **0.3%** | **60.7× more stable** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HEAD UNIT (0W heat)                     │
│   F3  F4  Fz  C3  C4  T7  T8  Pz  ← Ag/AgCl dry electrodes│
└──────────────────┬──────────────────────────────────────────┘
                   │ shielded cable
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              ELECTRONICS BOX (belt-worn)                    │
│   ADS1299EEG-FE │ ESP32-WROOM-32E │ LiPo 500mAh │ DRL      │
└──────────────────┬──────────────────────────────────────────┘
                   │ BLE 5.0
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAPTOP / HOST PC                         │
│                                                             │
│  Raw EEG → Butterworth → Notch → FastICA → Hamming Epoch   │
│                                                             │
│  31 Features = 17 spectral + 3 Hjorth + 6 PLV + 5 ratios  │
│                                                             │
│  L1 RF (99.99%) → L2 RF (73.54%) → L3 RF (61.06%)         │
│                                                             │
│  Intent → Gemini 2.5 Flash → Code/Text/Image Output        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install flask google-genai scikit-learn pandas numpy scipy
```

### Run Demo

```bash
git clone https://github.com/padaladurgaprasadu/SynapLink-BCI.git
cd SynapLink-BCI

# Add your Gemini API key in synaplink_pipeline_v2.py (line 7)
# Get free key at: aistudio.google.com

python synaplink_pipeline_v2.py
```

Open browser: **http://localhost:5000**

Click **🧠 RUN BRAIN-TO-CODE PIPELINE** → Watch it classify brain intent and generate code!

### Train Models (Optional)

```bash
# Place synaplink_real_eeg_dataset.csv in project folder
python train_models.py
# Generates: synaplink_models/m1.pkl, m2.pkl, m3.pkl, sc.pkl, le1.pkl, le2.pkl, le3.pkl
```

---

## 📁 Project Structure

```
SynapLink-BCI/
│
├── synaplink_pipeline_v2.py      # Main Flask app — no buttons, random signal
├── synaplink_pipeline.py         # Flask app with intent selector
├── synaplink_realtime.py         # Auto-predict every 3 seconds
├── train_models.py               # Train hierarchical RF models
├── check_models.py               # List available Gemini models
│
├── synaplink_models/             # Trained model files
│   ├── m1.pkl                    # L1 Random Forest (CODE/TEXT/IMAGE)
│   ├── m2.pkl                    # L2 Random Forest (14 categories)
│   ├── m3.pkl                    # L3 Random Forest (42 intents)
│   ├── sc.pkl                    # StandardScaler
│   ├── le1.pkl                   # L1 Label Encoder
│   ├── le2.pkl                   # L2 Label Encoder
│   └── le3.pkl                   # L3 Label Encoder
│
├── synaplink_demo.html           # 5-page professional demo app
├── synaplink_headset_v2.html     # Headset v2.0 hardware design
├── synaplink_signal_explainer.html  # Signal variability explainer
│
└── README.md
```

---

## 🧬 Intent Taxonomy — 42 Classes

```
L1 — Domain (3 classes)
├── CODE
│   ├── L2: ALGORITHM    → bubble_sort, fibonacci, binary_search
│   ├── L2: LOOP         → for_range, while_loop, nested_loop
│   ├── L2: CONDITIONAL  → if_else, switch_case, ternary
│   ├── L2: FUNCTION_DEF → simple_function, recursive_fn
│   ├── L2: CLASS_DEF    → simple_class, inheritance
│   └── L2: PRINT_STMT   → hello_world, variable_print
├── TEXT
│   ├── L2: COMPOSE      → email_draft, formal_letter
│   ├── L2: SUMMARIZE    → summarize_doc, key_points
│   └── L2: TRANSLATE    → en_to_te, en_to_hi
└── IMAGE
    └── L2: GENERATE     → ui_mockup, data_viz_prompt
```

---

## 💻 Tech Stack

| Component | Technology |
|-----------|------------|
| EEG Hardware | ADS1299EEG-FE + ESP32-WROOM-32E |
| Wireless | BLE 5.0 |
| Signal Processing | SciPy, numpy, FastICA |
| Machine Learning | scikit-learn Random Forest |
| LLM API | Google Gemini 2.5 Flash |
| Backend | Python Flask |
| Frontend | Vanilla JavaScript + Canvas API |
| Dataset | 37,800 samples × 43 columns |

---

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo web interface |
| `/random_signal` | GET | Sample random EEG from dataset |
| `/predict_from_signal` | POST | Classify intent from signal array |
| `/generate` | POST | Generate code via Gemini API |

---

## 📊 Classification Performance

```
L1 — Domain:    ████████████████████ 99.99%  (3 classes)
L2 — Category:  ███████████████      73.54%  (14 classes)
L3 — Intent:    ████████████         61.06%  (42 classes)

vs Random:      L1: 33% → 99.99% (3.0×)
                L3:  2.4% → 61.06% (25.6×)
```

> ⚠️ **Note:** All accuracy values measured on augmented synthetic dataset.
> Real hardware EEG validation planned at IIT Palakkad BMI Laboratory.

---

## 🔭 Roadmap

- [x] Hierarchical RF classifier (L1/L2/L3)
- [x] Session-invariant ratio features
- [x] Gemini 2.5 Flash API integration
- [x] Flask web demo with live classification
- [x] Custom headset v2.0 design (₹11,750 BOM)
- [ ] IIT Palakkad Emotiv EPOC+ hardware trials
- [ ] Physical headset construction
- [ ] CNN-LSTM upgrade (target: 78-85% L3)
- [ ] VS Code WebSocket extension
- [ ] DALL-E image generation for IMAGE domain
- [ ] EEG Foundation Model (zero-shot calibration)
- [ ] Clinical trials with ALS patients

---

## 📄 Research Paper

**Title:** SynapLink: A Brain-to-Code Interface Using EEG-Based Intent Decoding and Large Language Model-Driven Code Generation

**Authors:** Padala Durga Prasadu, Dr. P. Deepalakshmi

**Venue:** IEEE ICIRCA-2025 *(Under Preparation)*

**Key Contributions:**
1. First end-to-end pipeline from raw EEG to LLM code generation
2. 5 novel session-invariant ratio features (60.7× stability improvement)
3. Hierarchical 42-class intent taxonomy for BCI
4. Sub-1.5s brain-to-code latency with Gemini API

---

## 🔧 Troubleshooting

```bash
# Gemini 503 Error — server busy
# Wait 2 minutes and try again
# OR: check model fallback in synaplink_pipeline_v2.py

# Models not found
# Run train_models.py with dataset in project folder

# Port 5000 already in use
python synaplink_pipeline_v2.py  # auto-retries on 5001
```

---

## 👤 Author

**Padala Durga Prasadu**

- 🎓 B.Tech CSE (AI/ML) — KARE, Tamil Nadu (2025)
- 🧠 Independent BCI Researcher
- 📧 padalaprasad94@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/padaladurgaprasadu)
- 🌐 [Portfolio](https://padaladurgaprasadu.github.io)

**Research Supervisor:** Dr. P. Deepalakshmi, Dean, School of Computing, KARE

---

## 📜 License

MIT License — Free to use for research and educational purposes.

---

<div align="center">

**SynapLink — Think it. Code it.**

*🧠 → ⚡ → 💻*

</div>
