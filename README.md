<p align="center">
  <img src="Assets/alpha build banner.png" alt="EchoHands Banner" width="100%">
</p>

# EchoHands — Alpha Build

EchoHands is a modular AI-powered framework for real-time sign language recognition using a webcam. It provides a structured pipeline for detecting hand gestures, processing hand landmarks, recognizing signs, and converting them into digital text
The Alpha Build currently focuses on American Sign Language (ASL) and serves as a working foundation for further development, experimentation, and adaptation to other sign languages and recognition systems

---

## ⚙️ Technology & Working

- Webcam Capture: Capture real-time video using ***OpenCV***
- Hand Detection: Detect hands and extract 21 landmarks using ***MediaPipe***
- Feature Processing: Process landmark data into model-ready features using ***NumPy***
- Static Recognition: Recognize static signs using a scikit-learn Random Forest model
- Dynamic Recognition: Analyze multi-frame hand movements using a ***TensorFlow/Keras*** LSTM model
- Recognition Control: Check and filter model predictions before accepting them, ensuring a gesture is recognized only when it is reliable and preventing the same held gesture from being added repeatedly.
- Text Output: Build recognized signs into digital text through the application's text-building module

**Working sequence:**

OpenCV → MediaPipe → NumPy → Recognition Models → EchoHands Recognition Modules → Digital Text


---

# Alpha Build Purpose

The Alpha is basically the **working foundation of EchoHands**. It contains the real-time recognition system along with the components used for development, testing, debugging, retraining, and experimenting with the recognition pipeline.

So, this isn't meant to be the final polished EchoHands experience just yet. It is the current technical build that we're using to develop and improve the system.

> [!NOTE]
>
> The repository may contain trained **model files, datasets, source code, and other project assets** directly inside the project
   Sharing the repository publicly would make it possible for others to grab the models and a large part of the project itself — so let's not make copying the whole thing quite that easy 😅
>
> If access to the repository is private or controlled, contributors should not redistribute the source code, datasets, or model artifacts without permission from the project owner.
---
The Alpha Build is designed as a flexible recognition foundation rather than a final language-specific system.
---

### Expanding to Other Sign Languages

The current recognition pipeline can be adapted to other sign languages by collecting appropriate data, preparing language-specific datasets, and training suitable recognition models.

One important future direction is **Indian Sign Language (ISL)**.

The goal is not to assume that an ASL-trained model can directly recognize ISL. Instead, the existing EchoHands pipeline can serve as the technical foundation for building and training a separate recognition system using ISL-specific signs, datasets, labels, and models.

Conceptually:

```text
EchoHands Recognition Pipeline
            │
            ├── ASL Dataset + Models
            │        ↓
            │     ASL Recognition
            │
            └── ISL Dataset + Models
                     ↓
                  ISL Recognition
```

This makes EchoHands suitable for future expansion into a broader, modular sign-language recognition platform.

The broader direction and future plans for the project are covered in the **[Future Development](#future-development)** section below.

For the public-facing side of the project, you can also check out the main **[EchoHands repository](https://github.com/shivanshu43/Echohands)**.


# Supported Signs

EchoHands currently works with the trained signs included in the project.

The system handles two main categories of gestures:

## Static Gestures

Static gestures are recognized primarily from the configuration or pose of the hand in a captured frame.

The current Alpha Build supports the trained static ASL alphabet and numeric signs available in the included model.

## Dynamic Gestures

Some ASL signs involve motion and cannot be reliably recognized from a single frame.

EchoHands currently includes dedicated sequence-based recognition for:

- **J**
- **Z**

These gestures are recognized by analyzing hand movement across multiple frames.

<p align="center">
  <img src="sign description/sign letters.png" alt="Supported ASL Signs" width="750">
</p>

<p align="center">
  <em>Figure 1. ASL alphabet and numeric signs supported by the EchoHands recognition system.</em>
</p>

---

# Installation & Requirements

### Recommended Python Version

The Alpha Build is recommended to run with:

```text
Python 3.10
```

Using Python 3.10 helps maintain compatibility with the dependencies used by the Alpha Build.

---

## 1. Clone the Repository

Open **Command Prompt** or **PowerShell** and run:

```bash
git clone https://github.com/shivanshu43/EchoHands_Alpha-build.git
cd EchoHands_Alpha-build
```

> **Note:** The repository directory created by Git is `EchoHands_Alpha-build`. Use the directory name created on your system if it differs.

**Expected result:**

The repository is downloaded and the terminal moves into the Alpha Build project directory.

Your prompt should look similar to:

```text
E:\EchoHands_Alpha-build>
```

---

## 2. Create a Virtual Environment

The Alpha Build is recommended to use a Python 3.10 virtual environment.

### Windows

Run:

```bash
py -3.10 -m venv venv
```

**Expected result:**

The command normally produces no output and creates a new `venv` folder inside the project directory.

The project should now contain:

```text
EchoHands_Alpha-build/
└── venv/
```

---

## 3. Activate the Virtual Environment

### Windows Command Prompt

Run:

```bash
venv\Scripts\activate
```

**Expected result:**

`(venv)` appears at the beginning of the command prompt:

```text
(venv) E:\EchoHands_Alpha-build>
```

---

## 4. Verify Python

Confirm that the virtual environment is using the expected Python interpreter:

```bash
where python
```

The first path should point to the project's virtual environment, similar to:

```text
E:\EchoHands_Alpha-build\venv\Scripts\python.exe
```

Then verify the Python version:

```bash
python --version
```

**Expected result:**

```text
Python 3.10.x
```

---

### Virtual Environment Troubleshooting

If you face an error related to the virtual environment, use the following recovery procedure.

#### Step 1 — Deactivate the Current Environment

Run:

```bash
deactivate
```

You should go from:

```text
(venv) E:\EchoHands_Alpha-build>
```

to:

```text
E:\EchoHands_Alpha-build>
```

---

#### Step 2 — Delete the Incorrect `venv`

Run:

```bash
rmdir /s /q venv
```

This removes the incorrectly created virtual environment.

---

#### Step 3 — Check Whether Python 3.10 Is Installed

Run:

```bash
py -0p
```

You should see something similar to:

```text
Installed Pythons found by py Launcher for Windows
 -3.13-64 ...
 -3.10-64 C:\...\Python310\python.exe
```

**We specifically need to see a 3.10 entry.**

You can also directly test:

```bash
py -3.10 --version
```

The expected result is:

```text
Python 3.10.x
```

---

### If `py -3.10 --version` Succeeds

Perfect. Now create the environment **only with this command**:

```bash
py -3.10 -m venv venv
```

> **Important:** Do not run `python -m venv venv` after this command. Doing so may recreate the environment using another installed Python version, such as Python 3.13.

Then activate it:

```bash
venv\Scripts\activate
```

Now verify:

```bash
python --version
```

You should get:

```text
Python 3.10.x
```

Then:

```bash
where python
```

The first result should be:

```text
E:\EchoHands_Alpha-build\venv\Scripts\python.exe
```

So the final verification should look approximately like:

```text
(venv) E:\EchoHands_Alpha-build>python --version
Python 3.10.x

(venv) E:\EchoHands_Alpha-build>where python
E:\EchoHands_Alpha-build\venv\Scripts\python.exe
...
```

---

### If `py -3.10 --version` Fails

If you get something like:

```text
Requested Python version (3.10) is not installed
```

then **Python 3.10 is not installed on your machine**, which is why another installed version such as Python 3.13 may be used.

In that case, do not create another `venv` yet.

Install **Python 3.10.x** first, then come back to:

```bash
py -3.10 -m venv venv
```

---

## 5. Upgrade pip

Run:

```bash
python -m pip install --upgrade pip
```

**Expected result:**

pip is upgraded successfully. The final output should contain a message similar to:

```text
Successfully installed pip-...
```

---

## 6. Install Dependencies

Install the dependencies specified by the Alpha Build:

```bash
python -m pip install -r requirements.txt
```

**Expected result:**

pip downloads and installs the required packages.

The installation should finish with output similar to:

```text
Successfully installed ...
```

The exact dependencies and versions are defined in:

```text
requirements.txt
```

---

## 7. Verify the Alpha Model Files

The **Alpha Build uses models stored directly inside the project directory**. Unlike the Beta Build, the Alpha Build does not use the later versioned model-cache/download system.

Before starting the application, make sure the model files supplied with the Alpha Build are present in the repository's model directory.

The required model files should remain in their expected **in-directory model location**:

```text
EchoHands/
└── Models/
    ├── random_forest.pkl
    ├── label_encoder.pkl
    ├── dynamic_lstm.keras
    └── dynamic_label_encoder.npy
```

Do not move the model files outside the Alpha project unless the project structure has been intentionally modified.

---

# How to Operate EchoHands

## 8. Start EchoHands

From the project root, with the virtual environment still activated, run:

```bash
python -m src.app
```

**Expected result:**

EchoHands initializes the recognition system and opens the real-time camera application window.

Once the application is running, the webcam feed should be visible and the system can begin recognizing supported signs.

The Alpha Build supports:

- **Static gestures:** A–Y + 0–9
- **Dynamic gestures:** J / Z

---

<p align="center">
  <img src="Assets/alpha UI.png" alt="EchoHands Alpha Interface" width="450">
</p>

<p align="center">
  <em>Figure 2. EchoHands Alpha Build real-time recognition interface.</em>
</p>

## Understanding the Interface

The EchoHands Alpha interface provides a real-time view of the recognition process. As the user interacts with the system through hand gestures and keyboard controls, the interface continuously updates to show what EchoHands is detecting, predicting, processing, and finally accepting.

The different sections are connected to the user's interaction with the system rather than being independent displays.

### Mode

```text
Mode: NONE
```

The **Mode** indicator shows the recognition mode currently active in the application
shows what kind of gesture EchoHands is currently processing.

- NONE — The model isn't processing a gesture at the moment. This is generally shown when no hand is detected on the screen.
- STATIC — The system switches to this mode when a static sign such as an alphabet or number is detected. These signs can be recognized from a single hand position.
- DYNAMIC — The system switches to this mode for gestures where movement matters, such as J and Z. This mode will also be used for future dynamic and compound gestures such as HELLO, GOOD, LOVE, and others planned for later development.

### Prediction

```text
Prediction: No Hand Detected
```

The Prediction shows what the recognition model is currently predicting from what it sees.

No Hand Detected — No hand is currently detected by the system.
A sign such as A, B, J, or Z — The model is currently predicting that sign.

The prediction is live, so it can change as you move your hand or transition between gestures. A prediction does not automatically mean that the sign has been accepted into the text.

### Confidence

```text
Confidence: 0.0%
```

The Confidence shows how strongly the model supports its current prediction.

EchoHands uses a fixed confidence threshold to decide whether a prediction is reliable enough to be considered for recognition. Predictions below this threshold are filtered out instead of being accepted as a gesture.

This helps reduce incorrect recognitions caused by unclear hand positions, movement, or uncertain predictions.

### Sequence Frames

```text
Sequence frames: 0
```

The **Sequence Frames** indicator shows how many consecutive frames are currently being collected for dynamic gesture analysis.

This becomes important when the user performs a motion-based gesture. Unlike a static sign, which can be recognized mainly from the current frame (hand configuration), a dynamic sign requires EchoHands to observe movement over time.

As the user performs the gesture, frames are collected into a sequence. Once the sequence contains the required information, it can be analyzed by the dynamic recognition pipeline.

### Recognition State

```text
NONE — Waiting for next gesture
```

`The Recognition State shows what EchoHands is currently doing with the detected gesture.

When it shows:

NONE — Waiting for next gesture

the system has finished processing the previous gesture and is ready for the next one.

The repeated-gesture behavior is managed by EchoHands' gesture locking mechanism. Once a gesture has been accepted, the locking mechanism prevents the same held gesture from being added repeatedly until the appropriate hand or gesture transition occurs.

For example, holding the sign A for several frames should result in:

A

rather than:

AAAA

Once the gesture is accepted and locked, EchoHands waits for the gesture to be released or transitioned before allowing another recognition.

### Text Output

```text
Text: HELLO
```

The Text area shows the signs that EchoHands has actually accepted as input and added to the current text.

This part of the interface is managed by the Word Builder module, which keeps track of the accepted characters and builds them into words and text.

The important difference is:
```text
Prediction → What the model currently predicts
Text       → What EchoHands has accepted
```
So, the prediction can change while you move between gestures, but the text is updated only when a gesture passes through the recognition-control process and is accepted

As the user performs accepted signs one by one, the text is built gradually.

### Keyboard Controls

The keyboard controls are also handled through the Word Builder module and let you manage the generated text and the application.

Key	Behaviour
[SPACE]	Adds a space to the current text
[DOUBLE SPACE]	Clears the complete text output
[BACKSPACE]	Removes the last accepted character
[Q]	Closes the EchoHands application

For example, after recognizing:

HELLO

pressing [SPACE] lets you continue with the next word. [BACKSPACE] removes the last accepted character, while [DOUBLE SPACE] clears the current text.

---

# Project Structure

```text
EchoHands/
│
├── Assets/
│   ├── Project images
│   └── Detailed structural documentation
│
├── data/
│   └── Project datasets and processed data retained for development
│
├── models/
│   ├── Static recognition model artifacts
│   ├── Dynamic recognition model artifacts
│   ├── Label encoders
│   └── Related model metadata
│
├── sign description/
│   └── Supported sign reference images
│
├── src/
│   │
│   ├── app.py
│   │   Main real-time application entry point
│   │
│   ├── core/
│   │   Core real-time recognition components
│   │
│   ├── dataset/
│   │   Dataset collection, preparation and analysis modules
│   │
│   ├── training/
│   │   Training, evaluation and analysis modules
│   │
│   └── utils/
│       Utility components
│
├── tests/
│   └── Development and validation scripts
│
├── requirements.txt
│   Python dependencies
│
└── README.md
```

## Core Recognition Components

### `app.py`

`src/app.py` is the main entry point of EchoHands. It connects the major parts of the application, including camera input, hand detection, landmark processing, prediction, recognition control, sequence detection, text building, keyboard controls, and the real-time display.

The application is started with:

```bash
python -m src.app
```

### `camera.py`

Handles webcam access and frame retrieval for the real-time recognition loop.

### `hand_detector.py`

Uses MediaPipe to detect the user's hand and obtain hand landmark information from each webcam frame. It can also draw detected landmarks and hand connections on the displayed camera frame.

### `landmark_processor.py`

Processes raw hand landmark coordinates into a consistent representation that can be used by the recognition models.

### `predictor.py`

Handles static gesture prediction by loading the trained static recognition model and using processed hand features to predict the corresponding sign.

### `dynamic_predictor.py`

Handles dynamic gesture prediction from a sequence of hand information collected across multiple frames. This is used for motion-based signs such as **J** and **Z**.

### `recognition_controller.py`

Manages the real-time recognition state and helps coordinate how predictions are accepted.

### `sequence_detector.py`

Collects and manages sequences of hand information needed for dynamic gesture recognition.

### `word_builder.py`

Maintains the text currently constructed by EchoHands. It supports adding recognized characters and spaces, removing characters, clearing text, and resetting the text state.

---

## Models

The `models/` directory contains the trained model artifacts required by the Alpha application.

These include resources related to:

- Static sign recognition
- Dynamic sign recognition
- Label encoding
- Model class information
- Other model metadata

The required model files must remain in the locations expected by the source code.

Do not rename, move, or delete model artifacts unless the related source code is updated accordingly.

---

## 📊 Data capture and Preparation Modules

EchoHands includes dedicated modules for **collecting sign-language data, checking its quality, preparing datasets, creating additional features, preparing dynamic gesture sequences, training recognition models, and evaluating their performance**.

### Dataset Collection

The `src/dataset/collection/` folder contains the tools used to **capture and manage training data** from the webcam.

- **`collector.py`** — Provides the webcam interface used during dataset collection. It starts the camera, retrieves frames, and releases the camera when collection is finished.

- **`quality_checker.py`** — Checks whether a detected hand is suitable for adding to the dataset. It verifies that a hand is detected, only one hand is present, all 21 landmarks are available, coordinates are valid, and the detected hand has sufficient spatial spread.

- **`duplicate_detector.py`** — Checks whether a newly captured sample is too similar to the previously saved sample, helping reduce duplicate or nearly identical training samples.

- **`variation_manager.py`** — Guides dataset collection through different hand-position variations such as moving the hand left/right/up/down, changing distance, rotating the wrist, and tilting the palm.

- **`sequence_generator.py`** — Collects **dynamic gesture sequences** rather than individual frames. It records multiple frames for gestures such as **J** and **Z**, supports left- and right-hand collection, checks for minimum sequence length, and saves each completed sequence as an `.npz` file.


### Dataset Preparation

The `src/dataset/preparation/` folder contains scripts that convert collected data into datasets that can be used for model training.

- **`dataset_generator.py`** — Captures static sign samples from the webcam and stores their landmark features in a CSV dataset. It uses the quality checker, duplicate detector, hand detection, and variation manager while collecting samples.

- **`collect_targeted_augmentation.py`** — Collects additional samples for a specific sign to increase its representation in the dataset. It captures both left- and right-hand examples across controlled variations such as wrist rotation, palm tilt, and natural finger configuration.

- **`dataset_loader.py`** — Loads the processed static dataset, separates labels from features, encodes the labels, and creates the training/testing split used by the static model training and evaluation scripts.

- **`create_geometric_dataset.py`** — Creates additional geometric features from the 21 hand landmarks, including landmark distances and finger-joint angles. These features are combined with the original landmark features to produce the geometric dataset.

- **`prepare_dynamic_dataset.py`** — Converts the collected dynamic gesture sequences into a model-ready dataset. It adds geometric features to each frame and resamples sequences to a fixed length of **40 frames** before saving the processed dynamic dataset.

- **`split_dynamic_dataset.py`** — Splits the processed dynamic dataset into separate **training, validation, and test sets** while maintaining class distribution.

### Dataset Analysis

The `src/dataset/analysis/` folder contains small utilities for keeping track of dataset collection progress.

- **`statistics.py`** — Tracks the number of collected samples, remaining samples, completion status, and collection progress.

---

## 🧠 Training & Evaluation

The `src/training/` folder contains scripts for **training the recognition models and measuring their performance**.

### Static Model Training

- **`train_model.py`** — Trains the main static sign-recognition model using a **Random Forest classifier**. It loads the prepared dataset, trains the model, evaluates its accuracy, and saves the trained model and label encoder.

- **`trainer.py`** — Provides an alternative neural-network training pipeline using **TensorFlow/Keras**. It loads the static landmark dataset, encodes the labels, builds a small fully connected neural network, trains it, evaluates it, and saves the resulting model and label encoder.

### Dynamic Model Validation & Evaluation

- **`validate_sequences.py`** — Validates the collected dynamic J/Z sequences before they are used for training. It checks sequence shape, feature count, minimum frame count, stored label, stored hand, and invalid numerical values such as `NaN` or infinity.

- **`evaluate_dynamic_model.py`** — Evaluates the trained dynamic LSTM model using the locked test dataset. It reports accuracy, classification results, confusion matrix results, and individual predictions with confidence values.

### Static Model Evaluation

- **`evaluate_model.py`** — Evaluates the trained Random Forest model on the static test data and reports accuracy, a classification report, and a confusion matrix.

---

## 🧪 Tests & Development Scripts

The `tests/` folder contains scripts used to **verify individual components, experiment with recognition approaches, and test model behaviour during development**.

### Core Component Tests

These scripts test the recognition components individually rather than being part of the main application:

- **`tests/core/test_predictor.py`** — Checks that the static predictor and its label encoder can be loaded successfully and displays the available classes.

- **`tests/core/test_dynamic_predictor.py`** — Tests the dynamic predictor using processed J/Z sequences and reports the predicted class and confidence.

- **`tests/core/test_recognition_controller.py`** — Runs the recognition controller with the webcam and allows static signs and J/Z gestures to be tested through the complete recognition-control flow.

- **`tests/core/test_sequence_detector.py`** — Tests the dynamic sequence detector and displays its sequence states while performing J/Z gestures.

- **`tests/core/test_word_builder.py`** — Tests text construction functionality by adding characters, creating spaces, removing characters with backspace, and clearing the generated text.

### Model & Recognition Experiments

- **`tests/test_model.py`** — Loads a TensorFlow model and displays its input/output shapes and model summary.

- **`tests/test_predictor.py`** — Performs a simple predictor test using dummy landmark features and displays the predicted class and confidence.

- **`tests/training/test_dynamic_live.py`** — Tests dynamic gesture recognition directly from live webcam input using the trained dynamic LSTM model. It captures a sequence, resamples it to the required frame count, and produces a prediction and confidence.

- **`tests/training/test_geometric_model.py`** — Experiments with a Random Forest model trained on the geometric-feature dataset and reports its classification performance.

---

## 📁 Dataset Structure

The project also contains the actual collected and processed datasets used by these modules.

### Static Dataset

```text
data/processed/
├── keypoints.csv
└── keypoints_geometric.csv
```

- **`keypoints.csv`** — Stores collected static hand-landmark samples.
- **`keypoints_geometric.csv`** — Stores the static dataset after additional geometric features have been generated.

### Dynamic Dataset

```text
data/processed/
└── dynamic_sequences/
    ├── J/
    │   ├── LEFT/
    │   └── RIGHT/
    └── Z/
        ├── LEFT/
        └── RIGHT/
```

This folder contains the **individual recorded dynamic gesture sequences** for J and Z, separated by gesture and hand.

After preparation, the dynamic data is converted into:

```text
data/processed/
└── dynamic/
    ├── dynamic_dataset.npz
    └── split/
        ├── X_train.npy
        ├── y_train.npy
        ├── X_val.npy
        ├── y_val.npy
        ├── X_test.npy
        └── y_test.npy
```

This gives the training pipeline a structured dataset with fixed-length sequences ready for the dynamic recognition model.

---

They are not required for normal application usage.

Before major development changes, the source code can be checked using:

```bash
python -m compileall src tests
```

The stable application itself should then be tested using:

```bash
python -m src.app

```

## Assets and Documentation

The `Assets/` directory contains project images and detailed structural documentation related to the EchoHands architecture and development.

The `sign description/` directory contains supported sign reference images

---

# How EchoHands Works

The main recognition pipeline is:

```text
Camera Input
      ↓
Hand Detection
      ↓
Hand Landmark Extraction
      ↓
Landmark / Feature Processing
      ↓
Recognition Controller
      ↓
Static or Dynamic Recognition
      ↓
Gesture Acceptance
      ↓
WordBuilder
      ↓
Digital Text Output

```
---

              ┌─────────────────┐
              │     Webcam      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     OpenCV      │
              │  Frame Capture  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    MediaPipe    │
              │  Hand Detection │
              │  21 Landmarks   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Landmark /       │
              │ Feature          │
              │ Processing       │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
     ┌─────────────────┐ ┌─────────────────┐
     │ Static          │ │ Dynamic         │
     │ Recognition     │ │ Recognition     │
     │ Random Forest   │ │ TensorFlow LSTM │
     └────────┬────────┘ └────────┬────────┘
              │                   │
              └─────────┬─────────┘
                        ▼
              ┌─────────────────┐
              │   Recognition   │
              │    Controller   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Word Builder  │
              │   / Text Output │
              └─────────────────┘

---

The webcam continuously captures frames. MediaPipe detects the hand and extracts its landmarks. These landmarks are processed into the feature representation required by the recognition pipeline.

The system then decides whether the input should be handled through the static recognition pipeline or analyzed as part of a dynamic sequence.

For stable hand poses, the static model predicts the corresponding sign. For motion-based signs such as **J** and **Z**, the system analyzes movement across multiple frames.

Once a valid gesture is accepted, it is passed to the `WordBuilder` and added to the digital text output.

## Why Recognition Control Is Important

A webcam processes many frames every second. If a user holds the same gesture in front of the camera, the model may predict the same sign repeatedly:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → A
Frame 4 → A
```

Without recognition control, this could incorrectly produce:

```text
AAAA
```

even though the user intended to enter only one **A**.

EchoHands therefore uses gesture acceptance and recognition-state logic to prevent a held gesture from being repeatedly added to the text. The system waits for the appropriate transition or release behavior before accepting another gesture.

This is important because a model prediction alone should not automatically become a new character in the final text.

---

# Static Recognition

The static recognition model produces a predicted sign and associated confidence. The application then applies recognition logic before allowing that sign to affect the final text output.

A normal interaction with EchoHands follows this sequence:

```text
User shows hand to camera
        ↓
Hand is detected
        ↓
Interface updates the prediction and confidence
        ↓
Static pose or motion sequence is processed
        ↓
Recognition control decides whether to accept the gesture
        ↓
Accepted sign is added to the text output(World builder)
        ↓
System waits for the next gesture

---

# Dynamic Recognition

Some gestures contain meaningful motion, so a single frame does not provide enough information.

EchoHands therefore uses a sequence-based workflow:

```text
Hand Movement
→ Landmark Frames
→ Sequence Collection
→ Movement / Sequence Detection
→ Dynamic Model
→ Dynamic Prediction
```

The current dynamic recognition system supports:

- **J**
- **Z**

The dynamic model analyzes information collected across multiple frames to determine which motion-based sign was performed.

For dynamic gestures, the interaction additionally involves collecting multiple frames:

```text
User performs hand movement
        ↓
Sequence Frames increases
        ↓
Movement sequence is analyzed
        ↓
Dynamic gesture is predicted
        ↓
Recognition control accepts the gesture
        ↓
Character is added to Text
```
---

The interface therefore gives the user feedback throughout the interaction — from hand detection, to live prediction, to sequence processing, to final text generation.

---



---


## Modular Mobile Application Direction

A future version can separate the recognition system into modular parts so that the mobile application does not need to contain the complete training or heavy model-development environment.

A possible structure is:

```text
Mobile Application
        │
        ├── Camera Input
        ├── User Interface
        └── Communication Layer
                │
                ▼
        Remote Recognition Service
                │
                ├── Hand / Feature Processing
                ├── Static Recognition
                ├── Dynamic Recognition
                └── Trained Models
                │
                ▼
           Prediction Result
                │
                ▼
          Mobile Interface
```

In this direction, the mobile device would mainly provide the camera and user interface, while the recognition models could run remotely.


# Future-development

## Cloud-Based Recognition Architecture

The longer-term idea is to support a cloud or remote architecture where model inference is performed on infrastructure controlled by the project.

Instead of requiring every mobile device to run the complete recognition stack locally:

```text
Mobile Phone
      ↓
Camera / Recognition Input
      ↓
Remote or Cloud Service
      ↓
EchoHands Models
      ↓
Prediction
      ↓
Result Returned to Phone
```

This could make the mobile application lighter and make model updates easier to manage centrally.

The exact architecture is a future development goal and would require further work on networking, latency, privacy, security, scalability, and deployment.

---

# 👤 Author

**Shivanshu Khode**

<p align="center">
  <a href="mailto:shivanshukhode043@gmail.com">
    <img src="https://cdn.simpleicons.org/gmail" alt="Email" width="35">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/shivanshu-khode-a85343379">
    <img src="https://cdn.simpleicons.org/linkedin" alt="LinkedIn" width="35">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.instagram.com/shivanshu_khode/?hl=en">
    <img src="https://cdn.simpleicons.org/instagram" alt="Instagram" width="35">
  </a>
</p>
