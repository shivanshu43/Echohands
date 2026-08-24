<p align="center">
  <img src="Assets/alpha build banner.png" alt="EchoHands Banner" width="100%">
</p>

# EchoHands — Alpha Build

EchoHands is an AI-powered real-time American Sign Language (ASL) recognition system that uses a webcam to detect supported hand signs and convert recognized gestures into digital text.

The Alpha Build is the first complete working milestone of EchoHands. It brings together hand detection, landmark processing, static sign recognition, dynamic gesture recognition, recognition-state control, and word building into one real-time application.

> **Alpha Build status:** The core recognition system is working and serves as the stable baseline for future development.

---

## ✨ What EchoHands Can Do

The current Alpha Build includes:

- Real-time webcam-based hand detection
- MediaPipe hand landmark extraction
- Static ASL gesture recognition
- Dynamic gesture recognition for **J** and **Z**
- Sequence-based motion analysis
- Confidence-based prediction filtering
- Recognition-state control
- Protection against repeated recognition of a held gesture
- Word and text building
- Space, clear, and backspace controls

---

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

During execution, the webcam continuously captures frames.

The system detects the user's hand using MediaPipe and extracts the hand landmarks. These landmarks are processed into the feature representation required by the recognition pipeline.

The recognition controller manages how the real-time system behaves. It helps decide whether the current input should be handled through the static recognition pipeline or whether hand movement should be analyzed as part of a dynamic sequence.

For stable hand poses, the static recognition model predicts the corresponding sign.

For motion-based gestures such as **J** and **Z**, the system analyzes a sequence of frames rather than relying on a single frame.

Once a valid gesture is accepted, it is passed to the `WordBuilder`, which adds it to the recognized text.

> The `Assets/` directory also contains project images and detailed structural documentation related to the EchoHands architecture and development.

---

# Why Recognition Control Is Important

A webcam processes many frames every second.

If the user holds the same gesture in front of the camera, the recognition model may predict the same sign repeatedly:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → A
Frame 4 → A
```

Without recognition-state control, the application could incorrectly produce:

```text
AAAA
```

even though the user intended to enter only one **A**.

EchoHands therefore includes gesture acceptance and recognition-state logic to prevent the same held gesture from being repeatedly added to the text.

The system waits for the appropriate gesture transition or release behavior before accepting another gesture.

This makes real-time text construction significantly more reliable.

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

---

# Core Recognition Components

## `app.py`

`src/app.py` is the main entry point of EchoHands.

It initializes and connects the major parts of the application, including:

- Camera input
- Hand detection
- Landmark processing
- Static prediction
- Dynamic prediction
- Recognition control
- Sequence detection
- Word building
- Keyboard controls
- Real-time display

The normal application is started with:

```bash
python -m src.app
```

---

## `camera.py`

Handles webcam access and frame retrieval for the real-time recognition loop.

---

## `hand_detector.py`

Uses MediaPipe to detect the user's hand and obtain hand landmark information from each webcam frame.

It can also draw detected landmarks and hand connections on the displayed camera frame.

---

## `landmark_processor.py`

Processes raw hand landmark coordinates into a consistent representation that can be used by the recognition models.

This processing helps convert camera-detected landmark information into machine-learning features.

---

## `predictor.py`

Handles static gesture prediction.

It loads the trained static recognition model and uses processed hand features to predict the corresponding sign.

---

## `dynamic_predictor.py`

Handles dynamic gesture prediction.

Unlike static prediction, dynamic prediction operates on a sequence of hand information collected across multiple frames.

This is used for motion-based signs such as **J** and **Z**.

---

## `recognition_controller.py`

The recognition controller manages the real-time recognition state of the application.

It helps coordinate recognition behavior and prevents unstable frame-by-frame predictions from directly becoming text.

This component is an important part of making the live recognition system behave more reliably.

---

## `sequence_detector.py`

Collects and manages sequences of hand information needed for dynamic gesture recognition.

It is primarily used when the system needs to analyze movement over time.

---

## `word_builder.py`

Maintains the text currently constructed by EchoHands.

It supports:

- Adding recognized characters
- Adding spaces
- Removing the last character
- Clearing the complete text
- Returning the current text
- Resetting the text state

---

# Static Recognition

Static gestures are recognized from the hand configuration detected in a frame.

The general process is:

```text
Camera Frame
→ Hand Detection
→ Landmark Extraction
→ Feature Processing
→ Static Model
→ Predicted Sign
→ Recognition Acceptance
→ WordBuilder
```

The static recognition model produces a predicted sign and associated confidence.

The application then applies its recognition logic before allowing that sign to affect the final text output.

This separation is important because a model prediction alone should not automatically mean that a new character must be added.

---

# Dynamic Recognition

Some gestures contain meaningful motion.

For these signs, a single frame does not provide enough information.

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

The dynamic model analyzes information collected across multiple frames and uses the sequence to determine which motion-based sign was performed.

---

# Application Controls

The Alpha Build includes the following controls:

## Add Gesture

A recognized gesture is added automatically after it satisfies the application's recognition and gesture-acceptance conditions.

## Add Space

Press:

```text
SPACE
```

to add a space to the current text.

## Clear Text

Press:

```text
SPACE twice quickly
```

to clear the current text.

## Backspace

Press:

```text
BACKSPACE
```

to remove the last character.

## Exit

Press:

```text
Q
```

to close the application.

---

# Installation and Setup

## 1. Clone the Repository

Clone the EchoHands repository:

```bash
git clone https://github.com/shivanshu43/EchoHands_Alpha-build.git
```

Then move into the project directory:

```bash
cd EchoHands
```

If the Alpha repository is private or access-restricted, you must first receive permission from the project owner.

---

## 2. Recommended Python Version

The Alpha Build is recommended to run with:

```text
Python 3.10
```

Using the same Python version helps avoid dependency and compatibility issues.

---

## 3. Create a Virtual Environment

Each developer should create their own virtual environment.

### Windows

```bash
py -3.10 -m venv venv
```

If `py -3.10` is not available but Python 3.10 is already the active interpreter:

```bash
python -m venv venv
```

---

## 4. Activate the Virtual Environment

### Windows Command Prompt

```bash
venv\Scripts\activate
```

After successful activation, the terminal should display something similar to:

```text
(venv)
```

Verify the active Python interpreter:

```bash
where python
```

The first result should point to:

```text
EchoHands\venv\Scripts\python.exe
```

Also verify the Python version:

```bash
python --version
```

---

## 5. Install Dependencies

After activating the virtual environment, upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Using:

```bash
python -m pip
```

instead of only:

```bash
pip
```

helps ensure that packages are installed into the currently active virtual environment.

---

# Running EchoHands

From the root directory of the project, run:

```bash
python -m src.app
```

If the environment, dependencies, models, and camera access are correctly configured, EchoHands will initialize the recognition system and open the webcam window.

---

# Models

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

# Data, Dataset and Training Modules

The Alpha repository may still contain modules used during the development of the recognition system.

These include functionality related to:

- Data collection
- Duplicate detection
- Dataset quality checking
- Variation management
- Sequence generation
- Dataset preparation
- Feature generation
- Data augmentation
- Model training
- Model evaluation
- Validation
- Cross-validation
- Error analysis
- Recognition experiments

These modules were useful while building, training, testing, and improving EchoHands.

They are **not required for a normal user to run the real-time application**.

The normal application entry point is:

```bash
python -m src.app
```

They are retained in the Alpha build because the repository also serves as the technical development baseline for future improvement, retraining, debugging, and experimentation.

---

# Tests and Development Scripts

The `tests/` directory contains scripts used during development and validation.

These scripts were used to verify different parts of the recognition pipeline, including core behavior, model behavior, dynamic recognition, sequence handling, feature handling, and training-related experiments.

They are not required for normal application usage.

Before major development changes, the source code can be checked for syntax and compilation issues using:

```bash
python -m compileall src tests
```

The stable application itself should then be tested using:

```bash
python -m src.app
```

---

# Requirements

The project's Python dependencies are defined in:

```text
requirements.txt
```

The project uses libraries and technologies for:

- Real-time camera processing
- Hand tracking
- Numerical processing
- Dataset handling
- Machine learning
- Deep learning
- Model serialization

The exact dependency versions should be installed from the project's `requirements.txt`.

---

# Collaborator Guidelines

If you are given access to the EchoHands Alpha repository:

1. Create your own virtual environment. Do not copy another developer's `venv` directory.

2. Install dependencies using:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the application from the project root:

   ```bash
   python -m src.app
   ```

4. Do not directly modify the stable Alpha baseline without first creating a separate Git branch.

5. Do not replace trained model files unless you are intentionally testing or developing a new model version.

6. When training or retraining a model, record:
   - Dataset version
   - Feature representation
   - Model type
   - Label encoding
   - Training configuration
   - Evaluation results

7. After significant source changes, check the project with:

   ```bash
   python -m compileall src tests
   ```

8. Keep experimental work separate from the stable recognition pipeline whenever possible.

---

# Git Recommendations

The repository should not normally track local virtual environments or Python cache files.

The following should generally remain excluded through `.gitignore`:

```text
venv/
__pycache__/
*.pyc
```

Each collaborator should recreate their environment locally.

---

# Alpha Build Purpose

The Alpha Build represents the technical foundation of EchoHands.

Its purpose is to provide a stable working recognition system containing the complete pipeline developed so far, while also preserving the supporting development structure needed for future work.

The Alpha Build is intended primarily for controlled development and collaborator access.

Future work can build on this version without losing the core recognition functionality that has already been completed.

---

# Assets and Documentation

The `Assets/` directory contains more than application images.

It also contains project-related visual material and detailed structural documentation that can help explain the EchoHands architecture, modules, workflow, and development structure.

These documents are useful for understanding the internal organization of the project and should be retained as part of the Alpha build documentation.

---

# Future Development

The next major stage of EchoHands development is the **Beta Build**.

The Beta Build will focus more strongly on the user-facing application experience while preserving the stable recognition functionality developed during Alpha.

Planned directions include:

- Cleaner and more polished user interface
- Improved visual presentation of hand landmarks
- Multi-colored landmark visualization
- Clear prediction or recognized text display
- Improved recognition robustness
- Better user experience
- Additional dynamic gestures
- Expanded sign vocabulary
- Improved accessibility
- Better performance under different lighting and camera conditions
- Application packaging and deployment
- Controlled model distribution and application access

---

# Alpha → Beta Direction

The Alpha Build focuses on proving and stabilizing the complete recognition pipeline.

The Beta Build will build on that stable foundation.

```text
Alpha Build
Stable Recognition Pipeline
        ↓
Beta Build
Cleaner UI + Improved User Experience
        ↓
Future Versions
Expanded Recognition + Deployment + Accessibility
```

The goal is to improve the presentation and accessibility of EchoHands without breaking the core functionality already achieved in Alpha.

---

# License and Access

EchoHands Alpha is maintained as a controlled development project.

Repository access may be restricted.

If you are granted access to the repository, please do not redistribute the source code, trained models, datasets, documentation, or other project assets without permission from the project owner.

---

# Author

**Shivanshu Khode**

**EchoHands — AI-powered real-time sign language recognition.**


---

# Detailed Future Development Plan

EchoHands is planned to evolve in stages rather than changing the stable Alpha system all at once. The main principle is to preserve the working recognition pipeline developed during Alpha while moving experimental development, user-interface improvements, broader recognition capabilities, and deployment work into later versions.

## Phase 1 — Preserve the Alpha Baseline

The Alpha Build should remain the stable technical baseline of the project.

Before making major changes, this version should be treated as the known working state containing the completed recognition pipeline, trained models, dynamic gesture support, recognition control, and text-building behavior.

Future experimental changes should preferably be developed separately rather than directly replacing the stable Alpha implementation.

This makes it possible to return to a working version if a new experiment breaks the recognition system.

---

## Phase 2 — Beta Build: Clean User-Facing Experience

The immediate next step is the Beta Build.

The main objective of Beta is not to rebuild the recognition system from scratch. The core recognition functionality developed in Alpha should remain the foundation.

The focus will instead move toward creating a cleaner, simpler, and more polished user experience.

The intended Beta interface is centered around the live camera experience. The camera view should remain the primary visual element, while unnecessary development-oriented modules and interfaces should not be exposed to a normal user.

The planned visual direction includes:

- A clean and minimal camera-based interface
- Multi-colored hand landmarks and connections
- A visually clearer representation of the detected hand
- A predicted letter, word, or recognized output displayed clearly near the bottom center
- Reduced visual clutter
- Better spacing and presentation
- A more product-like experience compared with the development-oriented Alpha interface

The goal is to make EchoHands feel less like a technical prototype and more like an application while preserving the underlying recognition behavior.

---

## Phase 3 — Separate Development Components from the Public Beta Application

The Alpha repository contains modules that were useful during development, including dataset collection, dataset preparation, training, evaluation, analysis, and testing utilities.

These modules are valuable for continued research and development, but a normal Beta user does not need them in order to run the recognition application.

The Beta release can therefore be organized around a smaller runtime-focused application containing only the components needed for actual recognition.

The development-oriented modules can remain part of the restricted Alpha or internal development environment.

Conceptually, the project can move toward:

```text
Restricted Alpha / Development Environment
        │
        ├── Dataset collection
        ├── Dataset preparation
        ├── Training
        ├── Evaluation
        ├── Analysis
        ├── Testing
        └── Experimental development
                │
                ▼
        Stable trained models
                │
                ▼
Public Beta Application
        │
        ├── Runtime dependencies
        ├── Core recognition pipeline
        ├── Required models
        ├── Camera interface
        └── Clean user-facing UI
```

This separation allows development work to continue without forcing end users to receive every internal development tool.

---

## Phase 4 — Controlled Access and Model Protection

The Alpha Build should remain restricted to trusted collaborators.

Access to the Alpha source repository can be controlled through a private repository and by granting access only to approved collaborators.

The stable public-facing Beta application can be distributed separately from the complete Alpha development environment.

One future direction discussed for this separation is containerization.

A containerized runtime can package the application environment and the required dependencies together so that users do not need to manually reproduce the complete development setup.

The intended structure can eventually separate:

```text
Development Source
        ↓
Model Training / Improvement
        ↓
Stable Model Release
        ↓
Runtime Packaging
        ↓
Controlled Beta Distribution
```

This approach would help distinguish the internal development system from the application intended for public use.

However, containerization should be understood as an environment and deployment solution, not as absolute protection for model files. If model artifacts are physically distributed to an end user's machine, they should be treated as potentially extractable.

For stronger protection in future public releases, the architecture can move toward keeping sensitive models on infrastructure controlled by the project and exposing only the recognition service or application interface where practical.

---

## Phase 5 — Docker and Deployment Packaging

A later development stage can introduce Docker containerization for the runtime application.

The purpose would be to create a reproducible execution environment containing the required application dependencies and runtime configuration.

A containerized Beta release could help reduce problems caused by:

- Different Python versions
- Missing dependencies
- Incorrect package versions
- Local environment conflicts
- Manual installation mistakes

The general idea is:

```text
EchoHands Application
+ Required Python Environment
+ Required Dependencies
+ Runtime Configuration
        ↓
Docker Image
        ↓
Consistent Execution Environment
```

The exact deployment architecture should be decided when the Beta runtime structure is finalized.

---

## Phase 6 — Improve Recognition Robustness

After the Beta interface is stable, further work can focus on improving recognition quality.

Potential areas include:

- More reliable gesture initiation
- Better handling when a hand enters the camera frame
- Reduced accidental recognition
- Improved confidence filtering
- Better separation between repeated gestures and intentionally repeated letters
- Improved performance under different lighting conditions
- Better tolerance to different hand positions and orientations
- More variation in training data
- Better handling of different users and backgrounds

The aim is to make the system more reliable in realistic usage rather than only under controlled development conditions.

---

## Phase 7 — Expand Dynamic Gesture Recognition

The current dynamic recognition system supports J and Z.

Future work can extend the sequence-based approach to additional signs or gestures that require motion analysis.

This would involve:

1. Identifying motion-based signs
2. Collecting representative sequences
3. Preparing sequence datasets
4. Training or improving dynamic models
5. Evaluating recognition accuracy
6. Integrating the new gestures into the live recognition controller

The existing separation between static and dynamic recognition provides a useful foundation for this expansion.

---

## Phase 8 — Move Beyond Isolated Character Recognition

The current system builds recognized characters into text.

Future development can investigate higher-level interpretation.

Possible directions include:

```text
Individual Signs
        ↓
Character Sequences
        ↓
Word Formation
        ↓
Improved Interpretation
        ↓
More Natural Communication Output
```

This should be approached carefully because sign language communication is not simply a one-to-one replacement of every spoken-language word with individual alphabet characters.

The initial future goal is therefore to improve the current recognition and text-building experience before attempting broader language-level interpretation.

---

## Phase 9 — Expand Accessibility and Platform Support

After the recognition pipeline and user interface become sufficiently stable, EchoHands can be explored on additional platforms.

Potential directions include:

- Mobile camera integration
- Portable camera-based recognition
- Web-based interfaces
- Cloud-assisted recognition
- Access from compatible camera-enabled devices

The long-term objective is to make the recognition system easier to access without requiring every user to reproduce the full development environment.

---

## Phase 10 — Long-Term Architecture

The project can gradually evolve toward a layered architecture:

```text
                 Development Layer
        Data + Training + Evaluation + Research
                         │
                         ▼
                   Model Layer
             Stable trained model versions
                         │
                         ▼
                 Recognition Layer
        Hand detection + features + prediction
                         │
                         ▼
                 Application Layer
           UI + camera + interaction + output
                         │
                         ▼
                 Deployment Layer
        Packaging + distribution + access control
```

This separation will make it easier to improve one part of the system without unnecessarily disturbing the others.

---

# Current Roadmap

The practical development sequence currently planned is:

```text
Alpha Build — Completed Stable Baseline
        ↓
Lock and preserve Alpha
        ↓
Beta Build — Clean Runtime-Focused UI
        ↓
Separate public runtime from internal development modules
        ↓
Improve deployment and controlled distribution
        ↓
Improve recognition robustness
        ↓
Expand supported gestures and recognition capability
        ↓
Explore broader accessibility and platform support
```

The immediate next priority is therefore the **Beta Build**, with the Alpha Build preserved as the stable reference implementation.

