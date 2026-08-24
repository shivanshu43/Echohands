# Progress Log - Project Architecture Refinement & Dataset Generation

After modularizing the application into separate components, the next objective was to improve the overall project structure and prepare it for the upcoming machine learning pipeline. Although the application was working, the project organization still needed refinement to make future development easier and more maintainable.

---

## 1. Refined the Project Structure

The project was reorganized into a proper Python package structure where each directory has a dedicated responsibility.

```text
Project
│
├── src/
│   ├── core/
│   │   ├── camera.py
│   │   ├── hand_detector.py
│   │   ├── landmark_processor.py
│   │   └── predictor.py
│   │
│   ├── training/
│   │   ├── dataset_generator.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── labels.py
│   │   └── utils.py
│   │
│   └── app.py
│
├── tests/
├── data/
├── models/
└── outputs/
```

Instead of keeping all Python files together, related files are now grouped based on their functionality. This makes the project much easier to navigate and also provides a clear separation between application logic, training scripts, utility modules, and testing files.

---

## 2. Converted the Project into a Python Package

To allow Python to recognize each folder as a package, `__init__.py` files were added inside the `src`, `core`, `training`, and `utils` directories.

This allowed the project to use proper package imports instead of relying on relative paths or manually executing individual files.

Example:

```python
from src.core.camera import Camera
from src.utils.config import WINDOW_NAME
```

This approach keeps imports consistent throughout the project and avoids module resolution issues.

---

## 3. Standardized the Import Structure

During development, some files imported modules using:

```python
from core.camera import Camera
```

while others used

```python
from src.core.camera import Camera
```

Having multiple import styles can lead to inconsistent behavior depending on how the program is executed.

The entire project was updated to follow a single import convention:

```python
from src....
```

This makes every module follow the same structure and improves maintainability.

---

## 4. Removed Duplicate Modules

While reviewing the project structure, it was discovered that two different `predictor.py` files existed.

Only one predictor implementation was required, so the duplicate file was removed to eliminate confusion and ensure that the project has a single source of truth for prediction logic.

---

## 5. Organized Testing Files

Testing scripts were separated from the main application.

```text
tests/
├── test_model.py
└── test_predictor.py
```

Keeping test files outside the application code results in a cleaner project structure and follows common software engineering practices.

---

## 6. Centralized Configuration

Instead of hardcoding values throughout the project, common configuration values were moved into `config.py`.

Some examples include:

- Camera Index
- Window Name
- Frame Dimensions
- Model Path

Now, changing one of these values only requires updating a single file instead of searching through the entire codebase.

---

## 7. Improved Project Execution

The project now follows Python's package execution model.

Instead of running scripts directly:

```bash
python src/training/dataset_generator.py
```

the project is executed as:

```bash
python -m src.app

python -m src.training.dataset_generator A
```

This ensures that all modules are imported correctly and prevents package-related import errors.

---

# Building the Dataset Generation Pipeline

Once the project structure was finalized, work shifted towards creating our own dataset generation pipeline instead of relying entirely on the inherited dataset.

The objective of this phase was to capture hand landmarks directly from the webcam and save them in a format that can later be used for training our own machine learning model.

The overall workflow is shown below.

```text
                 python -m src.training.dataset_generator A
                                 │
                                 ▼
                            Camera
                                 │
                                 ▼
                         Hand Detector
                                 │
                                 ▼
                    Landmark Processor
                                 │
                                 ▼
                   Extract 42 Landmark Features
                                 │
                                 ▼
                          keypoint.csv
```

The dataset generator successfully performs the following steps:

- Captures frames from the webcam.
- Detects the user's hand using MediaPipe.
- Extracts the 21 hand landmarks.
- Converts them into the required 42 numerical features.
- Associates the captured sample with the provided label.
- Stores the extracted features inside `keypoint.csv`.

The generated CSV file will become the primary dataset for the upcoming training phase.

---

# Current Application Architecture

The application now follows a much cleaner modular workflow.

```text
                         User
                           │
                           ▼
                    python -m src.app
                           │
                           ▼
                        app.py
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
   Camera            Hand Detector     Landmark Processor
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                      Predictor
                           │
                           ▼
                     TensorFlow Model
                           │
                           ▼
                       Prediction
```

Each module now performs only one dedicated responsibility, making the project easier to understand, debug, extend, and maintain.

---

# Progress Summary

At the end of this phase, the project has achieved the following:

- Modular application architecture.
- Clean Python package structure.
- Consistent import system across the project.
- Organized testing environment.
- Centralized configuration management.
- Removal of duplicate modules.
- Working dataset generation pipeline.
- Successful generation of labeled landmark datasets from the webcam.
- Solid foundation for implementing the complete machine learning training pipeline.

The next phase will focus on improving the dataset collection process, training a custom classification model, evaluating its performance, and eventually replacing the inherited model with one trained entirely on our own collected dataset.