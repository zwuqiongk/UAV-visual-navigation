# Landmark-Aware Visual Navigation for UAVs in GNSS-Denied Wind Farms

## 📖 Introduction
This repository implements a robust **Landmark-Aware Visual Navigation System** designed for Unmanned Aerial Vehicles (UAVs) operating in GNSS-denied wind farm environments.

Visual navigation in complex environments often suffers from detection jitter, occlusion, and perspective distortion. To address these challenges, this project integrates **YOLOv12** for high-performance object detection and **BoT-SORT** for multi-object tracking. We propose two novel mechanisms to enhance localization accuracy:

1.  **Landmark Binding Lifecycle Management (LBL):** A Finite State Machine (FSM) strategy that filters out transient "ghost" detections and ensures temporal consistency (Detected $\to$ Confirmed $\to$ Bound).
2.  **Adaptive Spatio-Temporal Weighted Fusion (ASTW):** A probabilistic framework that weights landmark observations based on their spatial distance and visual fidelity, minimizing errors caused by edge distortion and long-range estimation.

The system is validated in a high-fidelity simulation environment powered by **RflySim**.

## 📂 File Structure

### Core Implementation
*   **`binding.py`**: The main entry point for the navigation system. It integrates the flight control loop, vision capture, object tracking, and the localization algorithm.
*   **`LandmarkNavSystem.py`**: Contains the core algorithmic classes:
    *   `LandmarkNavSystem`: Implements the Lifecycle Management FSM and the Adaptive Weighted Fusion strategy.
*   **`track.py`**: A standalone script to test and visualize the tracking performance of YOLOv12 + BoT-SORT without the full flight control loop.

### Data Processing & Training
*   **`train.py`**: Script for training the YOLOv12 model on the custom wind turbine dataset.
*   **`Data_set_division.py`**: Utility to split the dataset into training and validation sets.
*   **`Dataset_annotation.py`**: Tools for processing and formatting dataset annotations 
*   **`Video_frame_segmentation.py`**: Extracts frames from recorded videos to build the dataset.

### Simulation & Utilities
*   **`Video_recording.py`**: Controls the UAV in the simulation to record flight footage for dataset creation.
*   **`only_fly.py`**: Simple script for testing UAV flight control commands independent of visual processing.
*   **`requirements.txt`**: List of Python dependencies required to run the project.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/Your-Repo-Name.git
    cd Your-Repo-Name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Simulation Environment:**
    This project relies on the **RflySim** platform for HITL/SITL simulation. Ensure RflySim is installed and the `UE4CtrlAPI` and `PX4MavCtrl` interfaces are accessible.

## 🙏 Acknowledgments

*   **RflySim Platform**: We extensively use [RflySim](https://rflysim.com/) for our high-fidelity simulation environment, which supports photorealistic rendering and physics-based UAV control.
*   **Ultralytics**: For the implementation of the YOLO object detection framework.
