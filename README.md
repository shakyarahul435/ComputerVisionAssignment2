# Computer Vision Project: Classification, Detection, and Tracking

This repository contains Jupyter notebooks for three core computer vision tasks, utilizing PyTorch, YOLOv8, and OpenCV.

---

### 🚀 Tasks Overview

| Task | Notebook | Description | Models/Algorithms | Validation/Data |
| :--- | :--- | :--- | :--- | :--- |
| **1: Image Classification** | `st125982_notebook_task_1.ipynb` | Training and evaluation on the **CIFAR-10** dataset. | **SimpleCNN** (Custom) and **ResNet18** (Transfer Learning) | CIFAR-10 (downloaded to `data/`) |
| **2: Object Detection** | `st125982_notebook_task_2.ipynb` | Comparative analysis of two object detection models on images and a video. | **Faster R-CNN** and **YOLOv8n** | `video.mp4`, images in `photos/`. Validation uses `coco128_custom.yaml` |
| **3: Object Tracking** | `st125982_notebook_task_3.ipynb` | Comparison of traditional correlation filter-based object trackers on a video. | **KCF**, **CSRT**, and **MOSSE** (OpenCV) | `video.mp4` |

---

### 📦 Project Structure and File Locations
```
Assignment2/
├── 📁 coco128/ # Small COCO dataset for quick testing
│ └── ... (COCO128 data files)
│
├── 📁 data/ # CIFAR dataset and related files
│ └── ... (CIFAR data)
│
├── 📁 photos/ # Test images for object detection/tracking
│ └── ... (test photos)
│
├── 📁 runs/ # YOLO detection results and runs
│ └── detect/
│ └── ... (detection outputs)
│
├── 📁 task_2_outputs/ # Outputs from Task 2 (Object Detection)
│ └── ... (images, predictions)
│
├── 📁 task3_outputs/ # Outputs from Task 3 (Object Tracking)
│ ├── KCF/
│ ├── CSRT/
│ └── MOSSE/
│
├── 📁 virenv/ # Virtual environment (ignored in Git)
│ └── ...
│
├── 📁 zips/ # Submitted ZIP files for assignment
│ └── ...
│
├── .gitignore # Git ignore rules
├── coco128_custom.yaml # Custom dataset config for COCO128
├── jet.jpg # Example image or test image
├── README.md # Project documentation
├── requirement.txt # Python dependencies
├── resnet18_finetuned.pth # Fine-tuned ResNet-18 weights
├── simplecnn_best.pth # Best model weights for SimpleCNN
│
├── st125982_notebook_task_1.ipynb # Task 1 — Image Classification
├── st125982_notebook_task_2.ipynb # Task 2 — Object Detection
├── st125982_notebook_task_3.ipynb # Task 3 — Object Tracking
│
├── tracking_CSRT.mp4 # Output video using CSRT tracker
├── tracking_KCF.mp4 # Output video using KCF tracker
├── tracking_MOSSE.mp4 # Output video using MOSSE tracker
├── tracking_output.mp4 # Combined tracking result
├── video.mp4 # Original input video
│
├── yolov8n.pt # YOLOv8 Nano model
├── yolov8s.pt # YOLOv8 Small model
└── yolov8x-seg.pt # YOLOv8 segmentation model
```
---

### 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shakyarahul435/ComputerVisionAssignment2.git   #<Repository URL>
    cd ComputerVisionAssignment2   #<Repository Name>
    ```

2.  **Create and activate a virtual environment (`virenv`):**
    ```bash
    python -m venv virenv  # It’s recommended to use **Python 3.11.9**, as some newer Python versions may have limited support for PyTorch and torchvision.
    source virenv/bin/activate  # Linux/macOS
    # .\virenv\Scripts\activate  # Windows/PowerShell
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: PyTorch and Torchvision versions are specific. If running on a GPU machine, you may need to install the CUDA-specific binaries, e.g., `pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128`)*

4.  **Data Preparation:**
    * Place your **test images** inside the `./photos` folder.
    * Place the **test video** file named `video.mp4` in the project's root directory.
    * Ensure `coco128_custom.yaml` is in the root directory.
    * The `data/` (CIFAR) and `coco128/` (COCO) folders are for datasets and will be populated when you run the notebooks.

5.  **Run the Notebooks:**
    Launch the Jupyter environment and run the notebooks in order:
    ```bash
    jupyter notebook
    ```

    ```
    1. Outputs of task 2 stored in task_2_outputs whereas other outputs can be seen on ***runs/detect/val7*** and ***runs/detect/val8*** Folder with: 
    -- Confusion Matrix,
    -- Confusion Matrix Normalized
    -- Recall Curve, 
    -- Precision Curve,
    -- F1 Curve
    -- val_batch_labels
    -- val_batch_predictions
    
    2. Outputs of task 3 stored in task3_outputs:
    -- CSRT Tracking
    -- KCF Tracking
    -- MOSSE Tracking
    -- Tracking Output
    ```

---