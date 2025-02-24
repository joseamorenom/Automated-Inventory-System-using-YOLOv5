# Automated Classroom Inventory System Using YOLOv5

This project implements an automated inventory system for university classrooms using deep learning-based object detection (YOLOv5). The system identifies and counts objects such as backpacks, chairs, fans, and podiums, reducing manual effort and improving accuracy.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Organization](#dataset-organization)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running Inference](#running-inference)
  - [Generating Inventory Results](#generating-inventory-results)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The objective of this project is to develop an automated inventory system for university classrooms using deep learning object detection. The proposed system identifies common classroom items—backpacks, chairs, fans, and podiums—by fine-tuning a pre-trained YOLOv5 model on a custom dataset collected from real-world classroom scenarios at the University of Antioquia in Medellín. By automating the inventory process, the system reduces manual labor and minimizes human errors, thereby enabling more efficient resource management.

---

## Dataset Organization

The original dataset was collected with images of each object type captured individually as well as in full classroom scenes. Initially, each class had its own folder (e.g., `Backpacks`, `Chair`, `Fan`, `Podium`) with a subfolder named **Tags** containing YOLO-format labels and a `classes.txt` file. These were reorganized into a unified structure for training, validation, and testing.

---

## Installation
### Prerequisites
- Python (>= 3.8)
- pip
- Git
- An NVIDIA GPU with CUDA (recommended for training)
### Setup
Clone the YOLOv5 repository and install dependencies:
- git clone https://github.com/ultralytics/yolov5.git
- cd yolov5
- pip install -r requirements.txt

Ensure your dataset is organized as shown above in the processed_dataset folder and is placed relative to the YOLOv5 repository.


---
## Usage
Training the Model
From the yolov5 directory, run the following command to train the model:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data "../processed_dataset/dataset.yaml" --weights yolov5s.pt --workers 0
```
- --img 640: Resizes images to 640x640 pixels.
- --batch 16: Batch size of 16.
- --epochs 50: Train for 50 epochs.
- --data: Path to the dataset.yaml file.
- --weights yolov5s.pt: Uses pre-trained YOLOv5s weights for fine-tuning.
- --workers 0: Number of DataLoader workers (recommended on Windows).
# Running Inference
To test the trained model on new images (e.g., full classroom scenes):

Place your test images in a folder, e.g., test_images/.
From the yolov5 directory, run:

python detect.py --weights runs/train/expXX/weights/best.pt --source "C:/path/to/test_images" --img 640 --conf 0.25

Replace expXX with your specific experiment folder. The output images with detections will be saved in runs/detect/exp.

## Generating Inventory Results
To generate a tabular inventory of detected objects, run the provided script:

python detect_and_inventory.py
This script:
- Loads the trained model.
- Processes each test image.
- Draws bounding boxes on detected objects.
- Counts detections per class.
- Displays the results in a formatted table and exports them to CSV.
- Adjust paths within the script if necessary.

## Project Structure

project_folder/
├── yolov5/                    # YOLOv5 repository (cloned)
│   ├── train.py               # Training script
│   ├── detect.py              # Inference script
│   └── ...                    # Other YOLOv5 files
├── processed_dataset/         # Organized dataset (images, labels, dataset.yaml)
├── test_images/               # Folder for test images
├── detect_and_inventory.py    # Script for inference and inventory generation
└── README.md                  # This file
Ensure that your dataset is correctly placed relative to the YOLOv5 repository for the scripts to function properly.

---
## Future Work
### Dataset Expansion:
Expand the dataset by including additional object categories such as video beams, different types of chairs, boards, and televisions. This will enhance the model’s ability to generalize and distinguish between similar objects, particularly between chairs and backpacks.

### Enhanced Class Differentiation:
Improve differentiation between overlapping objects by capturing images that clearly separate similar items. Refining the annotation process and incorporating segmentation techniques could further enhance accuracy.

### Multiple YOLO Models:
Evaluate and integrate different YOLO architectures (e.g., YOLOv5, YOLOv8) to develop specialized models for specific object groups. This modular approach may increase overall detection performance by tailoring models to distinct object categories.

### Real-Time Deployment:
Implement the model on edge devices such as the NVIDIA Jetson Nano for real-time inventory updates, enabling practical, on-site usage by administrative staff.

### User Interface Integration and Advanced Evaluation:
Develop a graphical interface that displays the inventory in a user-friendly format. Additionally, incorporate robust cross-validation and detailed performance metrics (e.g., per-class mAP) to further refine the system.

---
## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
## License
This project is licensed under the MIT License.
