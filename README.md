# 🚀 Automated Computer Vision Data Pipeline (ETL)

An end-to-end data engineering and MLOps pipeline designed to ingest, standardize, and orchestrate raw multi-resolution camera datasets for deep learning model training inside Google Colab.

## 📋 Project Overview
In real-world computer vision applications (such as automated manufacturing inspection, smart traffic cameras, or agricultural monitoring), raw image feeds arrive in varied resolutions, dimensions, and color formats. Feeding messy data directly into deep learning models leads to poor performance or training instability. 

This project implements an **automated ETL (Extract, Transform, Load) pipeline** using **Prefect**, **OpenCV**, and **TensorFlow** to clean, scale, and optimize data seamlessly.

---

## 🏗️ Pipeline Architecture & Workflow

[ Raw Camera Feed / Storage ] 
            │
            ▼  (Extract Stage: Glob path scanning)
   [ Extract Task ]
            │
            ▼  (Transform Stage: OpenCV Processing)
   [ OpenCV Preprocessing ] 
   ├── BGR to RGB Color Conversion
   ├── Uniform Resizing (224x224)
   └── Min-Max Normalization [0.0, 1.0]
            │
            ▼  (Load Stage: tf.data Optimization)
   [ TensorFlow tf.data.Dataset ]
   ├── Batching (Batch Size = 4)
   └── GPU Prefetching (AUTOTUNE)
            │
            ▼
[ Downstream CNN / Deep Learning Training ]

---

## 🛠️ Tech Stack & Libraries
* **Python**: Core programming language.
* **Prefect**: Modern workflow orchestration for task automation and error handling (retries).
* **OpenCV (`cv2`)**: Image manipulation, color correction, and spatial resizing.
* **TensorFlow / Keras**: High-performance dataset optimization (`tf.data`) for model ingestion.
* **NumPy**: Fast array manipulation and pixel scaling.

---

## 🚀 Key Features & Implementation
1. **Automated Extraction (`@task`)**: Dynamically scans input directories to discover multi-format (`.jpg`, `.png`) camera images.
2. **Robust Transformation (`@task`)**: 
   - Converts standard OpenCV `BGR` matrices to standard `RGB` format.
   - Resizes all incoming images to a strict $224 \times 224$ matrix (matching standard CNN backbones like ResNet and MobileNet).
   - Scales pixel values from integers $[0, 255]$ to floating-point range $[0.0, 1.0]$ via **Min-Max Normalization** to eliminate exploding gradients during training.
3. **Optimized Loading (`@task`)**: Converts NumPy arrays into a native `tf.data.Dataset`, leveraging `batching` and `prefetch(AUTOTUNE)` to minimize CPU-GPU latency bottlenecks.
4. **Workflow Orchestration (`@flow`)**: Uses Prefect decorators to execute sequential steps with built-in retry mechanisms and logging.

---

## 💻 How to Run in Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and paste your pipeline code.
3. Run the cells sequentially. The pipeline automatically generates sample camera feeds and verifies output tensor shapes and pixel ranges.

---

## 📊 Verification Summary
When executed, the pipeline outputs verified tensor metrics:
* **Batch Tensor Shape**: `(4, 224, 224, 3)`
* **Pixel Value Range**: `0.00` to `1.00`
* **Data Type**: `float32`
