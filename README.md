
# AIRL Internship Coding Assignment

This repository contains the solutions for the AI for Robotics Research Lab (AIRL) internship coding assignment submitted on September 30, 2025.

* **Q1: Vision Transformer on CIFAR-10** (`q1.ipynb`)
* **Q2: Text-Driven Image Segmentation with SAM 2** (`q2.ipynb`)

---
## Q1: Vision Transformer on CIFAR-10

This solution implements a Vision Transformer (ViT) from scratch in PyTorch and trains it on the CIFAR-10 dataset to achieve the highest possible test accuracy.

### How to Run in Colab

1.  Open `q1.ipynb` in Google Colab.
2.  Ensure the runtime is set to a GPU (e.g., `T4 GPU`) by navigating to `Runtime` -> `Change runtime type`.
3.  Run all cells from top to bottom. The script will download the CIFAR-10 dataset, build the model, and run the training process. The final best accuracy will be printed at the end. [cite: 16]

### Best Model Configuration

The following configuration achieved the best test accuracy. It includes several optimization and regularization techniques.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `patch_size` | 4 | Size of image patches. |
| `embed_dim` | 512 | Embedding dimension for each patch. |
| `depth` | 6 | Number of Transformer Encoder blocks. |
| `heads` | 8 | Number of heads in Multi-Head Self-Attention. |
| `mlp_dim` | 1024 | Hidden dimension in the MLP blocks. |
| `epochs` | 100 | Total training epochs. |
| `batch_size` | 256 | Number of samples per batch. |
| `learning_rate` | 1e-3 | Max learning rate for the scheduler. |
| `optimizer` | AdamW | Optimizer with corrected weight decay. |
| `scheduler` | OneCycleLR | Scheduler with warmup and cosine decay. |
| `label_smoothing` | 0.1 | Regularization to prevent overconfidence. |
| `stochastic_depth`| 0.1 | Regularization that randomly drops residual blocks. |
| `mixup_alpha`| 0.4 | Augmentation that blends image pairs. |
| `tta` | True | Test-Time Augmentation (Horizontal Flip). |

### Results

| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **88.37%** |

*(Note: This was the best accuracy achieved during the final run.  The result may vary slightly on different runs due to the stochastic nature of training.)*

### Bonus Analysis: Performance Improvements

To maximize the ViT's performance, several key techniques were implemented:

* **Speed and Stability**:
    * **Mixed Precision Training**: Using `torch.amp` (`autocast` and `GradScaler`) provided a significant speedup (~1.5-2x) by using `float16` for computations.
    * **`torch.compile`**: JIT compilation in PyTorch 2.0+ offered an additional performance boost on the Colab GPU.

* **Accuracy and Regularization**:
    * **Strong Augmentations**: `RandAugment` and `RandomErasing` (Cutout) were used to create a more robust model and prevent overfitting on the small CIFAR-10 dataset.
    * **Advanced Regularization**: **Stochastic Depth** (DropPath) and **Label Smoothing** were critical in improving generalization and preventing the model from becoming overconfident.
    * **Mixup Augmentation**: Training on linear combinations of image pairs forced the model to learn smoother decision boundaries, providing a significant accuracy gain.
    * **Test-Time Augmentation (TTA)**: Averaging predictions on original and horizontally flipped test images provided a final, easy boost to the score.
    * **Training Recipe**: Using the **AdamW** optimizer with a **OneCycleLR** scheduler (which includes a warmup phase) was crucial for stable and effective training of the transformer architecture.

---
## Q2: Text-Driven Image Segmentation with SAM 2

This solution creates a pipeline that takes a text prompt and an image, and outputs a segmentation mask for the object described in the prompt using SAM 2.

### Pipeline Overview

The pipeline uses two main models from the `ultralytics` library:

1.  **YOLO-World**: A powerful zero-shot detection model that takes the text prompt (e.g., "a dog," "dog collar") and identifies the corresponding object's location in the image by generating a bounding box.
2.  **SAM (Segment Anything Model)**: This model takes the bounding box from YOLO-World as a prompt and generates a high-quality, precise segmentation mask for the object within that box.

### How to Run in Colab

1.  Open `q2.ipynb` in Google Colab.
2.  Ensure the runtime is set to a GPU.
3.  Run all cells from top to bottom. The script will install the `ultralytics` library and download the required model weights automatically on the first run.
4.  You can modify the `TEXT_PROMPT` and `IMAGE_PATH` variables in the final cell to run the pipeline on your own images and prompts.

### Limitations

* **Fine-Grained Details**: The YOLO-World detector can struggle with prompts for very small, fine-grained, or low-contrast objects (e.g., "dog collar" in the sample image). This is a common limitation of zero-shot models.
* Lowering the confidence threshold can sometimes help but may also increase false positives.
* **Text Ambiguity**: The model's performance depends on its ability to interpret the text prompt.Highly abstract or ambiguous prompts may not yield accurate detections.
* **Single Object Focus**: The current implementation is designed to segment the single most confident detection. It may not perform well on prompts that describe multiple, distinct objects.
