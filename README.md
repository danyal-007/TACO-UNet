# TACO UNet Semantic Segmentation Implementation

This repository contains a PyTorch implementation of a **modified UNet architecture** for semantic segmentation of the TACO (Trash Annotations in Context) dataset. The implementation includes data loading, preprocessing, model definition, training, and visualisation components, designed for efficiency and clarity.

## Overview

This project aims to perform **semantic segmentation** on images from the TACO dataset, which consists of images with annotations for various types of trash and waste. The implementation uses a **UNet architecture** which has been modified to accommodate 640x640 input images. The model is trained using bounding box annotations to create segmentation masks. This implementation is memory efficient and utilizes mixed precision training for faster training.

## Key Features

*   **Custom TACO Dataset Class**: A PyTorch `Dataset` class, `TACODataset`, handles loading and preprocessing of the TACO dataset. This class is designed to work with the JSON annotation file and includes functionality for:
    *   Loading annotations and images.
    *   Filtering valid images.
    *   Normalizing bounding boxes.
    *   Resizing images with padding while maintaining aspect ratio.
    *   Providing a mapping of category IDs to names.
    *   Creating segmentation masks based on bounding box annotations.
*   **Data Loaders**: The implementation provides a function, `create_dataloaders`, to generate efficient PyTorch `DataLoader` instances for training, validation, and testing. These data loaders include:
    *   Memory-efficient settings.
    *   Custom collate function (`collate_fn`) to handle batching of images and target dictionaries.
    *   Data splitting into training, validation and testing datasets.
*   **Modified UNet Architecture**: A custom `UNet` class is defined with:
    *   Adjusted feature map sizes for 640x640 input images.
    *   Optional batch normalization and spatial dropout.
    *   Skip connections for effective feature fusion.
    *   Double convolution blocks.
    *   A bridge in the bottleneck of the UNet.
*  **Training**: The training process is implemented in the `train_model` function using:
    *   **Mixed precision training** with `torch.cuda.amp` for faster training.
    *   **Adam optimiser** with a **ReduceLROnPlateau scheduler**.
    *   **CrossEntropyLoss** as the loss function.
    *   Dynamic mask creation during training using the ground truth bounding boxes.
    *   Saves the best model based on the validation loss.
*  **Visualisation**:
    *   The `visualize_dataset_samples` function allows visualisation of dataset samples from training, validation and test sets with bounding boxes and class labels.
    *   The `visualize_segmentation` function provides visualisation of model predictions on test samples, overlaying the predicted segmentation mask on the original image and ground truth bounding boxes.
    *   Both visualisation functions include denormalisation of images before display.

## Repository Structure

. ├── data/ # Contains the TACO dataset and annotations │ └── annotations.json # TACO annotations file │ └── images/ # TACO images ├── UNet/ # Trained model checkpoint will be saved here ├── FasterRCNN.py # Contains custom dataset utils if needed ├── TACO_Unet_Segmentation.py # Main script with all logic ├── readme.md # This file

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/danyal-007/your-repo-name.git
    cd your-repo-name
    ```
2.  **Download the TACO dataset**: Download the TACO dataset and place the `annotations.json` file and `images` folder in the `data` directory as shown in the file structure above.
3.  **Run the main script**: Execute the `TACO_Unet_Segmentation.py` script to begin training. The script will also generate visualizations:
    ```bash
    python TACO_Unet_Segmentation.py
    ```

## Usage

The main script `TACO_Unet_Segmentation.py` includes all necessary steps:

1.  **Data Loading and Preparation**:
    *   The script loads the TACO dataset and splits it into training, validation, and testing subsets.
    *   It also defines data loaders with specific batch sizes and configurations.
    *   The script prints information on the dataset splits, batch sizes, and number of workers.
2.  **Dataset Visualisation**:
    *   The script then visualises samples from the train, validation, and test sets using the `visualize_dataset_samples` function. This function will plot the image samples with bounding boxes and class labels.
3.  **Model Initialization and Training**:
    *   The script initialises a UNet model with the appropriate number of classes, based on the TACO dataset.
    *   The model is then trained using the `train_model` function with mixed precision training.
    *   During training, the best model, based on the validation loss, is saved.
4.  **Model Loading**:
    *   After training, the best trained model from the saved checkpoint is loaded.
5.  **Segmentation Visualisation**:
    *   The script generates segmentation masks using the `visualize_segmentation` function, displaying predicted segmentation masks with bounding boxes on test samples.
    *   The segmentation masks are overlayed on the original images and the class labels and bounding boxes are also shown.

## Code Details

### Data Loading and Preprocessing

*   The `TACODataset` class inherits from `torch.utils.data.Dataset` and is designed to work with the TACO dataset's JSON annotation file.
*   The `_prepare_dataset` method loads and filters the dataset based on the availability of the images and annotations.
*   The `_normalize_boxes` method normalizes the coordinates of the bounding boxes to a range between 0 and 1.
*   The `_resize_image_with_padding` method resizes images while maintaining the aspect ratio using padding.
*   The `__getitem__` method loads an image, resizes the image and bounding boxes, and converts them to the appropriate PyTorch tensors. The segmentation masks are generated during the training phase itself.
*   The `create_dataloaders` function sets up the PyTorch DataLoaders with random samplers for training, validation, and test sets with memory-efficient configurations. The `collate_fn` ensures that the data is batched correctly for training.

### Model Architecture

*   The `DoubleConv` class defines a double convolution block with optional batch normalization and spatial dropout. This is used as a fundamental building block in the UNet architecture.
*   The `UNet` class defines the full UNet architecture with encoding, decoding, and bridge layers. This modified architecture is designed to work with 640x640 input images and has skip connections.

### Training

*   The `train_model` function implements the training loop with mixed precision training, dynamic mask creation, and saving the best model checkpoint.
*   This function utilizes a mixed-precision training approach, which employs both single-precision (float32) and half-precision (float16) floating-point numbers to speed up the training process and reduce memory usage without significantly sacrificing accuracy.
*   The `create_segmentation_masks` function converts bounding box coordinates to segmentation masks. This function creates masks of the bounding box annotations scaled directly to 640x640.
*   During training, masks are created dynamically from the ground truth bounding boxes for each batch. This process is memory efficient as the segmentation masks are not stored prior to training.
*   The training loop calculates the loss using the `CrossEntropyLoss` criterion and employs the Adam optimizer. The learning rate is dynamically adjusted using a ReduceLROnPlateau scheduler.
*   The training loop incorporates validation after each epoch and saves the model with the best validation loss.

### Visualisation

*   The `visualize_dataset_samples` function plots sample images from the training, validation, and test datasets with their corresponding bounding boxes and class labels.
*   The `visualize_segmentation` function plots sample images from the test dataset, the predicted segmentation mask, and the original bounding boxes.

### Utility Functions

*   The `clear_gpu_memory` function clears GPU memory after usage using `torch.cuda.empty_cache` and garbage collection.
*   The `load_best_model` function loads the best saved model from the model checkpoint based on its validation performance and returns the model loaded to the specified device.

## Memory Management

*   The code utilizes memory-efficient practices, including `pin_memory=True` and `non_blocking=True` in data loading, dynamic mask creation during training, and mixed precision training.
*   The code also makes use of garbage collection and emptying the CUDA cache to prevent out of memory errors.

## Future Work

*   Add evaluation metrics (mIOU, Dice score) for a quantitative evaluation.
*   Implement data augmentations to improve model robustness.
*   Implement a more sophisticated segmentation mask generation approach using a different loss function.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
