# TACO UNet Semantic Segmentation Implementation

This repository contains a PyTorch implementation of a **modified UNet architecture** for semantic segmentation of the TACO (Trash Annotations in Context) dataset. The implementation includes data loading, preprocessing, model definition, training, and visualisation components, designed for efficiency and clarity.

## Overview

This project aims to perform **semantic segmentation** on images from the TACO dataset, which consists of images with annotations for various types of trash and waste [1, 2]. The implementation uses a **UNet architecture** which has been modified to accommodate 640x640 input images [3]. The model is trained using bounding box annotations to create segmentation masks [4]. This implementation is memory efficient and utilizes mixed precision training for faster training [5].

## Key Features

*   **Custom TACO Dataset Class**:  A PyTorch `Dataset` class, `TACODataset`, handles loading and preprocessing of the TACO dataset [1]. This class is designed to work with the JSON annotation file and includes functionality for [2]:
    *   Loading annotations and images.
    *   Filtering valid images.
    *   Normalizing bounding boxes [6, 7].
    *   Resizing images with padding while maintaining aspect ratio [7].
    *   Providing a mapping of category IDs to names [6].
    *   Creating segmentation masks based on bounding box annotations [4, 8].
*   **Data Loaders**: The implementation provides a function, `create_dataloaders`, to generate efficient PyTorch `DataLoader` instances for training, validation, and testing [9]. These data loaders include [9]:
    *   Memory-efficient settings.
    *   Custom collate function (`collate_fn`) to handle batching of images and target dictionaries [10].
    *   Data splitting into training, validation and testing datasets [9].
*   **Modified UNet Architecture**: A custom `UNet` class is defined with [3]:
    *   Adjusted feature map sizes for 640x640 input images.
    *   Optional batch normalization and spatial dropout.
    *   Skip connections for effective feature fusion [11].
    *   Double convolution blocks [12].
    *   A bridge in the bottleneck of the UNet [3].
*  **Training**: The training process is implemented in the `train_model` function using [5]:
    *   **Mixed precision training** with `torch.cuda.amp` for faster training [5].
    *   **Adam optimiser** with a **ReduceLROnPlateau scheduler** [5].
    *   **CrossEntropyLoss** as the loss function [5].
    *   Dynamic mask creation during training using the ground truth bounding boxes [5].
    *   Saves the best model based on the validation loss [13].
*  **Visualisation**:
    *   The `visualize_dataset_samples` function allows visualisation of dataset samples from training, validation and test sets with bounding boxes and class labels [14, 15].
    *   The `visualize_segmentation` function provides visualisation of model predictions on test samples, overlaying the predicted segmentation mask on the original image and ground truth bounding boxes [16, 17].
    *   Both visualisation functions include denormalisation of images before display [14].

## Repository Structure

. ├── data/ # Contains the TACO dataset and annotations │ └── annotations.json # TACO annotations file │ └── images/ # TACO images ├── UNet/ # Trained model checkpoint will be saved here ├── FasterRCNN.py # Contains custom dataset utils if needed ├── TACO_Unet_Segmentation.py # Main script with all logic ├── readme.md # This file └── requirements.txt # Required python packages

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/danyal-007/your-repo-name.git
    cd your-repo-name
    ```
2.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the TACO dataset**: Download the TACO dataset and place the `annotations.json` file and `images` folder in the `data` directory as shown in the file structure above.
4.  **Run the main script**: Execute the `TACO_Unet_Segmentation.py` script to begin training. The script will also generate visualizations [18-20]:
    ```bash
    python TACO_Unet_Segmentation.py
    ```

## Usage

The main script `TACO_Unet_Segmentation.py` includes all necessary steps:

1.  **Data Loading and Preparation**:
    *   The script loads the TACO dataset and splits it into training, validation, and testing subsets [9, 18].
    *   It also defines data loaders with specific batch sizes and configurations [9, 18].
    *   The script prints information on the dataset splits, batch sizes, and number of workers [18, 19, 21].
2.  **Dataset Visualisation**:
    *   The script then visualises samples from the train, validation, and test sets using the `visualize_dataset_samples` function. This function will plot the image samples with bounding boxes and class labels [14, 19].
3.  **Model Initialization and Training**:
    *   The script initialises a UNet model with the appropriate number of classes, based on the TACO dataset [20, 22].
    *   The model is then trained using the `train_model` function with mixed precision training [5, 22].
    *   During training, the best model, based on the validation loss, is saved [13].
4.  **Model Loading**:
    *   After training, the best trained model from the saved checkpoint is loaded [20, 23].
5.  **Segmentation Visualisation**:
    *   The script generates segmentation masks using the `visualize_segmentation` function, displaying predicted segmentation masks with bounding boxes on test samples [16, 20].
    *   The segmentation masks are overlayed on the original images and the class labels and bounding boxes are also shown [17].

## Code Details

### Data Loading and Preprocessing

*   The `TACODataset` class inherits from `torch.utils.data.Dataset` and is designed to work with the TACO dataset's JSON annotation file [1].
*  The `_prepare_dataset` method loads and filters the dataset based on the availability of the images and annotations [2].
*  The `_normalize_boxes` method normalizes the coordinates of the bounding boxes to a range between 0 and 1 [6].
* The `_resize_image_with_padding` method resizes images while maintaining the aspect ratio using padding [7].
* The `__getitem__` method loads an image, resizes the image and bounding boxes, and converts them to the appropriate PyTorch tensors [24]. The segmentation masks are generated during the training phase itself [5].
*   The `create_dataloaders` function sets up the PyTorch DataLoaders with random samplers for training, validation, and test sets with memory-efficient configurations [9]. The `collate_fn` ensures that the data is batched correctly for training [10].

### Model Architecture

*   The `DoubleConv` class defines a double convolution block with optional batch normalization and spatial dropout [12]. This is used as a fundamental building block in the UNet architecture.
* The `UNet` class defines the full UNet architecture with encoding, decoding, and bridge layers [3]. This modified architecture is designed to work with 640x640 input images and has skip connections [3, 11].

### Training

*   The `train_model` function implements the training loop with mixed precision training, dynamic mask creation, and saving the best model checkpoint [5].
*   This function utilizes a mixed-precision training approach, which employs both single-precision (float32) and half-precision (float16) floating-point numbers to speed up the training process and reduce memory usage without significantly sacrificing accuracy [5].
*   The `create_segmentation_masks` function converts bounding box coordinates to segmentation masks [4]. This function creates masks of the bounding box annotations scaled directly to 640x640 [4].
*   During training, masks are created dynamically from the ground truth bounding boxes for each batch. This process is memory efficient as the segmentation masks are not stored prior to training [5].
*  The training loop calculates the loss using the `CrossEntropyLoss` criterion and employs the Adam optimizer [5]. The learning rate is dynamically adjusted using a ReduceLROnPlateau scheduler [5].
*   The training loop incorporates validation after each epoch and saves the model with the best validation loss [13].

### Visualisation

*   The `visualize_dataset_samples` function plots sample images from the training, validation, and test datasets with their corresponding bounding boxes and class labels [14, 15].
*   The `visualize_segmentation` function plots sample images from the test dataset, the predicted segmentation mask, and the original bounding boxes [16, 17].

### Utility Functions

*   The `clear_gpu_memory` function clears GPU memory after usage using `torch.cuda.empty_cache` and garbage collection [25].
*   The `load_best_model` function loads the best saved model from the model checkpoint based on its validation performance and returns the model loaded to the specified device [23].

## Memory Management

*   The code utilizes memory-efficient practices, including `pin_memory=True` and `non_blocking=True` in data loading, dynamic mask creation during training, and mixed precision training [5, 9].
*   The code also makes use of garbage collection and emptying the CUDA cache to prevent out of memory errors [25].

## Citation

If you use this implementation in your research, please cite this repository using the following format:

Danyal Kailani. (2024). TACO UNet Semantic Segmentation Implementation. https://github.com/danyal-007/TACO-UNet

## Future Work

*   Add evaluation metrics (mIOU, Dice score) for a quantitative evaluation.
*   Implement data augmentations to improve model robustness.
*   Implement a more sophisticated segmentation mask generation approach using a different loss function.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
