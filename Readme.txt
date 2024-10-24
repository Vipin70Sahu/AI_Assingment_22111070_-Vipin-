# Grad-CAM Visualization for Synthetic Medical Images

This project demonstrates the use of Grad-CAM (Gradient-weighted Class Activation Mapping) on synthetic medical images using a pre-trained VGG16 model. 
Grad-CAM helps in visualizing the important regions of an image that the model focuses on when making predictions, specifically useful for interpretability in medical imaging.
"For Tumor diagnostics."
## Features

Synthetic Image Generation: Creates a 224x224 grayscale image with or without a simulated tumor (circle) and random noise.
Pre-trained VGG16 Model: Utilizes the VGG16 architecture from PyTorch's `torchvision` library, pre-trained on ImageNet.
Grad-CAM Implementation: Registers hooks to the final convolutional layer of the VGG16 model to calculate and visualize Grad-CAM heatmaps.
Visualization: Displays a heatmap that highlights the important areas of the image for the model’s prediction.

## Requirements

- `torch`
- `torchvision`
- `PIL` (Python Imaging Library via `Pillow`)
- `numpy`
- `matplotlib`

You can install the required dependencies using the following command:

```bash
pip install torch torchvision pillow numpy matplotlib
