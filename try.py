import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define function to generate a synthetic medical image
def generate_image(has_tumor):
    image_size = 224
    image = np.zeros((image_size, image_size))
    
    if has_tumor:
        # Draw a circle (simulating a tumor)
        rr, cc = np.ogrid[:image_size, :image_size]
        center = (np.random.randint(16, 208), np.random.randint(16, 208))
        radius = np.random.randint(5, 30)
        circle = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2
        image[circle] = 255
        
    # Add random noise
    noise = np.random.normal(0, 25, (image_size, image_size))
    image += noise
    
    # Normalize to range 0-1
    image = np.clip(image, 0, 255) / 255.0
    return image

# Generate a synthetic image for Grad-CAM
has_tumor = True  # or False
img = generate_image(has_tumor)
img = (img * 255).astype(np.uint8)  # Convert back to 0-255 range for PIL

# Convert to PIL Image
img_pil = Image.fromarray(img)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(img_pil).unsqueeze(0)

# Load the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Define hooks to get gradients and activations
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks for the last convolutional layer
        last_conv_layer = self.model.features[-1]
        self.hook_handles.append(last_conv_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(last_conv_layer.register_backward_hook(backward_hook))

    def get_grad_cam(self, class_idx):
        # Forward pass
        self.model.zero_grad()
        output = self.model(img_tensor)
        target = output[0][class_idx]
        target.backward()

        # Pool the gradients across the channels
        gradients = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0].detach()

        # Compute the Grad-CAM heatmap
        heatmap = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i in range(gradients.size(0)):
            heatmap += gradients[i] * activations[i]
        heatmap = F.relu(heatmap)
        heatmap /= heatmap.max()

        return heatmap.numpy()

# Apply Grad-CAM to the synthetic image
grad_cam = GradCAM(model)
class_idx = torch.argmax(model(img_tensor)[0]).item()
heatmap = grad_cam.get_grad_cam(class_idx)

# Display the heatmap
plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.show()
