import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function to generate a synthetic medical image
def generate_image(has_tumor):
    image_size = 64
    image = np.zeros((image_size, image_size))
    
    if has_tumor:
        # Draw a circle (simulating a tumor)
        rr, cc = np.ogrid[:image_size, :image_size]
        center = (np.random.randint(16, 48), np.random.randint(16, 48))
        radius = np.random.randint(5, 15)
        circle = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2
        image[circle] = 255
        
    # Add random noise
    noise = np.random.normal(0, 25, (image_size, image_size))
    image += noise
    
    # Normalize to range 0-1
    image = np.clip(image, 0, 255) / 255.0
    return image

# Generate synthetic dataset
num_samples = 1000
images = []
labels = []

for i in range(num_samples):
    has_tumor = np.random.rand() > 0.5
    image = generate_image(has_tumor)
    images.append(image)
    labels.append(int(has_tumor))

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Display some examples
plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 6, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.show()
