import cv2
import numpy as np
from matplotlib import pyplot as plt

before_image_path = 'user 66_R_1.png'
before_image = cv2.imread(before_image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Enhance contrast using histogram equalization
enhanced_image = cv2.equalizeHist(before_image)

# Step 2: Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

# Step 3: Use adaptive thresholding to highlight vein patterns
vein_pattern = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
)

# Step 4: Morphological operations to enhance vein patterns
kernel = np.ones((3, 3), np.uint8)
vein_cleaned = cv2.morphologyEx(vein_pattern, cv2.MORPH_OPEN, kernel)
vein_dilated = cv2.dilate(vein_cleaned, kernel, iterations=1)

# Step 5: Colorize the resulting pattern using a colormap (for similar visual effect as "after" image)
colored_pattern = cv2.applyColorMap(vein_dilated, cv2.COLORMAP_MAGMA)

# Display the results: before, binary pattern, and after (colored)
plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.imshow(before_image, cmap='gray')
plt.title('Before (Original)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(vein_dilated, cmap='gray')
plt.title('Vein Pattern (Binary)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(colored_pattern)
plt.title('After (Colored)')
plt.axis('off')

plt.show()

# Save the processed images if needed
cv2.imwrite('/mnt/data/vein_pattern_binary.png', vein_dilated)
cv2.imwrite('/mnt/data/vein_pattern_colored.png', colored_pattern)