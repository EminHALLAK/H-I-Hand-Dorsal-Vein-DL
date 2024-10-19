from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the uploaded image
R_hand_image_path1 = 'Ignored_Data\\Dorsal Hand Dataset\\user 14_R_1.png'
L_hand_image_path1 = 'Ignored_Data\\Dorsal Hand Dataset\\user 14_L_1.png'

def Process_Image(image_path):
    # Görüntüyü açın
    image = Image.open(image_path)

    # Görüntüyü gri tonlamalıya çevirin
    image_gray = np.array(image.convert('L'))

    # CLAHE ile kontrast artırma (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_gray)

    # Morfolojik dönüşüm uygulama (Closing)
    kernel = np.ones((5, 5), np.uint8)
    image_morph = cv2.morphologyEx(image_clahe, cv2.MORPH_CLOSE, kernel)

    # Kenar algılama (Canny Edge Detection)
    edges_enhanced = cv2.Canny(image_morph, 50, 150)

    # Sonuçları gösterme
    plt.figure(figsize=(14, 7))

    # Orijinal gri tonlamalı görüntü
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    # Morfolojik dönüşüm uygulanmış görüntü
    plt.subplot(1, 3, 2)
    plt.imshow(image_morph, cmap='gray')
    plt.title('Morphological Transformation')
    plt.axis('off')

    # Kenar algılama sonrası görüntü
    plt.subplot(1, 3, 3)
    plt.imshow(edges_enhanced, cmap='gray')
    plt.title('Edge Detection (Enhanced)')
    plt.axis('off')

    plt.show()


def Process_Image2(image):
    
    image = Image.open(image)
    
    # Görüntüyü gri tonlamalıya çevirin
    image_gray = np.array(image.convert('L'))

    # CLAHE ile kontrast artırma (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25, 25))
    image_clahe = clahe.apply(image_gray)

    # Morfolojik dönüşüm uygulama (Closing)
    kernel = np.ones((5, 5), np.uint8)
    image_morph = cv2.morphologyEx(image_clahe, cv2.MORPH_CLOSE, kernel)

    # Kenar algılama (Canny Edge Detection)
    edges_enhanced = cv2.Canny(image_morph, 100, 110)

    # Sonuçları gösterme
    plt.figure(figsize=(14, 7))

    # Orijinal gri tonlamalı görüntü
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    # Morfolojik dönüşüm uygulanmış görüntü
    plt.subplot(1, 3, 2)
    plt.imshow(image_morph, cmap='gray')
    plt.title('Morphological Transformation')
    plt.axis('off')

    # Kenar algılama sonrası görüntü
    plt.subplot(1, 3, 3)
    plt.imshow(edges_enhanced, cmap='gray')
    plt.title('Edge Detection (Enhanced)')
    plt.axis('off')

    plt.show()


Process_Image(R_hand_image_path1)
Process_Image2(R_hand_image_path1)