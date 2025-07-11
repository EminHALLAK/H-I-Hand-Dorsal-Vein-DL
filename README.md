# Human Identification Based on Hand Dorsal Vein Pattern Images

## Project Overview
This project implements a biometric identification system using hand dorsal vein patterns. The system uses deep learning to recognize and authenticate individuals based on the unique vascular patterns on the back of their hands.

## Dataset
The project uses a dataset of hand dorsal vein images organized in folders by subject identity. The images are:
- Preprocessed to 250x250 pixel resolution
- Normalized to improve training stability

## Model Architecture
We implemented a Siamese Neural Network architecture, which excels at similarity-based recognition tasks:

1. **Base Model**: Custom CNN with the following structure:
   - Five convolutional blocks with increasing filters (64→128→256→512→512)
   - Each block includes: Conv2D → BatchNormalization → MaxPooling → Dropout
   - Final layers include GlobalAveragePooling and dense embedding layers (256→128)

2. **Siamese Network**:
   - Twin networks with shared weights process pairs of images
   - Manhattan distance (L1) measures similarity between embeddings
   - Final sigmoid layer outputs probability of match

## Training Methodology
- **Pair Generation**: Created positive pairs (same person) and negative pairs (different people)
- **Data Augmentation**: Applied rotation, shift, shear, zoom, and flip transformations
- **Optimization**:
  - Binary cross-entropy loss
  - Adam optimizer with exponential learning rate decay
  - Initial learning rate: 0.001
  - Batch size: 6
  - Epochs: 100
- **Validation**: 20% of data reserved for validation with stratified sampling

## Results
- **Final Validation Accuracy**: 89.58%
- **Training Pattern**: 
  - Validation accuracy consistently higher than training accuracy
  - Validation loss decreased steadily throughout training
  - Model successfully learned to differentiate between individuals

## Usage
1. Prepare input images to 250x250 resolution
2. Use the model to generate embeddings for each image
3. Compare embeddings using L1 distance
4. Apply threshold to determine if images belong to the same individual

## Dependencies
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
