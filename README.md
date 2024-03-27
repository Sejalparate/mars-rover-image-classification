# Mars Rover Image Classification

## Overview
This project aims to classify images captured by NASA's Curiosity rover on Mars using deep learning Convolutional Neural Networks (CNN). The frontend interface is developed using Streamlit, providing an intuitive user experience for interacting with the classification model.

## Dataset
The dataset used in this project is provided by NASA and consists of images captured by the Curiosity rover during its exploration of Mars. The dataset contains images from various angles, lighting conditions, and terrains on the Martian surface. Each image is labeled with a corresponding category based on the content or features present in the image.
<p>Original dataset link: https://www.kaggle.com/datasets/brsdincer/mars-surface-and-curiosity-image-set-nasa</p>

1. **Modify dataset**: To utilize the modified dataset, the images have been categorized into their respective classes based on CSV files containing the labels for each corresponding image. To split the dataset accordingly, execute the split_dataset.py file. Subsequently, merge all the images from the test, train, and validation sets into a single class directory, labeling each directory with a unique class identifier such as 0, 1, 2, etc. (Few images have been deleted to avoid data imbalances in hampering the model performance).<br><br>
  The updated structure:
  ```
  mars-rover-image-classification/
  |
  ├── Model_training_dataset_split/
  │   ├── 0/
  │   ├── 1/
  │   ├── 2/
  │   │...
  ```

2. **Using original dataset**: For those preferring the unaltered dataset, access the files within the "Model_training_without_dataset_split" directory.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Sejalparate/mars-rover-image-classification.git
   ```
2. Download the dataset from the provided NASA link and place it in the appropriate directory.
3. Preprocess the data, train the model, and evaluate its performance using the provided scripts.
4. Make predictions on new images using the trained model.
5. After training the model, start the Streamlit app:
    ```
    streamlit run app.py
    ```
6. Access the Streamlit interface in your web browser.
7. Upload an image captured by the Curiosity rover.
8. The app will display the predicted class for the uploaded image.

## Results
The model achieves 90% accuracy on the test set, indicating its capability to classify Martian surface images effectively. The detailed evaluation metrics and visualizations of model performance can be found in the project's documentation.
