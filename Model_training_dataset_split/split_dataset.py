import os
import pandas as pd
import shutil

# Distributes images into class folders based on a CSV file
def distribute_images_from_csv(csv_file, dataset_folder):
    df = pd.read_csv(csv_file)
    class_label_column = 'LABELS'

    # Create class folders
    for class_label in df[class_label_column].unique():
        class_folder = os.path.join(dataset_folder, str(class_label))
        os.makedirs(class_folder, exist_ok=True)  # Create if not exists

    # Process each image-label pair
    for index, row in df.iterrows():
        image_filename = os.path.basename(row['JPG'])  # Extract filename without path
        class_label = row[class_label_column]
        class_label_str = str(class_label)
        image_path = os.path.join(dataset_folder, image_filename)
        destination = os.path.join(dataset_folder, class_label_str, image_filename)

        try:
            shutil.move(image_path, destination)
        except FileNotFoundError:
            print(f"Warning: Image '{image_filename}' not found. Skipping.")

# Replace placeholders with your actual paths
# Do the same for test and val datasets
csv_file = 'Mars Surface and Curiosity Image dataset\Train_CSV.csv'
dataset_folder = 'Mars Surface and Curiosity Image dataset\images'
distribute_images_from_csv(csv_file, dataset_folder)