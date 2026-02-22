import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------
# Config
# -------------------
IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# -------------------
# Helper function
# -------------------
def load_images(folder_path, label):
    images = []
    labels = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue  # skip corrupted images

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        images.append(image)
        labels.append(label)

    return images, labels

# -------------------
# Main pipeline
# -------------------
def main():
    cats_path = os.path.join(RAW_DATA_DIR, "cats")
    dogs_path = os.path.join(RAW_DATA_DIR, "dogs")

    print("Loading cat images...")
    cat_images, cat_labels = load_images(cats_path, 0)

    print("Loading dog images...")
    dog_images, dog_labels = load_images(dogs_path, 1)

    X = np.array(cat_images + dog_images)
    y = np.array(cat_labels + dog_labels)

    print(f"Total samples: {len(X)}")

    # 80 / 10 / 10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)

    print("Preprocessing completed successfully.")
    print(f"Processed data saved to: {PROCESSED_DATA_DIR}")

# -------------------
# Entry point
# -------------------
if __name__ == "__main__":
    main()
