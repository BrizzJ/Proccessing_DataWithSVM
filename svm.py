import os
import numpy as np
import pandas as pd
import joblib
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import joblib
from skimage.io import imread
from skimage.transform import resize, rotate, rescale
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy.ndimage import shift
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------
# Data Augmentation
# ------------------------------

def augment_image(image, num_augments=3, image_size=(32, 32)):
    """Apply random augmentations to an image and return a list of augmented versions."""
    augmented = []

    for _ in range(num_augments):
        aug = image.copy()

        # Random rotation (-15° to +15°)
        angle = np.random.uniform(-15, 15)
        aug = rotate(aug, angle, mode="edge")

        # Random shift (up to 3 pixels in x/y)
        shift_x = np.random.uniform(-3, 3)
        shift_y = np.random.uniform(-3, 3)
        aug = shift(aug, shift=(shift_y, shift_x), mode="nearest")

        # Random zoom (scale 0.9–1.1)
        scale = np.random.uniform(0.9, 1.1)
        aug = rescale(aug, scale, mode="edge", channel_axis=None, anti_aliasing=True)
        # Pad/crop back to original size
        if aug.shape[0] > image_size[0]:
            aug = aug[:image_size[0], :image_size[1]]
        else:
            pad_y = image_size[0] - aug.shape[0]
            pad_x = image_size[1] - aug.shape[1]
            aug = np.pad(aug, ((0, pad_y), (0, pad_x)), mode="edge")

        # Random brightness adjustment (0.8–1.2)
        factor = np.random.uniform(0.8, 1.2)
        aug = np.clip(aug * factor, 0, 1)

        # Random blur (10% chance)
        if np.random.rand() < 0.1:
            aug = gaussian(aug, sigma=np.random.uniform(0.5, 1.0))

        # Random noise (10% chance)
        if np.random.rand() < 0.1:
            aug = random_noise(aug, mode="gaussian", var=0.01)

        # Resize back to desired size
        aug = resize(aug, image_size)
        augmented.append(aug)

    return augmented


# ------------------------------
# Feature extraction from CSV
# ------------------------------

def extract_features_from_csv(csv_file, root_folder, image_size=(32, 32), is_test=False, augment=False):
    """Extracts HOG features (with optional augmentation) from images listed in a CSV file."""

    features, labels = [], []
    df = pd.read_csv(csv_file)

    file_col = 'Path' if 'Path' in df.columns else 'Filename'
    label_col = 'ClassId' if 'ClassId' in df.columns else None

    if is_test:
        print(f"Using CSV labels from {csv_file} (mapping Test/ → test_control/)")
    else:
        print(f"Using CSV labels from {csv_file}")

    for _, row in df.iterrows():
        path = row[file_col]

        if is_test:
            path = path.replace("Test/", "test_control/")

        img_path = os.path.join(root_folder, *path.split('/'))

        if not os.path.exists(img_path):
            print(f"Skipped {img_path}: File does not exist")
            continue

        try:
            image = imread(img_path, as_gray=True)
            image = resize(image, image_size)

            # HOG for original image
            feature = hog(image, pixels_per_cell=(8, 8))
            features.append(feature)
            if label_col:
                labels.append(int(row[label_col]))

            # Augmentation (only for training)
            if augment and not is_test:
                for aug in augment_image(image, image_size=image_size):
                    aug_feature = hog(aug, pixels_per_cell=(8, 8))
                    features.append(aug_feature)
                    if label_col:
                        labels.append(int(row[label_col]))

        except Exception as e:
            print(f"Skipped {img_path}: {e}")

    X = np.array(features)
    y = np.array(labels) if labels else None
    return X, y


# ------------------------------
# Train SVM
# ------------------------------

def train_svm(X_train, y_train, save_path="svm_model.pkl"):
    """Trains a linear SVM classifier on the training data."""
    print("Training SVM...")
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_train, y_train)
    joblib.dump(clf, save_path)
    print(f"Model saved at {save_path}")
    return clf


# ------------------------------
# Evaluate SVM
# ------------------------------

def evaluate_svm(clf, X_test, y_test, output_folder):
    """Evaluates the trained SVM on the test set and saves predictions."""
    if X_test.size == 0 or y_test is None:
        print("Test set is empty. Cannot evaluate SVM.")
        return

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    np.save(os.path.join(output_folder, "predictions.npy"), y_pred)
    print(f"Predictions saved in {output_folder}/predictions.npy")


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    data_root = "data"
    output_root = os.path.join(os.getcwd(), "svmOutput")
    os.makedirs(output_root, exist_ok=True)

    train_csv = os.path.join(data_root, "Train.csv")
    test_csv = os.path.join(data_root, "Test.csv")

    # Train set with augmentation
    X_train, y_train = extract_features_from_csv(train_csv, data_root, augment=True)
    np.save(os.path.join(output_root, "features_train.npy"), X_train)
    np.save(os.path.join(output_root, "labels_train.npy"), y_train)

    # Test set (no augmentation)
    X_test, y_test = extract_features_from_csv(test_csv, data_root, is_test=True, augment=False)
    np.save(os.path.join(output_root, "features_test.npy"), X_test)
    if y_test is not None:
        np.save(os.path.join(output_root, "labels_test.npy"), y_test)

    print(f"Training features shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}, labels shape: {None if y_test is None else y_test.shape}")

    # Train & evaluate
    model = train_svm(X_train, y_train, save_path=os.path.join(output_root, "svm_model.pkl"))
    evaluate_svm(model, X_test, y_test, output_folder=output_root)
