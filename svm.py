import os
import numpy as np
import pandas as pd
import joblib
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------
# Feature extraction from CSV
# ------------------------------

def extract_features_from_csv(csv_file, root_folder, image_size=(32, 32), is_test=False):
    
    # Extracts HOG features from images listed in a CSV file

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

        # Map Test/ prefix to test_control folder
        if is_test:
            path = path.replace("Test/", "test_control/")

        # Convert CSV path to OS path
        img_path = os.path.join(root_folder, *path.split('/'))

        if not os.path.exists(img_path):
            print(f"Skipped {img_path}: File does not exist")
            continue

        try:
            image = imread(img_path, as_gray=True)
            image = resize(image, image_size)
            feature = hog(image, pixels_per_cell=(8, 8))
            features.append(feature)
            if label_col:
                labels.append(int(row[label_col]))
        except Exception as e:
            print(f"Skipped {img_path}: {e}")

    X = np.array(features)
    y = np.array(labels) if labels else None
    
    # Returns:
    # X: NumPy array of shape (num_samples, num_features) with image features
    # y: NumPy array of labels, or None if test labels are not available
    return X, y


# ------------------------------
# Train SVM
# ------------------------------

def train_svm(X_train, y_train, save_path="svm_model.pkl"):
    
    # Trains a linear SVM classifier on the training data

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
    
    # Evaluates the trained SVM on the test set and saves predictions

    if X_test.size == 0 or y_test is None:
        print("Test set is empty. Cannot evaluate SVM.")
        return

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save predictions in the output folder
    np.save(os.path.join(output_folder, "predictions.npy"), y_pred)
    print(f"Predictions saved in {output_folder}/predictions.npy")


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    
    # Set up paths
    data_root = "data"
    output_root = os.path.join(os.getcwd(), "svmOutput")  # folder in current directory
    os.makedirs(output_root, exist_ok=True)  # create if it doesn't exist

    train_csv = os.path.join(data_root, "Train.csv")
    test_csv = os.path.join(data_root, "Test.csv")

     # Extract training features and labels
    X_train, y_train = extract_features_from_csv(train_csv, data_root)
    np.save(os.path.join(output_root, "features_train.npy"), X_train)
    np.save(os.path.join(output_root, "labels_train.npy"), y_train)
    
    # Extract test features and labels
    X_test, y_test = extract_features_from_csv(test_csv, data_root, is_test=True)
    np.save(os.path.join(output_root, "features_test.npy"), X_test)
    if y_test is not None:
        np.save(os.path.join(output_root, "labels_test.npy"), y_test)

    # Print dataset shapes for confirmation
    print(f"Training features shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}, labels shape: {None if y_test is None else y_test.shape}")

    # Train and evaluate
    model = train_svm(X_train, y_train, save_path=os.path.join(output_root, "svm_model.pkl"))
    evaluate_svm(model, X_test, y_test, output_folder=output_root)
