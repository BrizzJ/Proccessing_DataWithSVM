import os
import numpy as np
import pandas as pd

# ------------------------------
# Paths
# ------------------------------
test_folder = "data/test_control"  # Folder containing test images
test_csv = os.path.join(test_folder, "Test.csv")  # CSV with test image labels
predictions_file = os.path.join("svmOutput", "predictions.npy")  # Predictions stored by SVM

# ------------------------------
# Load predictions
# ------------------------------
# Load the predicted labels from the .npy file
predictions = np.load(predictions_file)

# ------------------------------
# Load filenames
# ------------------------------
if os.path.exists(test_csv):
    # If the CSV exists, use the filenames listed in the CSV
    df = pd.read_csv(test_csv)
    filenames = df['Filename'].values
else:
    # Fallback: list all images in the test folder (common formats)
    filenames = [f for f in os.listdir(test_folder) if f.lower().endswith((".png", ".jpg", ".ppm"))]
    filenames.sort()  # Sort to ensure order matches predictions

# ------------------------------
# Combine filenames and predictions
# ------------------------------
# Create a DataFrame pairing each filename with its predicted class
results = pd.DataFrame({
    "Filename": filenames,
    "PredictedClass": predictions
})

# ------------------------------
# Display a preview
# ------------------------------
# Print the first 10 predictions for quick verification
print(results.head(10))

# ------------------------------
# Save results to CSV
# ------------------------------
# Ensure output directory exists
os.makedirs("svmOutput", exist_ok=True)

# Define output CSV path inside svmOutput
output_csv = os.path.join("svmOutput", "predictionsReview.csv")

# Save the DataFrame to CSV without the index
results.to_csv(output_csv, index=False)
print(f"Saved {output_csv} with filenames and predicted classes")
