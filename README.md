# Proccessing Data with SVM
For the CS-490 Group 2 Project, this is a continuation of Hayden's implementation using SVM.

# First Step
The first step of this project is to follow the directions in this repository:

https://github.com/haydenshubert/Processing_Data

The only roadblocks I encountered while setting up his implementation were related to the Conda environment. The errors I received were due to some of the included packages not existing or having different names on Windows.

I have included a new environment.yml that allows you to create the environment without issues and run his original implementation successfully as well.

Before moving onto using SVM, please listen to his instructions carefully, remember to change the names of the data folders as directed in his README, and carefully look at the output .npy files in the data folder given from his program.

# Running SVM
After downloading / cloning all files, make sure that you're in the current working directory and have the Conda environment activated. My two main Python files are svm.py and predictionsReport.py. These must be in the same directory (Processing_Data)  that was cloned / downloaded from his repository. 

After moving to the directory and activating the environment, you can run the SVM implementation by using the command

`python svm.py`

*Note: reading the test and train csv labels may take some time*

# Given Output

The output of this file will similar to:

<img width="1023" height="159" alt="image" src="https://github.com/user-attachments/assets/7f04dc94-43a9-4a88-a63d-495294ec9219" />

<img width="527" height="1141" alt="image" src="https://github.com/user-attachments/assets/7ac54569-a9db-42f6-aa58-96f478f03f8b" />


The first few lines of the output are explaining that the program read your training and test CSVs.
In my own run, the stats were:

- Training features: 39,209 images, each represented by 324 HOG features.

- Training labels: 39,209 labels (one for each image).

- Test features: 12,630 images, 324 features each.

- Test labels: 12,630 labels.

Then, the linear SVM was trained on the training features, and the model was saved to svm_model.pkl for later use (no retraining needed).

- The SVM correctly classified ~81.5% of the test images.

- This is overall accuracy: correct predictions / total predictions.

In the given classification report, there are 4 columns. These 4 are precision, recall, f1-score, and support.

- Precision: How many predicted as class X were actually class X.

- Recall: How many actual class X were correctly predicted.

- F1-score: Harmonic mean of precision and recall.

- Support: Number of images in each class in the test set.

Under this report there are the average of these metrics,

- Accuracy: Overall correct predictions.

- Macro avg: Average across all classes (treats all classes equally).

- Weighted avg: Average across classes weighted by number of images per class.

The included Confusion Matrix at the bottom of the given output gives a table used to evaluate the performance of a classification model. It shows how well the model predicted each class compared to the true labels. Each row represents the true class, and each column represents the predicted class. However this is not the entire matrix as there are many more classes in the dataset.

- Rows = actual labels, columns = predicted labels.

- The entry at row i, column j tells you how many samples of class i were predicted as class j.

- Example: 36 in row 0, column 0 → 36 images of class 0 were correctly predicted as class 0.

- 18 in row 0, column 1 → 18 images of class 0 were incorrectly predicted as class 1.

- Diagonal entries (top-left to bottom-right) are the correct predictions.

- Off-diagonal entries are misclassifications.

**Overall Output Files**

These will all be saved under the folder svmOutput in the same directory.

- svm_model.pkl = trained SVM model

- features_train.npy = extracted HOG features for training

- labels_train.npy = labels for training

- features_test.npy = extracted HOG features for test set

- labels_test.npy = labels for test set

- predictions.npy = predicted classes on test set
  
# Readable Results

Basically all of the given output files are in .npy format, which are just a NumPy array of integers. The file predictionsReport.py which can be ran by the command:

'python predictionsReport.py'

can be used **after** svm.py is ran and given output is correct. This will generate a .csv file that will be in the format Filename,PredictedClass. I.E. 00000.png,16.

Although this is a long list, it improves the readability of the data provided by the SVM predictor.

