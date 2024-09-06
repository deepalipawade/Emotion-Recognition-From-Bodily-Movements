import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Function to map original labels to new labels
def map_labels(labels):
    # Mapping: 0,1,2 -> 0; 3,4,5 -> 1; 6,7,8 -> 2; 9,10,11 -> 3
    new_labels = np.copy(labels)
    new_labels[(labels >= 0) & (labels <= 2)] = 0
    new_labels[(labels >= 3) & (labels <= 5)] = 1
    new_labels[(labels >= 6) & (labels <= 8)] = 2
    new_labels[(labels >= 9) & (labels <= 11)] = 3
    return new_labels

print("------------------- view 1 pair 7 ----------------")
# Load embeddings and labels
train_embeddings = np.load('view1_7/train_i3d_embedding.npy')
train_labels = np.load('view1_7/train_labels.npy')
val_embeddings = np.load('view1_7/val_i3d_embeddings.npy')
val_labels = np.load('view1_7/val_labels.npy')
test_embeddings = np.load('view1_7/test_i3d_embeddings.npy')
test_labels = np.load('view1_7/test_labels.npy')

# Map original labels to new labels
train_labels = map_labels(train_labels)
val_labels = map_labels(val_labels)
test_labels = map_labels(test_labels)

# Combine train and validation embeddings and labels
X_train_val = np.concatenate((train_embeddings, val_embeddings), axis=0)
y_train_val = np.concatenate((train_labels, val_labels), axis=0)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # SVM kernel
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_val, y_train_val)

# Best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Train the SVM classifier with the best hyperparameters on the combined train+validation set
best_svm_classifier = grid_search.best_estimator_

# Predict on the training data
train_predictions = best_svm_classifier.predict(train_embeddings)

# Calculate accuracy on the training set
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f'Accuracy of SVM classifier on the training set: {train_accuracy * 100:.2f}%')

# Predict on the test data
test_predictions = best_svm_classifier.predict(test_embeddings)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Accuracy of SVM classifier on the test set: {test_accuracy * 100:.2f}%')

# Calculate other metrics on the test set with zero_division parameter
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average='weighted', zero_division=0)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:\n", conf_matrix)

# Detailed classification report with zero_division parameter
print("Classification Report:\n", classification_report(test_labels, test_predictions, zero_division=0))

# Optionally, you can also evaluate on the validation set (even though it's included in training)
val_predictions = best_svm_classifier.predict(val_embeddings)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Accuracy of SVM classifier on the validation set: {val_accuracy * 100:.2f}%')
