import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load I3D embeddings and labels
train_embeddings = np.load('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/train_i3d_embeddings_4_power.npy')
train_labels = np.load('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/train_labels_4_power.npy')
val_embeddings = np.load('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/val_i3d_embeddings_4_power.npy')
val_labels = np.load('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/val_labels_4_power.npy')
test_embeddings = np.load('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/test_i3d_embeddings_4_power.npy')
test_labels = np.load('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/test_labels_4_power.npy')

# Standardize the features
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
val_embeddings = scaler.transform(val_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Combine train and validation embeddings and labels
X_train_val = np.concatenate((train_embeddings, val_embeddings), axis=0)
y_train_val = np.concatenate((train_labels, val_labels), axis=0)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_val = pca.fit_transform(X_train_val)
test_embeddings = pca.transform(test_embeddings)

# Define and train the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(X_train_val, y_train_val)

# Predict on the test data
test_predictions = svm_classifier.predict(test_embeddings)

# Calculate accuracy, F1 score, precision, and recall on the test set
test_accuracy = accuracy_score(test_labels, test_predictions)
test_f1_score = f1_score(test_labels, test_predictions, average='weighted')
test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')

print("------------------------- Approach 4-power ------------------------- ")
print(f'Accuracy of SVM classifier on the test set: {test_accuracy * 100:.2f}%')
print(f'F1 Score of SVM classifier on the test set: {test_f1_score:.2f}')
print(f'Precision of SVM classifier on the test set: {test_precision:.2f}')
print(f'Recall of SVM classifier on the test set: {test_recall:.2f}')

# Optionally, you can also evaluate on the validation set
val_embeddings_pca = pca.transform(val_embeddings)
val_predictions = svm_classifier.predict(val_embeddings_pca)
val_accuracy = accuracy_score(val_labels, val_predictions)
val_f1_score = f1_score(val_labels, val_predictions, average='weighted')
val_precision = precision_score(val_labels, val_predictions, average='weighted')
val_recall = recall_score(val_labels, val_predictions, average='weighted')

print(f'Accuracy of SVM classifier on the validation set: {val_accuracy * 100:.2f}%')
print(f'F1 Score of SVM classifier on the validation set: {val_f1_score:.2f}')
print(f'Precision of SVM classifier on the validation set: {val_precision:.2f}')
print(f'Recall of SVM classifier on the validation set: {val_recall:.2f}')
