import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

# Define the label distribution counts
label_counts = {
    'OP': 7441,
    'NA': 14913,
    'ID': 2414,
    'IN': 8878,
    'HI': 1433,
    'LY': 477,
    'IP': 1402,
    'OTHER': 407,
    'SP': 514
}

# Total number of instances
total_rows = 33905

# Create a list of all possible labels
all_labels = list(label_counts.keys())

# Calculate the probability of each label based on its frequency
adjusted_label_probabilities = {label: count / total_rows for label, count in label_counts.items()}

# Initialize MultiLabelBinarizer for the 9 labels
mlb = MultiLabelBinarizer(classes=all_labels)

# Simulate true multilabel data based on label frequency for each class
np.random.seed(42)  # For reproducibility
freq_multilabels = []
for _ in range(total_rows):
    labels_for_row = []
    for label in all_labels:
        # Randomly decide whether to include a label based on its frequency probability
        if np.random.rand() < adjusted_label_probabilities[label]:
            labels_for_row.append(label)
    if not labels_for_row:
        # Ensure at least one label is selected, if none were added
        labels_for_row.append(np.random.choice(all_labels))
    freq_multilabels.append(labels_for_row)

print(freq_multilabels)
# Binarize the true labels
true_binarized_corrected = mlb.fit_transform(freq_multilabels)
print(true_binarized_corrected)

# Majority vote: Predict only 'NA' for all rows as a multi-label prediction
majority_prediction_multilabel_corrected = [['NA'] for _ in range(total_rows)]
majority_binarized_corrected = mlb.transform(majority_prediction_multilabel_corrected)

# Calculate the majority vote micro F1 score (multi-label)
majority_f1_multilabel_corrected = f1_score(true_binarized_corrected, majority_binarized_corrected, average='micro')

# Simulate random multi-label predictions based on the label distribution
random_multilabel_predictions_corrected = []
for _ in range(total_rows):
    labels_for_row = []
    for label in all_labels:
        # Randomly decide whether to include a label based on its frequency probability
        if np.random.rand() < adjusted_label_probabilities[label]:
            labels_for_row.append(label)
    if not labels_for_row:
        # Ensure at least one label is selected, if none were added
        labels_for_row.append(np.random.choice(all_labels))
    random_multilabel_predictions_corrected.append(labels_for_row)

# Binarize the random predictions
random_binarized_corrected = mlb.transform(random_multilabel_predictions_corrected)

# Calculate the random micro F1 score (multi-label)
random_f1_multilabel_corrected = f1_score(true_binarized_corrected, random_binarized_corrected, average='micro')

# Print the results
print(f"Majority Vote Micro F1 Score: {majority_f1_multilabel_corrected}")
print(f"Random Micro F1 Score: {random_f1_multilabel_corrected}")