import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Define base directory
BASE_DIR = '/Users/prashantambati/Documents/cropx'

print("Loading the best trained model...")
model_path = os.path.join(BASE_DIR, 'best_cropx_model.keras')
cropx_model = tf.keras.models.load_model(model_path)

# Load the preprocessor and label encoder
try:
    with open(os.path.join(BASE_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    print("Preprocessor and label encoder loaded successfully")
except Exception as e:
    print(f"Error loading preprocessor or label encoder: {e}")
    exit(1)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(os.path.join(BASE_DIR, 'crop_data_new.csv'))
print(f"Dataset shape: {data.shape}")

# Separate features and target
X = data.drop('Crop', axis=1)
y = data['Crop']

# Identify categorical and numerical columns
categorical_cols = ['Soil Type', 'Region']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Encode the target
y_encoded = label_encoder.transform(y)

# Split the data into test set (20% of the data)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Preprocess the test data
X_test_processed = preprocessor.transform(X_test)

# Evaluate the model
print("\nEvaluating the best model...")
evaluation_results = cropx_model.evaluate(X_test_processed, y_test_encoded, verbose=1)
cropx_loss = evaluation_results[0]
cropx_accuracy = evaluation_results[1]
cropx_top3_accuracy = evaluation_results[2]

print(f"\nCropX v3 test metrics:")
print(f"  - Loss: {cropx_loss:.4f}")
print(f"  - Accuracy: {cropx_accuracy:.4f}")
print(f"  - Top-3 Accuracy: {cropx_top3_accuracy:.4f}")

# Generate predictions
y_pred = np.argmax(cropx_model.predict(X_test_processed), axis=1)

# Get class names
test_classes_names = [label_encoder.classes_[i] for i in np.unique(y_test_encoded)]

# Print classification report
print("\nDetailed classification report:")
print(classification_report(y_test_encoded, y_pred, labels=np.unique(y_test_encoded), 
                           target_names=test_classes_names))

# Generate confusion matrix visualization
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_encoded, y_pred)

# Normalize confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('CropX v3 Normalized Confusion Matrix', fontsize=16)
plt.colorbar()

# Add class labels
tick_marks = np.arange(len(test_classes_names))
plt.xticks(tick_marks, test_classes_names, rotation=45, ha='right', fontsize=8)
plt.yticks(tick_marks, test_classes_names, fontsize=8)

# Add text annotations
thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, format(cm_norm[i, j], '.2f'),
                 ha="center", va="center",
                 color="white" if cm_norm[i, j] > thresh else "black",
                 fontsize=7)

plt.tight_layout()
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.savefig(os.path.join(BASE_DIR, 'cropx_v3_confusion_matrix_test.png'), dpi=300)
print("Confusion matrix saved as 'cropx_v3_confusion_matrix_test.png'")

# Test the model with a few sample inputs
print("\nTesting model with sample inputs...")

# Get 5 random samples from the test set
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y = y_test_encoded[sample_indices]
sample_y_names = [label_encoder.classes_[i] for i in sample_y]

# Preprocess the samples
sample_X_processed = preprocessor.transform(sample_X)

# Get predictions
sample_predictions = cropx_model.predict(sample_X_processed)
sample_pred_classes = np.argmax(sample_predictions, axis=1)
sample_pred_names = [label_encoder.classes_[i] for i in sample_pred_classes]

# Get top 3 predictions for each sample
top3_indices = np.argsort(-sample_predictions, axis=1)[:, :3]
top3_probs = np.sort(-sample_predictions, axis=1)[:, :3] * -1
top3_names = [[label_encoder.classes_[i] for i in row] for row in top3_indices]

# Print results
print("\nSample predictions:")
for i in range(len(sample_X)):
    print(f"\nSample {i+1}:")
    print(f"  Input features: {sample_X.iloc[i].to_dict()}")
    print(f"  Actual crop: {sample_y_names[i]}")
    print(f"  Predicted crop: {sample_pred_names[i]} (confidence: {sample_predictions[i][sample_pred_classes[i]]:.4f})")
    print(f"  Top 3 predictions:")
    for j in range(3):
        print(f"    {j+1}. {top3_names[i][j]} ({top3_probs[i][j]:.4f})")

print("\nModel testing complete!")