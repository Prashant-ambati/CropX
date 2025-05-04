import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define base directory
BASE_DIR = '/Users/prashantambati/Documents/cropx'

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(os.path.join(BASE_DIR, 'crop_data_new.csv'))

# Display basic dataset information
print(f"Dataset shape: {data.shape}")
print(f"Crop distribution:\n{data['Crop'].value_counts()}")

# Separate features and target
X = data.drop('Crop', axis=1)
y = data['Crop']

# Identify categorical and numerical columns
categorical_cols = ['Soil Type', 'Region']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"Numerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Create preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Get unique crop classes
crop_classes = y.unique()
n_classes = len(crop_classes)
print(f"Number of crop classes: {n_classes}")
print(f"Crop classes: {crop_classes}")

# Convert target to categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check for classes with only one sample
class_counts = np.bincount(y_encoded)
rare_classes = np.where(class_counts == 1)[0]
if len(rare_classes) > 0:
    print(f"Warning: Classes with only one sample: {rare_classes}")
    print("Using non-stratified split due to rare classes")
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
else:
    # Split the data with stratification
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Verify that all classes are represented in both training and test sets
train_classes = np.unique(y_train_encoded)
test_classes = np.unique(y_test_encoded)
print(f"Number of classes in training set: {len(train_classes)}")
print(f"Number of classes in test set: {len(test_classes)}")
print(f"Classes in training set: {train_classes}")
print(f"Classes in test set: {test_classes}")

# Preprocess the data
print("Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get the number of features after preprocessing
n_features = X_train_processed.shape[1]
print(f"Number of features after preprocessing: {n_features}")

# =====================================================================
# CropX - Custom Deep Learning Architecture
# =====================================================================

def create_cropx_model_v3(n_features, n_classes):
    """
    CropX v3: An optimized deep learning model with efficient architecture
    for crop prediction based on soil and environmental features.
    Designed for faster training while maintaining good performance.
    """
    # Input layer
    inputs = Input(shape=(n_features,))
    
    # First branch - Deep representation with simplified architecture
    x1 = Dense(384, activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Dense(192, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.25)(x1)
    
    # Second branch - Direct representation for simpler patterns
    x2 = Dense(256, activation='relu')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Third branch - Specialized for soil and environmental features
    x3 = Dense(128, activation='relu')(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.25)(x3)
    
    # Merge branches
    merged = Concatenate()([x1, x2, x3])
    
    # Final layers with efficient dimensionality reduction
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # Output layer
    outputs = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="CropX_v3")
    
    # Compile model with optimized learning rate and metrics
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Slightly higher learning rate for faster convergence
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

# Create CropX v3 model
print("\nCreating CropX v3 model...")
cropx_model = create_cropx_model_v3(n_features, n_classes)
cropx_model.summary()

# Optimized callbacks for training with shorter timeframe
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,  # Reduced patience to avoid long training times
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # Faster learning rate reduction
    patience=5,  # Reduced patience for learning rate reduction
    min_lr=0.0001,  # Slightly higher minimum learning rate
    verbose=1
)

# Add model checkpoint callback to save the best model during training
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(BASE_DIR, 'best_cropx_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train CropX v3 model with optimized parameters for faster training
print("\nTraining CropX v3 model...")
start_time = time.time()

# Use a more sophisticated training approach with class weights if needed
class_weights = None
if len(rare_classes) > 0:
    # Calculate class weights to handle imbalanced classes
    print("Calculating class weights for imbalanced classes...")
    class_weights = {i: len(y_train_encoded) / (len(np.unique(y_train_encoded)) * count) 
                    for i, count in enumerate(np.bincount(y_train_encoded)) if count > 0}
    print(f"Class weights: {class_weights}")

cropx_history = cropx_model.fit(
    X_train_processed, y_train_encoded,
    epochs=30,  # Reduced epochs to avoid timeout
    batch_size=128,  # Larger batch size for faster training
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    class_weight=class_weights,
    verbose=1
)

cropx_training_time = time.time() - start_time
print(f"CropX v3 training time: {cropx_training_time:.2f} seconds")

# Evaluate CropX model with more detailed metrics
print("\nEvaluating CropX v3 model...")
evaluation_results = cropx_model.evaluate(X_test_processed, y_test_encoded, verbose=1)
cropx_loss = evaluation_results[0]
cropx_accuracy = evaluation_results[1]
cropx_top3_accuracy = evaluation_results[2]

print(f"CropX v3 test metrics:")
print(f"  - Loss: {cropx_loss:.4f}")
print(f"  - Accuracy: {cropx_accuracy:.4f}")
print(f"  - Top-3 Accuracy: {cropx_top3_accuracy:.4f}")

# =====================================================================
# Model Performance Visualization
# =====================================================================
print("\n===== CropX v3 Model Performance =====")
print(f"CropX v3 (Enhanced Deep Learning): {cropx_accuracy:.4f}")
print(f"CropX v3 Top-3 Accuracy: {cropx_top3_accuracy:.4f}")

# Enhanced visualization for CropX v3
plt.figure(figsize=(15, 10))

# Accuracy plot
plt.subplot(2, 2, 1)
plt.plot(cropx_history.history['accuracy'], label='Train', linewidth=2)
plt.plot(cropx_history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('CropX v3 Model Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Loss plot
plt.subplot(2, 2, 2)
plt.plot(cropx_history.history['loss'], label='Train', linewidth=2)
plt.plot(cropx_history.history['val_loss'], label='Validation', linewidth=2)
plt.title('CropX v3 Model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Top-3 Accuracy plot
plt.subplot(2, 2, 3)
plt.plot(cropx_history.history['top_3_accuracy'], label='Train', linewidth=2, color='green')
plt.plot(cropx_history.history['val_top_3_accuracy'], label='Validation', linewidth=2, color='darkgreen')
plt.title('CropX v3 Top-3 Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Top-3 Accuracy', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Learning Rate plot if available
if 'lr' in cropx_history.history:
    plt.subplot(2, 2, 4)
    plt.semilogy(cropx_history.history['lr'], linewidth=2, color='purple')
    plt.title('Learning Rate Schedule', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'cropx_v3_training_history.png'), dpi=300)
print("CropX v3 training history saved as 'cropx_v3_training_history.png'")

# Generate confusion matrix visualization
plt.figure(figsize=(12, 10))
y_pred = np.argmax(cropx_model.predict(X_test_processed), axis=1)
cm = confusion_matrix(y_test_encoded, y_pred)

# Normalize confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('CropX v3 Normalized Confusion Matrix', fontsize=16)
plt.colorbar()

# Add class labels
test_classes_names = [label_encoder.classes_[i] for i in np.unique(y_test_encoded)]
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
plt.savefig(os.path.join(BASE_DIR, 'cropx_v3_confusion_matrix.png'), dpi=300)
print("CropX v3 confusion matrix saved as 'cropx_v3_confusion_matrix.png'")

# Generate detailed classification report for the CropX v3 model
print(f"\nDetailed report for the CropX v3 model:")
# Get the classes present in the test set
test_classes_names = [label_encoder.classes_[i] for i in np.unique(y_test_encoded)]
print(f"Classes in test set: {test_classes_names}")
# Use only the classes present in the test set for the classification report
report = classification_report(y_test_encoded, y_pred, labels=np.unique(y_test_encoded), 
                           target_names=test_classes_names, output_dict=True)
print(classification_report(y_test_encoded, y_pred, labels=np.unique(y_test_encoded), 
                           target_names=test_classes_names))

# Save classification report as CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(BASE_DIR, 'cropx_v3_classification_report.csv'))
print("Classification report saved as 'cropx_v3_classification_report.csv'")

# =====================================================================
# Save Model and Preprocessing Components
# =====================================================================
print("\nSaving model and preprocessing components...")

# Save the model
try:
    # Save in .keras format (TensorFlow 2.x preferred format)
    cropx_model.save(os.path.join(BASE_DIR, 'cropx_model.keras'))
    print("Model saved successfully as cropx_model.keras")
except Exception as e:
    print(f"Error saving model in .keras format: {e}")
    # Try saving in h5 format as a fallback
    try:
        cropx_model.save(os.path.join(BASE_DIR, 'cropx_model.h5'))
        print("Model saved successfully as cropx_model.h5")
    except Exception as e2:
        print(f"Error saving model in h5 format: {e2}")

# Save preprocessor and label encoder
with open(os.path.join(BASE_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)
    print("Preprocessor saved successfully")
    
with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
    print("Label encoder saved successfully")

# Save model results
model_results = {
    "model_version": "CropX v3",
    "model_architecture": "Optimized multi-branch neural network",
    "dataset": "crop_data_new.csv",
    "accuracy": cropx_accuracy,
    "top_3_accuracy": cropx_top3_accuracy,
    "loss": cropx_loss,
    "training_time": cropx_training_time,
    "n_classes": n_classes,
    "n_features": n_features,
    "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_parameters": {
        "batch_size": 128,
        "initial_learning_rate": 0.0005,
        "epochs_trained": len(cropx_history.history['loss']),
        "early_stopping_patience": 10,
        "class_weighting_used": class_weights is not None
    }
}

with open(os.path.join(BASE_DIR, 'model_results.pkl'), 'wb') as f:
    pickle.dump(model_results, f)
    print("Model results saved successfully")

print("\nCropX v3 model, preprocessor, and results saved successfully!")
print(f"Model accuracy: {cropx_accuracy:.4f}")
print(f"Top-3 accuracy: {cropx_top3_accuracy:.4f}")
print(f"Training time: {cropx_training_time:.2f} seconds")
print("\nTraining complete!")