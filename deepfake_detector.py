import os
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.interpolate import make_interp_spline

# Set parameters
input_shape = (224, 224, 3)
batch_size = 64
epochs = 10
epoch_list = list(range(1, epochs + 1))
network_name = "ResNet50V2"

# Paths
base_path = r'Source path'
train_dir = os.path.join(base_path, 'filtered-dataset-full', 'training')
test_dir = os.path.join(base_path, 'filtered-dataset-full', 'testing')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='binary', shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='binary', shuffle=False)

# Load and build model
base_model = ResNet50V2(input_shape=input_shape, include_top=False, weights="imagenet")
for layer in base_model.layers[:50]:
    layer.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# Callbacks
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator,
                    callbacks=[reduce, early_stopping])

# Create directories to save output
os.makedirs(f"./Reference_Data/Graphs/{network_name}", exist_ok=True)
os.makedirs("./Reference_Data/Summary", exist_ok=True)
model_path = f"./Reference_Data/Model/{network_name}"
os.makedirs(model_path, exist_ok=True)

# Save accuracy & loss graphs
def plot_graph(metric, val_metric, name):
    x = np.linspace(min(epoch_list), max(epoch_list), 200)
    spline_train = make_interp_spline(epoch_list, metric, k=3)
    spline_val = make_interp_spline(epoch_list, val_metric, k=3)

    plt.figure()
    plt.plot(x, spline_train(x), label='Train')
    plt.plot(x, spline_val(x), label='Validation')
    plt.xticks(np.arange(1, epochs + 1, 1))
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.title(f'{name} vs Epochs')
    plt.legend()
    plt.savefig(f"./Reference_Data/Graphs/{network_name}/{name}VEpochs.png", dpi=300, bbox_inches='tight')

plot_graph(history.history['accuracy'], history.history['val_accuracy'], "Accuracy")
plot_graph(history.history['loss'], history.history['val_loss'], "Loss")

# Save model summary
with open(f"./Reference_Data/Summary/{network_name}_summary.txt", 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Save model in both formats
model.save(f"{model_path}/{network_name}.h5")
model.save(f"{model_path}/{network_name}.keras")

# Predict and evaluate
Y_pred = model.predict(test_generator)
y_pred = np.round(Y_pred).astype(int)
y_true = test_generator.classes

# Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])

# Save confusion matrix
plt.figure(figsize=(6, 5))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(f"./Reference_Data/Graphs/{network_name}/ConfusionMatrix.png", dpi=300, bbox_inches='tight')

# Save classification report
with open(f"./Reference_Data/Summary/{network_name}_report.txt", 'w') as f:
    f.write("Classification Report\n")
    f.write(report)

print("✅ Training completed and all outputs saved successfully!")
