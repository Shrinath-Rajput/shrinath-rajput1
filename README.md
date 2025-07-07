# shrinath-rajput1
Grain Adultration Detection Using Deep Learing and Image processing
## 📌 Project Description
This project is a deep learning-based food grain detection system that can identify and classify different types of grains (like rice, wheat, lentils, etc.) from images using computer vision and stores the output in a MySQL database.

## 🎯 Objectives
- Real-time detection of food grains from images
- Classification using deep learning model (Keras/TensorFlow)
- Storing image paths and predictions into a MySQL database
- Retrieval and analysis of previously scanned data

## 🛠️ Tech Stack
- *Python*
- *OpenCV*
- *TensorFlow / Keras*
- *MySQL*
- *Google Colab*

## 📦 Features
- Image input via camera or upload
- Preprocessing using OpenCV
- Trained model for classification
- Database connection and storage
- Lightweight, scalable codebase

## 📸 Sample Grains
- Wheat
- Rice
- Lentils
- Corn
- Beans

## 💡 Use Cases
- Grain quality inspection
- Smart farming automation
- Food sorting machines
- Agricultural analytics

## 🚀 How to Run
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Configure your MySQL credentials
4. Run main.py or use the Colab Notebook for testing

## 🔗 Google Colab Link
[Run the notebook](https://colab.research.google.com/drive/1DrIIxyerGB05QCJTGck3U66FM3Li-r-z)
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# --- CONFIG ---
IMG_SIZE = 224
EPOCHS = 5
BATCH_SIZE = 16
DATASET_DIR = "/content/drive/MyDrive/foodgrainimages/foodgrain" # Corrected DATASET_DIR
MODELS = {
    "MobileNetV2": (MobileNetV2, mobilenet_preprocess),
    "VGG16": (VGG16, vgg_preprocess),
    "ResNet50": (ResNet50, resnet_preprocess),
}

# --- Debugging: Print directory contents ---
print(f"Contents of DATASET_DIR: {os.listdir(DATASET_DIR)}")
for folder in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(path):
        print(f"Contents of {folder}: {os.listdir(path)[:5]}...") # Print only the first 5 files

# --- Load images from folders ---
X, y, labels = [], [], []
label_map = {}
label_counter = 0

for folder in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path): continue # Ensure it's a directory

    if folder not in label_map:
        label_map[folder] = label_counter
        label_counter += 1

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        # Check if the item in the folder is a file before trying to load it
        if os.path.isfile(img_path):
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label_map[folder])
            except:
                continue

X = np.array(X)
y = np.array(y)
CATEGORIES = list(label_map.keys())
print(CATEGORIES)
y_cat = to_categorical(y, num_classes=len(CATEGORIES))

# --- Train/Test Split ---
# Adjusted train_size to 0.8 to ensure a non-empty training set
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42,train_size=0.8)

# --- Train & Evaluate Models ---
results = {}

for model_name, (model_class, preprocess) in MODELS.items():
    print(f"\n🔧 Training {model_name}...")

    X_train_p = preprocess(X_train.copy().astype('float32'))
    X_test_p = preprocess(X_test.copy().astype('float32'))

    base_model = model_class(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CATEGORIES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_p, y_train, validation_data=(X_test_p, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    y_pred_probs = model.predict(X_test_p)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    label=list(range(len(CATEGORIES)))
    print(label)

    report = classification_report(y_true, y_pred, target_names=CATEGORIES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    acc = report['accuracy']
    precision = np.mean([report[label]['precision'] for label in CATEGORIES])
    recall = np.mean([report[label]['recall'] for label in CATEGORIES])
    f1 = np.mean([report[label]['f1-score'] for label in CATEGORIES])

    # --- Store results ---
    results[model_name] = {
        "accuracy": acc * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "conf_matrix": cm,
        "history": history,
        "y_true": y_true,
        "y_pred_probs": y_pred_probs
    }

    # --- Plot Accuracy & Loss ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # --- ROC Curve (macro-average)
    plt.figure()
    fpr = dict()
    tpr = dict()
    for i in range(len(CATEGORIES)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        plt.plot(fpr[i], tpr[i], label=f'{CATEGORIES[i]}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Best Model ---
best_model = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n🏆 Best Model: {best_model} with Accuracy = {results[best_model]['accuracy']:.2f}%")
