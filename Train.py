import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2 as cv
import os
import keras_tuner as kt

# Load data
base_dir = "extracted_defungi"
types = os.listdir(base_dir)

def img_extract(image_types):
    features, Name = [], []
    for image_type in image_types:
        type_dir = os.path.join(base_dir, image_type)
        for i in os.listdir(type_dir):
            img_path = os.path.join(type_dir, i)
            img = cv.imread(img_path)
            img = cv.resize(img, (224, 224)) # Convert to standard input expected by the model
            features.append(img)
            Name.append(image_type)
    return np.array(features), np.array(Name)

features, Name = img_extract(types)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Name)
encoded_labels = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize

# Model builder function for Keras Tuner
def model_builder(hp):
    base_model_name = hp.Choice('base_model', ['ResNet50', 'MobileNetV2', 'EfficientNetB0'])

    base_model_class = {
        'ResNet50': tf.keras.applications.ResNet50,
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0
    }[base_model_name]

    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(hp.Int('units', min_value=64, max_value=512, step=64), activation='relu')(x)
    x = tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(x)
    output = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Initialize Keras Tuner
tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='fungi_tuner_dir',
    project_name='fungi_classifier'
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(1)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(1)

# Search for best hyperparameters
tuner.search(train_ds, validation_data=val_ds, epochs=10)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]

final_model = model_builder(best_hps)

# Retrain with best params for 30 epochs
history = final_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

# Predict on test set
y_pred_final = final_model.predict(X_test)
y_pred_one_hot_final = np.eye(y_pred_final.shape[1])[np.argmax(y_pred_final, axis=1)]

final_accuracy = accuracy_score(y_test, y_pred_one_hot_final)
print(f"Final test accuracy: {final_accuracy}")

# Confusion matrix
def cnfsn_matrix(y_test, y_pred):
    class_labels = label_encoder.classes_
    confusion = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

cnfsn_matrix(y_test, y_pred_final)
