import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

data_directory = 'C:/Users/dasda/OneDrive/Documents/Defungi'
image_height = 224
image_width = 224
batch_size = 32
epochs = 15
validation_split = 0.2 #Use 20% of the data
num_classes = 5

train_ds = image_dataset_from_directory(
    data_directory,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = image_dataset_from_directory(
    data_directory,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical'
)


class_names = train_ds.class_names
autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)

#Data Augmentation to prevent overfitting
data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(image_height, image_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ]
)

#Using MobileNetV2 but will test other models in the future
base_model = tf.keras.applications.MobileNetV2(input_shape=(image_height, image_width, 3),
                                               include_top=False,
                                               weights='imagenet')

#Freeze the base model
base_model.trainable = False

inputs = keras.Input(shape=(image_height, image_width, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) #Preprocess input
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x) #Final layer is softmax

model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle("Model Training History", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

final_loss, final_accuracy = model.evaluate(val_ds)
print(f"Final Validation Loss: {final_loss:}")
print(f"Final Validation Accuracy: {final_accuracy:}")


model.save('fungi_classifier_model.keras')
