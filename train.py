import os
import random
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dat_path = r"C:\Users\hp\OneDrive\Documents\projects\Histopathology-Image-Classification\raw_data\Kather_texture_2016_image_tiles_5000\pictures"
os.listdir(dat_path)

classes = dat_path

classes

#show 1 image per class
import matplotlib.image as mpimg
plt.figure(figsize=(15, 8))


for idx, tissue_class in enumerate(classes):
    class_folder = os.path.join(dat_path, tissue_class)
    image_name = random.choice(os.listdir(class_folder))  # Pick random image
    image_path = os.path.join(class_folder, image_name)

    img = mpimg.imread(image_path)

    plt.subplot(2, 4, idx + 1)
    plt.imshow(img)
    plt.title(tissue_class)
    plt.axis("off")

plt.suptitle("Example Images from Each Histopathology Class", fontsize=16)
plt.tight_layout()
plt.show()


# Step 1: image preprocessing
# Loading and resizing all images
IMG_SIZE = 64  # resizing all images to 60x60
X = []         # Features (the images)
y = []         # Labels (the tissue types)

for label_idx, class_name in enumerate(classes):
    class_folder = os.path.join(dat_path, class_name)
    for image_file in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_file)
        img = cv2.imread(image_path)                # Load image as numbers
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Resize to 60x60
        X.append(img)                               # Image pixels (numbers)
        y.append(label_idx)  # Stored class index instead of name (like 0,1,2...)


# Converting to NumPy Arrays
import numpy as np
X = np.array(X)
y = np.array(y)

# Normalization(Feature Scaling: MinMaxScaler Manually)
X = X / 255.0  # Normalize the image pixel values to be between 0 and 1
# Spliting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)


# Encoding(One-hot encoding method)
y_train = to_categorical(y_train, num_classes=8)
y_test = to_categorical(y_test, num_classes=8)


# Visualizing using matplotlib
plt.imshow(X_train[0])             # Displays the image visually
plt.title(f"Label: {y_train[0]}")  # y_train to show the image
plt.axis('off')
plt.show()


# Naming numerical labels manually
class_names = {
    0: "TUMOR",
    1: "STROMA",
    2: "COMPLEX",
    3: "LYMPHO",
    4: "DEBRIS",
    5: "MUCOSA",
    6: "ADIPOSE",
    7: "EMPTY"
}


# Step 2: Definition and Building the CNN Model
model = Sequential()

# 1st Convolutional Block to Detect Basic Features
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Block to Detect Deeper Features
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Block for Deeper Insights
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Global Average Pooling(lightweight)
model.add(GlobalAveragePooling2D())


# Dense Layers
model.add(Dropout(0.4))  # To reduce overfitting
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))  # To reduce overfitting
model.add(Dense(8, activation='softmax'))  # 8 classes


# Step 3: Compiling the CNN Model to decide how it learn
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# Step 4: Augmentation (Feeding images into the CNN Model using ImageDataGenerator)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()


# Learning from training data
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_test, y_test, batch_size=32)


# Step 5: Training the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=14
)

# Step 6: Plotting the Training history to evaluate model(trained) performance

# Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Step 7: Evaluation of Model on Test Set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))


# Step 8: Predicting the Test Data
# Predict class probabilities
y_pred_probs = model.predict(X_test)

