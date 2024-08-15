import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Data preparation
food_folders = ['path_to_food1', 'path_to_food2', 'path_to_foodN']
data, labels, calories = [], [], []
food_labels = {0: 'Food1', 1: 'Food2', 2: 'FoodN'}
food_calories = {0: 100, 1: 200, 2: 300}
img_size = 128

for label, folder in enumerate(food_folders):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
            calories.append(food_calories[label])

data = np.array(data).astype('float32') / 255.0
labels = np.array(labels)
calories = np.array(calories)

X_train, X_test, y_train, y_test, cal_train, cal_test = train_test_split(data, labels, calories, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(food_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

# Evaluate food recognition accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Food Recognition Accuracy: {accuracy * 100:.2f}%")

# Predict food labels
predicted_labels = model.predict(X_test)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Estimate calorie content
estimated_calories = [food_calories[label] for label in predicted_labels]

# Calculate the mean absolute error
mae = mean_absolute_error(cal_test, estimated_calories)
print(f"Mean Absolute Error in Calorie Estimation: {mae:.2f} calories")

# Function for recognizing food and estimating calories
def recognize_and_estimate_calories(image):
    img = cv2.resize(image, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predicted_label = np.argmax(model.predict(img), axis=1)[0]
    estimated_calories = food_calories[predicted_label]
    return food_labels[predicted_label], estimated_calories

# Example usage
image = cv2.imread('path_to_food_image.jpg')
food_item, calories = recognize_and_estimate_calories(image)
print(f"Food Item: {food_item}, Estimated Calories: {calories} kcal")
