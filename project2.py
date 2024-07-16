#2.Build and train a simple neural network using a framework like TensorFlow or PyTorch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np

# Define the input paths
training_path = "/home/manichandana-sandhaboina/Downloads/train"
validation_path = "/home/manichandana-sandhaboina/Downloads/validation"
testing_path = "/home/manichandana-sandhaboina/Downloads/test"

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

# Load training data
train_images = []
train_labels = []
for filename in os.listdir(training_path):
	path = os.path.join(training_path, filename)
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	if img is not None:
		img = cv2.resize(img, (64, 64))
		train_images.append(img)
		train_labels.append(0)  # Replace with actual label
	else:
		print(f"Error reading image file: {path}")

# Load validation data
val_images = []
val_labels = []
for filename in os.listdir(validation_path):
	path = os.path.join(validation_path, filename)
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	if img is not None:
		img = cv2.resize(img, (64, 64))
		val_images.append(img)
		val_labels.append(0)  # Replace with actual label
	else:
		print(f"Error reading image file: {path}")

# Load testing data
test_images = []
test_labels = []
for filename in os.listdir(testing_path):
	path = os.path.join(testing_path, filename)
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	if img is not None:
		img = cv2.resize(img, (64, 64))
		test_images.append(img)
		test_labels.append(0)  # Replace with actual label
	else:
		print(f"Error reading image file: {path}")

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize pixel values to between 0 and 1
train_images = train_images.astype('float32') / 255.0
val_images = val_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape data for Conv2D (add channel dimension)
train_images = np.expand_dims(train_images, axis=-1)
val_images = np.expand_dims(val_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Train the model with early stopping
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(val_images, val_labels), callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Calculate individual accuracy
test_pred = model.predict(test_images)
test_pred_class = tf.round(test_pred)
test_acc_individual = accuracy_score(test_labels, test_pred_class)

# Print the results
print("Training accuracy:", history.history['accuracy'][-1] * 100)
print("Validation accuracy:", history.history['val_accuracy'][-1] * 100)
print("Test accuracy:", test_acc * 100)
