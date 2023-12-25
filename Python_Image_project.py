import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers

(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data();
training_images, testing_images = training_images/255, testing_images/255

model = tf.keras.models.Sequential();

# model = tf.keras.models.load_model("image_classifier.model")

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 30, validation_data = (testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

model.save("image_classifier.model")