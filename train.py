"""
Deep Learning & AI Internship — Week 1
MNIST Digit Recognizer Neural Network Training Script
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def train_and_save_model():
    print("[*] Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values between 0.0 and 1.0
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("[*] Building Deep Neural Network architecture...")
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("[*] Training model for 5 epochs...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    print("[*] Evaluating model accuracy...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n[✔] Model Test Accuracy: {test_acc * 100:.2f}%")

    print("[*] Saving trained model to 'mnist_model.h5'...")
    model.save("mnist_model.h5")
    print("[✔] Model training complete and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()