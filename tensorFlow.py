#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:57:32 2019

@author: abdulrahman_ce
"""

import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images / 255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
   
print(test_labels[0])
print(classifications[0]) 
print(max(classifications[0]))
plt.imshow(test_images[0])