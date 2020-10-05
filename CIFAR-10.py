import os
import tensorflow as tf
import matplotlib.pyplot as plot_img
import numpy as np


from tensorflow.keras import datasets, layers, models

def load_and_preprocess_dataset(batch_size):
  (trn_imgs, y_train), (tst_imgs, tst_lbs) = datasets.cifar10.load_data()

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  # convert dataset from integers to floats
  trn_imgs = trn_imgs / np.float32(255) # Normalize by dividing by 255.0 to ensure it falls btw range 0-1
  # convert the dataset from integer to floats
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (trn_imgs, y_train)).shuffle(60000).repeat().batch(batch_size)

  
  # Just for Fun. Plot the first 100 images.
  plot_img.figure(figsize=(20,20))
  for i in range(100):
    plot_img.subplot(10, 10, 1 + i)
    plot_img.imshow(trn_imgs[i] ,cmap=plot_img.cm.binary)
    plot_img.xlabel(class_names[y_train[i][0]])
  plot_img.show()
  return train_dataset, tst_imgs, tst_lbs

def build_and_compile_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  
  # Flatten the features before feeding to the densly connected layers which
  # determines the class of the image based on the absense or the presence of the featur thereoff
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10))

  
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model


# Evaluate the model
def evaluate_model(model, tst_imgs,  tst_lbs):
  test_loss, test_accuracy = model.evaluate(tst_imgs,  tst_lbs, verbose=2)
    
  return test_loss, test_accuracy
