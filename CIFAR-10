%tensorflow_version 2.x  # Comment out this line if running in a notepad environment
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plot_img

# helper function to load the train and test dataset
def load_dataset():
	#  Load dataset
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

  # Plot the first few images. Just for Fun
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  for i in range(9):
    plot_img.subplot(331 + i)
    plot_img.imshow(train_images[i] ,cmap=plt.cm.binary)
    plot_img.xlabel(class_names[train_labels[IMG_INDEX][0]])
  plot_img.show()
  return train_images, train_labels, test_images, test_labels

	# Helper function to data pre-processing
def pre_process(train_images, test_images):
	# convert dataset from integers to floats
  train_float = train_images.astype('float32')
  test_float = test_images.astype('float32')
  # Normalize by dividing by 255.0 to ensure it falls btw range 0-1
  train_normalized = train_float / 255.0
  test_normalized = test_float / 255.0
  
  train_images, test_images = train_images / 255.0, test_images / 255.0
  return train_images, test_images

# Helper function to define the training model
def training_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  
  # Flatten the features before feeding to the densly connected layers which
  # determines the class of the image based on the absense or the presence of the featur thereoff
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10))
  return model

def main():
  # load the dataset
  train_images, train_labels, test_images, test_labels = load_dataset()
  # call pre-processing helper function 
  train_images, test_images = pre_process(train_images, test_images)
  # call training model helper function
  model = training_model()
  # display the model summary
  model.summary()  
  # fit model
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  history = model.fit(train_images, train_labels, epochs=1, shuffle=True,
                    validation_data=(test_images, test_labels))

  #Evaluate the model
  test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)

  print("Accuracy: " + str(test_accuracy))
# main calling function
main()
