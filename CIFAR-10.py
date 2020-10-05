%tensorflow_version 2.x  
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plot_img

# from tensorflow.keras.callbacks import TensorBoard
import datetime

 
# Enable XLA - Accelerated Linear Algebra. 
# The results are improvements in speed and memory usage: 
# most internal benchmarks run ~1.15x faster after XLA is enabled.
tf.config.optimizer.set_jit(True)


n_cpus = 2


# sess = tf.Session(config=tf.ConfigProto(
#     device_count={ "CPU": n_cpus },
#     inter_op_parallelism_threads=n_cpus,
#     intra_op_parallelism_threads=1,
# ))


# Function to load the dataset
def load_dataset():
	#  Load dataset
  (trn_imgs, trn_lbs), (tst_imgs, tst_lbs) = datasets.cifar10.load_data()

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  
  # Just for Fun. Plot the first 100 images.
  plot_img.figure(figsize=(20,20))
  for i in range(100):
    plot_img.subplot(10, 10, 1 + i)
    plot_img.imshow(trn_imgs[i] ,cmap=plot_img.cm.binary)
    plot_img.xlabel(class_names[trn_lbs[i][0]])
  plot_img.show()
  return trn_imgs, trn_lbs, tst_imgs, tst_lbs

	# Function for data pre-processing
def pre_process(trn_imgs, tst_imgs):
  if trn_imgs.all():
    # Randomly flip the image horizontally.
    trn_imgs = tf.image.random_flip_left_right(trn_imgs)
    # Randomly adjust hue, contrast and saturation.
    trn_imgs = tf.image.random_hue(trn_imgs, max_delta=0.05)
    trn_imgs = tf.image.random_contrast(trn_imgs, lower=0.3, upper=1.0)
    trn_imgs = tf.image.random_brightness(trn_imgs, max_delta=0.2)
    trn_imgs = tf.image.random_saturation(trn_imgs, lower=0.0, upper=2.0)
    # convert dataset from integers to floats
    train_float = trn_imgs.astype('float32')
    # Normalize by dividing by 255.0 to ensure it falls btw range 0-1
    train_normalized = train_float / 255.0
  
  elif tst_imgs.all():
    # convert dataset from integers to floats
    test_float = tst_imgs.astype('float32')
    # Normalize by dividing by 255.0 to ensure it falls btw range 0-1
    test_normalized = test_float / 255.0
  
  trn_imgs, tst_imgs = trn_imgs / 255.0, tst_imgs / 255.0
  return trn_imgs, tst_imgs

# Function for the training model
def training_model():
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
  return model

# train model
def train_model(model):
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  log_dir="logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  
  # write to file
  file_wr = tf.summary.create_file_writer(log_dir + "/training_metrics")
  file_wr.set_as_default()

  # Callback for saving the model
  save_callback = tf.keras.callbacks.ModelCheckpoint('saved_model.h5', verbose=1, save_weights_only=False)
  
  history = model.fit(trn_imgs, trn_lbs, epochs=3, shuffle=True,
                    validation_data=(tst_imgs, tst_lbs), callbacks=[save_callback])
  return history

# Evaluate the model
def evaluate_model(model, tst_imgs,  tst_lbs):
  # if tf.keras.models.load_model('saved_model.h5'):
  #   saved_model = tf.keras.models.load_model('saved_model.h5')
  #   test_loss, test_accuracy = saved_model.evaluate(tst_imgs,  tst_lbs, verbose=2)
  # else:
  #   test_loss, test_accuracy = model.evaluate(tst_imgs,  tst_lbs, verbose=2)

  test_loss, test_accuracy = model.evaluate(tst_imgs,  tst_lbs, verbose=2)
    
  return test_loss, test_accuracy

# Present Result
def present_result(history, test_loss, test_accuracy):
  template = '\nTest Loss: {}, \nTest Accuracy: {}'
  print(template.format(test_loss, test_accuracy*100))

  plot_img.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plot_img.plot(history.history['accuracy'], label='accuracy')
  plot_img.ylabel('Accuracy')
  plot_img.xlabel('Epoch')
  plot_img.ylim([0.3, 1])
  plot_img.legend(loc='upper right')

def main():
  global trn_imgs, trn_lbs, tst_imgs, tst_lbs  

  # load the dataset
  trn_imgs, trn_lbs, tst_imgs, tst_lbs = load_dataset()
  # call pre-processing function 
  trn_imgs, tst_imgs = pre_process(trn_imgs, tst_imgs)

  # call model definition function
  model = training_model()

  # display the model summary
  model.summary()

  # Train model
  history = train_model(model)

  #Evaluate the model
  test_loss, test_accuracy = evaluate_model(model, tst_imgs,  tst_lbs)

  # present results
  present_result(history, test_loss, test_accuracy)

# main calling function
main()
