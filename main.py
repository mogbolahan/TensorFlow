import os
import json

import tensorflow as tf
import cifar_10
import subprocess
import multiprocessing as mp

import datetime


n_cpu = mp.cpu_count()

def controller(batch_size, tf_config):
      numb_of_workers = len(tf_config['cluster']['worker'])

      print ("I am worker: #" + str(tf_config['task']['index'] + 1))
      print(json.dumps(tf_config, indent= 4))
      
      # print("\nNumber. of CPU available for processing: " + str(n_cpu) +'\n')

      
      # configure environment for worker
      # os.environ["TF_CONFIG"] = json.dumps(tf_config)

      # Use tf.distribute to scale to multiple workers.
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

      batch_size = batch_size* numb_of_workers
      train_dataset, tst_imgs, tst_lbs = cifar_10.load_and_preprocess_dataset(batch_size)

      with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        model = cifar_10.build_and_compile_model()

        
        # display the model summary
        model.summary()


        
      log_dir="logs" + datetime.datetime.now().strftime("%H%M%S")
        
      # write to file
      file_wr = tf.summary.create_file_writer(log_dir + "/training_metrics")
      file_wr.set_as_default()

      # Callback for saving the model
      save_callback = tf.keras.callbacks.ModelCheckpoint('saved_model.h5', verbose=1, save_weights_only=False)

      history = model.fit(train_dataset, epochs=3, steps_per_epoch=70, shuffle=True, callbacks=[save_callback])

      #Evaluate the model
      test_loss, test_accuracy = cifar_10.evaluate_model(model, tst_imgs, tst_lbs)


      # present results
      template = '\nTest Loss: {}, \nTest Accuracy: {} %'
      print(template.format(test_loss, test_accuracy*100))

      return(history, test_loss, test_accuracy)
    
