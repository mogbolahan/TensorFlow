import os
import json

import tensorflow as tf
import cifar_10

import datetime

batch_size_per_worker = 128

# Very important!!! change the 'index' value for each worker. ie 0 for worker_1, index: 1 for worker 2 and index: 2 for worker 3.
tf_config = {
    'cluster': {
        'worker': ['localhost:8080', 'localhost:8081', 'localhost:8082']
    },
    'task': {'type': 'worker', 'index': 0}
}

# Very important!!! Uncomment this line to set the TF_CONFIG for each worker
# os.environ["TF_CONFIG"] = json.dumps(tf_config)


numb_of_workers = len(tf_config['cluster']['worker'])

# Use tf.distribute to scale to multiple workers.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

batch_size = batch_size_per_worker * numb_of_workers
train_dataset, tst_imgs, tst_lbs = cifar_10.load_and_preprocess_dataset(batch_size)

with strategy.scope():
  # Keep the building and compiling code snippet within strategy.scope() to leveage the tf.distribute functionality of TensorFlow 2.x.
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
