# TensorFlow
# Problem Statement:
Use TensorFlow and code a classification model to train the CIFAR-10 dataset on CPU
using a simple convolutional neural network of choice. The training should be able to run on a single computer (without
a GPU), but using 3 workers on that computer (i.e. 3 processes).Take advantage of data parallel frameworks to scale to multiple workers.
Important: Confirm that your solution uses 3 distinct workers.

# Solution approach / Pseudo-code
# 1. Load the Cifar-10 image set. 
 a) Load the data from the toronto.edu url
 b.) Perform data preprocessing:
    # Randomly flip the image horizontally.
    # Randomly adjust hue, contrast and saturation.
    # convert dataset from integers to floats
    
# 2. Define a model: TensorFlow 2.x utilizes Keras API. Though a model can be defined in Keras using either the Sequential API or functional API, I used the Sequential API for this solution because it is more consise
# 3. Feed the data into the model: Shovel the data. 
  # Flatten the features before feeding to the densly connected layers which determines the class of the image based on the absense or the presence of the featur thereoff
# 4. Train the model. 
Add callbacks for monitoring progress
# 5. Evaluate the model
# 6. # Present Result

# IMPORTANT Intermediate step
Accelerate the training speed with multiple CPU. I utilized multi-node istributed strategy in the training by leveraging Tensorflow's tf.distribute.experimental.MultiWorkerMirroredStrategy() function which requires that the multi-node a TF_CONFIG environment variable to be set for each node.
The training was distributed to the three workers by enclosing the:
model building and model.compile() call within strategy.scope() which then dictates how and where the variables thereoff are created, 

The configuration for each worker is as follows:

# batch_size_per_worker = 128

# worker 1
tf_config = {
    'cluster': {
        'worker': ['localhost:8080', 'localhost:8081', 'localhost:8082']
    },
    'task': {'type': 'worker', 'index': 0}
}

# worker 2
tf_config = {
    'cluster': {
        'worker': ['localhost:8080', 'localhost:8081', 'localhost:8082']
    },
    'task': {'type': 'worker', 'index': 1}
}

# worker 3
tf_config = {
    'cluster': {
        'worker': ['localhost:8080', 'localhost:8081', 'localhost:8082']
    },
    'task': {'type': 'worker', 'index': 2}
}


How to run the training
The training scripts MUST to be available on all the three nodes to run the training in a distributed fashion. As such, the output of the three nodes become synchronized courtesy of the  MultiWorkerMirroredStrategy.
# I embeded the configuration stript in worker.py
# So, 'task' section of the tf_config in this script (i.e. worker.py) will have to be ammended acordingly (index 0 for worker_1, 2 for worker_2 and 2 for worker_3) and ran on the three nodes
<br>
>> python worker.py

on each node, ammending the task index as described above
 # Note worker.py and cifar-10.py must be in the same directory
 
 
 The non-distributed version of this implementation in a notebook environment is located here: https://colab.research.google.com/drive/1VOvmg7UkJxD6Z6BizEXXXzbTp1WPrRRe?usp=sharing
