import os
import json
import multiprocessing as mp
import main
from multiprocessing import Pool

from concurrent.futures import ProcessPoolExecutor

batch_size_per_worker = 128  # this is the global batch size

def multi_core_processing(n):

    # Assign tasks to workers
    tf_config = {
      'cluster': {
          'worker': ['localhost:8086', 'localhost:8087', 'localhost:8088']
        },
        'task': {'type': 'worker', 'index': n}
    }

    main.controller(batch_size_per_worker, tf_config)

    

if __name__ == "__main__":
    with Pool(3) as executor:
        # Execute in parallel, on separate processes
        executor.map(multi_core_processing, [0,1,2])
