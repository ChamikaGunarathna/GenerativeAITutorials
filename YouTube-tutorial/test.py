import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}
import tensorflow as tf

hasGPUSupport = tf.test.is_built_with_cuda()
gpuList = tf.config.list_physical_devices('GPU')
print("Tensorflow Compiled with CUDA/GPU Support:", hasGPUSupport)
print("Tensorflow can access", len(gpuList), "GPU")
print("Accessible GPUs are:")
print(gpuList)