print(tensorflow.__version__)

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
