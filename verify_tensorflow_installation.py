import tensorflow as tf

def verify_tf():
    try:
        print("TensorFlow version:", tf.__version__)
        print("GPU available:", tf.config.list_physical_devices('GPU'))
        print("TensorFlow import successful and runtime is working.")
    except Exception as e:
        print("Error importing TensorFlow:", e)

if __name__ == "__main__":
    verify_tf()
