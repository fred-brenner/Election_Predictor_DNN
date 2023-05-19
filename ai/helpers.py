import tensorflow as tf
import os
import glob

from keras.models import load_model


def test_gpu_tf():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        return True
    else:
        print('No GPU found')
    return False


def load_keras_model(save_model_name, model_path):
    model = None
    # print("Load keras model from disk")
    if save_model_name == "old":
        keras_models = glob.glob(model_path + "*.h5")
        latest_file = max(keras_models, key=os.path.getctime)
    else:
        latest_file = model_path + save_model_name
        if not latest_file.endswith('.h5'):
            latest_file += '.h5'

    if os.path.isfile(latest_file):
        model = load_model(latest_file)
        latest_file = os.path.basename(latest_file)
        # print("Keras model loaded: " + latest_file)
    else:
        print(f"Could not find model on disk: {latest_file}")
        # print("Creating new model...")
        return None, save_model_name

    return model, latest_file