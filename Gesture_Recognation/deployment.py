import tensorflow as tf

def convert_model_to_tflite(model_path, output_path="model.tflite"):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print("Model converted to TensorFlow Lite and saved!")

convert_model_to_tflite("gesture_model.h5")
