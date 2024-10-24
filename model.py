import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('cnn1.h5')

# Print model specifications
model.summary()
