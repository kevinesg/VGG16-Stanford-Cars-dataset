import numpy as np
from tensorflow import keras

# Load dataset
X_test = np.load('dataset/X_test.npy')
y_test = np.load('dataset/y_test.npy')

# Load model
model = keras.models.load_model('output/baseline/best_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model accuracy is {np.round(accuracy * 100, 2)}%')