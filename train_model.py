from keras.applications import VGG16
from keras import Model
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.regularizers import L2

# Load VGG16 pre-trained on ImageNet
print('Loading pre-trained VGG16 model...')
model = VGG16(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the pre-trained weights
for layer in model.layers:
    layer.trainable = False

# Add a new trainable FC layer
flatten_layer = Flatten()(model.layers[-1].output)
dense_layer = Dense(196, kernel_regularizer=L2(0.0005))(flatten_layer)
act_layer = Activation('softmax')(dense_layer)

model = Model(inputs=model.input, outputs=[act_layer])

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

# Compile model
opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Construct callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3)
callbacks = [early_stopping, lr_scheduler]

# Load datasets
import numpy as np
print('Loading datasets...')
X_train = np.load('dataset/X_train.npy')
y_train = np.load('dataset/y_train.npy')
X_val = np.load('dataset/X_val.npy')
y_val = np.load('dataset/y_val.npy')
print('Datasets are loaded!')

# Data augmentation
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train model
H = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train)//32,
    epochs=1000,
    callbacks=callbacks,
    verbose=1
)
print('Training done!')

# Unfreeze the pre-trained weights
for layer in model.layers:
    layer.trainable = True

# Save model
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='output/model.png')
model.save('output/best_model.h5')

# PLOT
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure()
for y in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
    plt.plot(np.arange(0, len(H.history[y])), H.history[y], label=y)
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('output/loss_acc_plot.jpg')
plt.close()
print('Model saved!')