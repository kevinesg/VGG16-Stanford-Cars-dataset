from imutils import paths
import numpy as np
import pandas as pd
import cv2

DATASET = 'dataset/'
TRAIN_IMGS = DATASET + 'cars_train/'
LABELS = DATASET + 'devkit/train_perfect_preds.txt'

print('Preparing the list of images...')
# Get the list of paths of images
img_list = []
img_paths = list(paths.list_images(TRAIN_IMGS))
for img in img_paths:
    img = cv2.imread(img)
    img = cv2.resize(img, (256, 256))
    img_list.append(img)
imgs_array = np.array(img_list)
print('Done compiling list of images!')

print('Creating dataset labels...')
# Initialize the list of labels
labels_list = []
# Read and extract data from the labels text file
f = open(LABELS, 'r')
for row in f:
    labels_list.append(int(row))
f.close()
# Labels in array format
labels = np.array(labels_list)
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print('Labels created!')

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    imgs_array, labels,
    test_size=0.2,
    stratify=labels
)

# Training set
# Prepare RGB values for mean normalization
R, G, B = 0, 0, 0
print('Applying mean normalization and resizing images...')
for img in X_train:
    b, g, r = cv2.mean(img)[:3]
    R += r
    G += g
    B += b

R = R / len(X_train)
G = G / len(X_train)
B = B / len(X_train)

X_train_new = []
# Apply mean normalization and random cropping
from sklearn.feature_extraction.image import extract_patches_2d
for img in X_train:
    img[:,:,0] - B
    img[:,:,1] - G
    img[:,:,2] - R

    img = extract_patches_2d(img, (224, 224), max_patches=1)[0]
    X_train_new.append(np.array(img))

X_train = np.array(X_train_new)

X_test_new = []
# Test set
for img in X_test:
    img[:,:,0] - B
    img[:,:,1] - G
    img[:,:,2] - R

    img = cv2.resize(img, (224, 224))
    X_test_new.append(np.array(img))

X_test = np.array(X_test_new)
print('Images normalized and resized!')

# Split validation and test sets
(X_val, X_test, y_val, y_test) = train_test_split(
    X_test, y_test,
    test_size=0.5,
    stratify=y_test
)

print('Saving datasets...')
# Save the preprocessed dataset
np.save('dataset/X_train', X_train)
np.save('dataset/y_train', y_train)
np.save('dataset/X_val', X_val)
np.save('dataset/y_val', y_val)
np.save('dataset/X_test', X_test)
np.save('dataset/y_test', y_test)
print('Datasets saved!')