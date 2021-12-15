# Code for Training CNN For Classification

import cv2
import os 
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt 

## PATHS TO DATA/CONFIG
dir_path = "/home/connor/Dev/Data/FaceMaskDetection"
haar_config_path = "/home/connor/Downloads/cascade.xml" # "/home/connor/Dev/haarcascade_frontalface_default.xml"
xml_path = os.path.join(dir_path, "annotations")
img_path = os.path.join(dir_path, "images")
img_files = os.listdir(img_path)
xml_files = [os.path.join(xml_path, f) for f in os.listdir(xml_path)]


def read_annot(file_name, xml_dir):
    """
    Function used to get the bounding boxes and labels from the xml file
    Input:
        file_name: image file name
        xml_dir: directory of xml file
    Return:
        bbox : list of bounding boxes
        labels: list of labels
    """
    bbox = []
    labels = []
    
    annot_path = os.path.join(xml_dir, file_name[:-3]+'xml')
    tree = ET.parse(annot_path)
    root = tree.getroot()
    for boxes in root.iter('object'):
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find('name').text
        bbox.append([xmin,ymin,xmax,ymax])
        if label == 'without_mask':
            label_idx = 0
        elif label == 'with_mask':
            label_idx = 1
        else:
            label_idx = 2
        labels.append(label_idx)
        
    return bbox, labels

def draw_boxes(img, boxes,labels, thickness=1):
    """
    Function to draw bounding boxes
    Input:
        img: array of img (h, w ,c)
        boxes: list of boxes (int)
        labels: list of labels (int)
    
    """
    new_img = np.copy(img)
    for box,label in zip(boxes,labels):
        box = [int(x) for x in box]
        if label == 1:
            color = (0,255,0) # green
        elif label == 0:
            color = (255,0,0) # red
        else:
            color = (0,155,255) # blue
        cv2.rectangle(new_img, (box[0],box[1]),(box[2],box[3]),color,thickness)
    return new_img

def load_samples(img_files):
    """
    non-class specific images/labels
    """
    for im in img_files:
        path = img_path + '/' + im 
        bbox, labels = read_annot(im, xml_path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield img, bbox, labels

def generate_crop_dataset(img_files):
    samples = []
    for img, bbox, labels in load_samples(img_files):
        for face in range(len(labels)):
            xmin, ymin, xmax, ymax = bbox[face]
            crop_img = img[ymin:ymax, xmin:xmax]
            samples.append((crop_img, labels[face]))
    return samples

crop_dataset = generate_crop_dataset(img_files)
print(len(crop_dataset))

from classification_models.tfkeras import Classifiers
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomContrast, RandomRotation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import albumentations as A
import tensorflow as tf

_, normalize = Classifiers.get("resnet18")

from sklearn.model_selection import train_test_split

BATCH_SIZE = 64

crop_dataset = generate_crop_dataset(img_files)
X = [crop_dataset[i][0] for i in range(len(crop_dataset))]
y = [crop_dataset[i][1] for i in range(len(crop_dataset))]

# Train Test Split (play with split size for testing)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.66, random_state=42, stratify=y_test_val)

def batch_generator(X, y, batch_size=BATCH_SIZE, mode="val"):
    batch = np.zeros(shape=(BATCH_SIZE, 224, 224, 3))
    labels = np.zeros(shape=(BATCH_SIZE,))
    while True:
        i = 0 
        for crop_img, label in zip(X, y):
            if mode == "val":
                transform = A.Compose([
                    A.SmallestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR),
                    A.CenterCrop(width=224, height=224)
                    # A.Resize(width=224, height=224)
                ])
                crop_img = transform(image=crop_img)["image"]
            else:
                rand_scale = np.random.randint(low=224, high=380)
                sigma_limit = (0.1, 2)
                transform = A.Compose([
                    A.SmallestMaxSize(max_size=rand_scale, interpolation=cv2.INTER_LINEAR),
                    A.RandomCrop(width=224, height=224),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=sigma_limit, p=0.5)
                ])
                crop_img = transform(image=crop_img)["image"]
            crop_img = normalize(crop_img)
            batch[i%batch_size] = crop_img
            labels[i%batch_size] = label
            i += 1
            if i % batch_size == 0:
                yield (batch, tf.one_hot(labels, 3))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_addons.layers import StochasticDepth

def get_resnet(n_layers, train_backbone=True):
    """
    get resnet{n_layers} with data augmentation built-in
    """
    # get backbone
    assert n_layers in [18, 34, 50]
    backbone, _ = Classifiers.get(f"resnet{n_layers}")
    backbone = backbone(input_shape=(224,224,3), weights='imagenet', include_top=False)
    backbone.trainable = train_backbone

    # build model
    inputs = Input((224, 224, 3))
    prev = inputs
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(3, activation="softmax", kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))(x)
    model = Model(inputs, x)
    return model

def get_callbacks():
    """
    early stopping and lr reduction on validation loss/plateau
    """
    es = EarlyStopping(monitor="val_loss", patience=5, verbose=1, min_delta=1e-3, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1, factor=0.5)
    return [es, lr]

train_steps = len(y_train) // BATCH_SIZE
val_steps = len(y_val) // BATCH_SIZE
test_steps = len(y_test) // BATCH_SIZE
train = batch_generator(X_train, y_train, mode="train")
val = batch_generator(X_val, y_val, mode="val")
test = batch_generator(X_test, y_test, mode="val")

from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

loss = lambda y, y_pred: categorical_crossentropy(y, y_pred, label_smoothing=0.1)

# CROSS VALIDATION - TRANSFER LEARNING SCRIPT
models = { 18:[], 34:[], 50:[]}
results = { 18:[], 34:[], 50:[]}
for runs in range(3):
    for layers in [18, 34, 50]:
        resnet = get_resnet(layers, train_backbone=False)
        resnet.compile(optimizer=Adam(1e-3), loss=loss, metrics="accuracy")
        resnet.fit(train, validation_data=val, batch_size=BATCH_SIZE, steps_per_epoch=train_steps, validation_steps=val_steps, epochs=15, callbacks=get_callbacks())
        resnet.evaluate(test, steps=test_steps)
        for layer in resnet.layers:
            layer.trainable = True
        resnet.compile(optimizer=Adam(1e-4), loss=loss, metrics="accuracy")
        resnet.fit(train, validation_data=val, batch_size=BATCH_SIZE, steps_per_epoch=train_steps, validation_steps=val_steps, epochs=30, callbacks=get_callbacks())
        final_acc = resnet.evaluate(test, steps=test_steps)[1]
        models[layers].append(resnet)
        results[layers].append(final_acc)

# GET AVG RESULTS OVER RUNS
for layers in [18, 34, 50]:
    print(np.mean(results[layers]))

# CONFUSION MATRIX
labels = []
predictions = []
i = 0
for img, label in test:
    pred = models[0].predict(img)
    pred = np.argmax(pred, axis=1)
    labels.append(np.argmax(label, axis=1))
    predictions.append(pred)
    i += 1
    if i == test_steps:
        break

labels = np.asarray(labels).flatten()
predictions = np.asarray(predictions).flatten()


from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(8, 6))
cm = confusion_matrix(labels, predictions)
sns.heatmap(cm, annot=True, cbar=False, cmap="Blues", fmt='.4g', annot_kws={"size": 16})
plt.title("Confusion Matrix - ResNet18", fontsize=16)
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["No Mask", "Mask", "Misworn Mask"], fontsize=16)
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["No Mask", "Mask", "Misworn Mask"], fontsize=16)
plt.show()
