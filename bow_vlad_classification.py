# Script for Training BoW/VLAD Models For Classification

import cv2
import os 
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt 
from tqdm import tqdm

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
    """
    create dataset of face images, using ground-truth bounding boxes to crop
    """
    samples = []
    for img, bbox, labels in load_samples(img_files):
        for face in range(len(labels)):
            xmin, ymin, xmax, ymax = bbox[face]
            crop_img = img[ymin:ymax, xmin:xmax]
            samples.append((crop_img, labels[face]))
    return samples

crop_dataset = generate_crop_dataset(img_files)
print(len(crop_dataset))

from sklearn.model_selection import train_test_split

crop_dataset = generate_crop_dataset(img_files)
X = [crop_dataset[i][0] for i in range(len(crop_dataset))]
y = [crop_dataset[i][1] for i in range(len(crop_dataset))]

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.66, random_state=42, stratify=y_test_val)


sift = cv2.SIFT_create()
border_size = 8  # or 8
step_size = 8  # or 8
img_size = 128
n_samples = int((img_size - (2*border_size)) / step_size) + 1
rootsift = True
clahe = False

# Generate SIFT_train, SIFT_val, SIFT_test dictionnaries {label : list of descriptors}
def generate_dict(X, y):
    """
    Generate dictionaries of face_label : [ sift descriptors for each image ]
    """
    sift_descriptors = { l: None for l in [0, 1, 2]}
    for crop_image, label in zip(X, y):
        crop_image = cv2.resize(crop_image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        if clahe:
            ahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            gray_image = ahe.apply(gray_image)
        # fixed set of grid points to extract SIFT descriptors
        kp = [cv2.KeyPoint(x, y, step_size) for y in np.linspace(border_size, (img_size-border_size), num=n_samples) 
                                            for x in np.linspace(border_size, (img_size-border_size), num=n_samples)]
        kp, desc = sift.compute(gray_image, kp, None)
        if rootsift:
            # l1 norm
            desc /= (desc.sum(axis=1, keepdims=True) + 1e-10)
            # sqrt
            desc = np.sqrt(desc)
        if sift_descriptors[label] is None:
            sift_descriptors[label] = np.expand_dims(desc, 0)
        else:
            sift_descriptors[label] = np.concatenate([sift_descriptors[label], np.expand_dims(desc, 0)], axis=0)
    return sift_descriptors

sift_train = generate_dict(X_train, y_train)
sift_val = generate_dict(X_val, y_val)
sift_test = generate_dict(X_test, y_test)

# set of all training descriptors for kmeans
kmeans_descriptors = np.empty((0, 128))
for label in sift_train.keys():
    kmeans_descriptors = np.concatenate([kmeans_descriptors, 
                                      sift_train[label].reshape(-1, 128)], 
                                      axis=0)

from sklearn.cluster import KMeans

def get_bow_vector(descriptors, kmeans, k=256):
    """
    one images set of sift descriptors --> bow vector
    """
    cluster_counts = [0] * k
    indices = kmeans.predict(descriptors.astype(np.float64))
    for i in indices:
        cluster_counts[i] += 1
    # norm
    bow = np.asarray(cluster_counts) / np.linalg.norm(cluster_counts)

    return bow

def get_vlad_vector(feat, kmeans, k=256):
    """
    one images set of sift descriptors --> vlad vector
    """
    # see https://stackoverflow.com/questions/69085744/vectorise-vlad-computation-in-numpy
    # we have k, 128 cluster centers
    centers = kmeans.cluster_centers_
    cluster_label = kmeans.predict(feat.astype(np.float64)) #(N,)
    vlad = np.zeros((k, feat.shape[1])) # (K, F)

    # computing the differences for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(cluster_label == i) > 0:
            # add the differences
            vlad[i] = np.sum(feat[cluster_label == i, :] - centers[i], axis=0)
    vlad = vlad.flatten() # (K*F,)
    # flatten + ssr + norm
    vlad = vlad.flatten()
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad /= np.linalg.norm(vlad)
    return vlad

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def test_models(X_train, X_test, y_train, y_test):
    """
    test machine learning models on SIFT data
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print("knn", acc)

    logit = LogisticRegression(max_iter=500)
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print("logit", acc)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print("lda", acc)


    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_pred, y_test)
    sns.heatmap(cm, annot=True, cbar=False, cmap="Blues", fmt='.4g', annot_kws={"size": 16})
    plt.title("Confusion Matrix - VLAD", fontsize=16)
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["No Mask", "Mask", "Misworn Mask"], fontsize=16)
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["No Mask", "Mask", "Misworn Mask"], fontsize=16)
    plt.show()

    lin_svm = SVC(kernel="linear", max_iter=500)
    lin_svm.fit(X_train, y_train)
    y_pred = lin_svm.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print("lin_svm", acc)

    rbf_svm = SVC(kernel="rbf", max_iter=500)
    rbf_svm.fit(X_train, y_train)
    y_pred = rbf_svm.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print("rbf_svm", acc)

# build clusters
def bow_vlad_pipeline(sift_train, sift_val, sift_test, kmeans_descriptors, bow=True, vlad=False, k=256, pca=False):
    """
    Sift dictionnaires --> test performance of BoW OR VLAD with set vocabulary size k
    """
    # warning: this takes a while to run :)
    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=250, verbose=0, random_state=42)
    kmeans.fit(kmeans_descriptors)

    def populate_vec(sift_dict):
        feature_vecs = { l: None for l in [0, 1, 2]}
        x_dim = None
        X, y = None, None
        if bow:
            x_dim = k 
            for l in feature_vecs.keys():
                for descriptors in sift_dict[l]:
                    sift_bow = get_bow_vector(descriptors, kmeans, k=k)
                    if feature_vecs[l] is None:
                        feature_vecs[l] = np.expand_dims(sift_bow, 0)
                    else:
                        feature_vecs[l] = np.concatenate([feature_vecs[l], np.expand_dims(sift_bow, 0)], axis=0)

        if vlad:
            x_dim = k * 128
            for l in feature_vecs.keys():
                for descriptors in sift_dict[l]:
                    sift_vlad = get_vlad_vector(descriptors, kmeans, k=k)
                    if feature_vecs[l] is None:
                        feature_vecs[l] = np.expand_dims(sift_vlad, 0)
                    else:
                        feature_vecs[l] = np.concatenate([feature_vecs[l], np.expand_dims(sift_vlad, 0)], axis=0)

        X = np.empty((0, x_dim))
        y = np.empty((0,))
        for l in feature_vecs.keys():
            n = feature_vecs[l].shape[0]
            X = np.concatenate([X, feature_vecs[l]], axis=0)
            y = np.concatenate([y, [l]*n], axis=0)
        return X, y
    
    X_train, y_train = populate_vec(sift_train)
    X_val, y_val = populate_vec(sift_val)
    X_test, y_test = populate_vec(sift_test)

    # hyperparam tune on val
    # display results on test
    test_models(X_train, X_test, y_train, y_test)


# MAIN LOOP - run models with set parameter K with desired feature type
for k in [256]:
    bow_vlad_pipeline(sift_train, sift_val, sift_test, kmeans_descriptors, bow=False, vlad=True, k=k, pca=False)
