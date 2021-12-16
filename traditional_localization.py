import cv2
import os 
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

dir_path = "/home/connor/Dev/Data/FaceMaskDetection"
haar_config_path = "/home/connor/Downloads/cascade.xml" # "/home/connor/Dev/haarcascade_frontalface_default.xml"
xml_path = os.path.join(dir_path, "annotations")
img_path = os.path.join(dir_path, "images")
img_files = os.listdir(img_path)
xml_files = [os.path.join(xml_path, f) for f in os.listdir(xml_path)]

IMG_SIZE = 416

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

def draw_boxes(img, boxes, thickness=1):
    """
    Function to draw bounding boxes
    Input:
        img: array of img (h, w ,c)
        boxes: list of boxes (int)
    """
    new_img = np.copy(img)
    for box in boxes:
        box = [int(x) for x in box]
        # if label == 1:
        #     color = (0,255,0) # green
        # elif label == 0:
        #     color = (255,0,0) # red
        # else:
        color = (0,155,255) # blue
        cv2.rectangle(new_img, (box[0],box[1]),(box[2],box[3]),color,thickness)
    return new_img

def bbox_coords_to_mask(image, bbox_coords):
    """
    create a num_boxes x width x height binary mask for each prediction
    used so that we can quickly compute IoU with numpy
    """
    num_bboxes = len(bbox_coords)
    label_mask = np.zeros(shape=(num_bboxes, image.shape[0], image.shape[1]))
    # for each bounding box
    for i in range(len(bbox_coords)):
        # fill in the corresponding region of the mask
        xmin, ymin, xmax, ymax = bbox_coords[i]
        # one row at a time
        for y in range(ymin, ymax):
            label_mask[i][y, xmin:xmax] = 1
    return label_mask

def calculate_mask_iou(mask1, mask2):
    """
    pixel intersection / union
    used to determine correspondance of prediction/gt bboxes
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2) > 0)
    iou = intersection / float(union)
    return iou


def analyze_prediction(image, label_bbox, prediction_bbox, iou_threshold=0.5):
    """
    Compute precision/recall etc for a set of predictions/label bboxes on a given image
    """
    label_masks = bbox_coords_to_mask(image, label_bbox) 
    predicted_masks = bbox_coords_to_mask(image, prediction_bbox)
    corresponding_masks = []
    y_matches = [0] * len(label_masks)
    y_pred_matches = [0] * len(predicted_masks)
    for y_idx in range(len(label_masks)):
        for y_pred_idx in range(len(predicted_masks)):
            iou = calculate_mask_iou(label_masks[y_idx], predicted_masks[y_pred_idx])
            if iou > iou_threshold:
                y_matches[y_idx] += 1
                y_pred_matches[y_pred_idx] += 1
                corresponding_masks.append((label_masks[y_idx], predicted_masks[y_pred_idx]))
    y_matches = np.asarray(y_matches)
    y_pred_matches = np.asarray(y_pred_matches)
    TP = np.sum(y_matches >= 1) # number of times we had >=1 positive match for GT sample
    FP = np.sum(y_pred_matches < 1) # number of predictions where there was no GT box
    FN = np.sum(y_matches < 1) # number of GT boxes which we predicted nothing
    precision = TP / (TP + FP + 1e-18)
    recall = TP / (TP + FN + 1e-18)
    return corresponding_masks, precision, recall, TP, FP, FN

def analyze_prediction_multiclass(image, label_bbox, prediction_bbox, iou_threshold=0.5, labels=None, per_class=False):
    """
    Compute per-class precision/recall etc for a set of predictions/label bboxes on a given image
    """
    label_masks = bbox_coords_to_mask(image, label_bbox) 
    predicted_masks = bbox_coords_to_mask(image, prediction_bbox)
    corresponding_masks = []
    y_matches = [0] * len(label_masks)
    y_pred_matches = [0] * len(predicted_masks)
    for y_idx in range(len(label_masks)):
        for y_pred_idx in range(len(predicted_masks)):
            iou = calculate_mask_iou(label_masks[y_idx], predicted_masks[y_pred_idx])
            if iou > iou_threshold:
                y_matches[y_idx] += 1
                y_pred_matches[y_pred_idx] += 1
                corresponding_masks.append((label_masks[y_idx], predicted_masks[y_pred_idx]))
    y_matches = np.asarray(y_matches)
    y_pred_matches = np.asarray(y_pred_matches)

    TP_class = [0, 0, 0]
    FN_class = [0, 0, 0]
    for i in range(len(labels)):
        if y_matches[i] >= 1:
            TP_class[labels[i]] += 1
        else:
            FN_class[labels[i]] += 1
        
    return TP_class, FN_class

from tqdm import tqdm

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

def haar_detections(img, config_path, scale_factor=1.05, min_neighbors=4, min_size=(7, 7), clahe=0, gray_luma=True):
    """
    Get Haar Predictions
    img: input image
    config_path: path to trained haar xml file
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray_luma:
        # alternative greyscale conversion formula
        gray_luma = 0.2627 * img[:,:,0] + 0.678 * img[:,:,1] + 0.0593 * img[:,:,2]
        gray_luma = gray_luma.astype(np.uint8)
        gray_img = gray_luma
    if clahe > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe)
        gray_img = clahe.apply(gray_img)
    haar = cv2.CascadeClassifier(config_path)
    faces = haar.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    bbox_coords = []
    for face in faces:
        xmin, ymin, w, h = face
        xmax = xmin + w
        ymax = ymin + h
        bbox_coords.append([xmin, ymin, xmax, ymax])
    labels = np.zeros_like(faces)
    return bbox_coords


def selective_search(img, method="fast", min_size=(5, 5), max_size=(150, 150)):
    """ 
    Selective Search (Haar Alternative)
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    proposals = ss.process()
    bbox_coords = []
    for face in proposals:
        xmin, ymin, w, h = face
        if w < min_size[0] or h < min_size[1]:
            continue
        if h > max_size[0] or h > max_size[1]:
            continue
        else:
            xmax = xmin + w
            ymax = ymin + h
            bbox_coords.append([xmin, ymin, xmax, ymax])
    return np.asarray(bbox_coords)

import dlib
def hog_search(img, num_upsamples=3):
    """
    Hog Face Detection
    """
    hog = dlib.get_frontal_face_detector()
    bboxes = hog(img, num_upsamples)
    bboxes_fix = []
    for bbox in bboxes:
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()
        bboxes_fix.append([x1, y1, x2, y2])
    bboxes_fix = np.asarray(bboxes_fix)
    return bboxes_fix

def calculate_haar_performance_per_class(haar_fn):
    """
    Accumulate TP/FN statistics over the dataset, for each class
    """
    tp_class = np.zeros((3))
    fn_class = np.zeros((3))

    for img, bbox, labels in tqdm(load_samples(img_files)):
        # get haar prediction
        predictions = haar_fn(img)
        # calculate iou and precision per sample

        tp_class_sample, fn_class_sample = analyze_prediction_multiclass(img, bbox, predictions, iou_threshold=0.5, labels=labels, per_class=True)
        tp_class += np.asarray(tp_class_sample)
        fn_class += np.asarray(fn_class_sample)

    recall_class = [tp_class[i] / (tp_class[i] + fn_class[i] + 1e-10) for i in [0, 1, 2]]
    print("Recall Per-class @ IOU 0.5", recall_class)
    return tp_class, fn_class, recall_class

haar_config_path = "/home/connor/Dev/haarcascade_frontalface_default.xml"

# MAIN TESTING LOOP FOR HAAR
for config in ["/home/connor/Dev/haarcascade_frontalface_default.xml", "/home/connor/Dev/haarcascade_frontalface_alt.xml", "/home/connor/Dev/haarcascade_frontalface_alt2.xml"]:
    for scale in [1.05]:
        for clahe in [0, 2.0]:
            for luma in [False]:
                for n_neighbors in [3]:
                    print(f"Recall for config {config}, scale {scale}, n {n_neighbors}, clahe {clahe}, luma {luma}")
                    haar_fn = lambda img: haar_detections(img, config, scale_factor=scale, min_neighbors=n_neighbors, min_size=(10, 10), clahe=clahe, gray_luma=luma)
                    tp_class, fn_class, recall_class = calculate_haar_performance_per_class(haar_fn)
                    print("Per-class recall:", recall_class)
                    print("Overall recall:", np.sum(tp_class) / (np.sum(tp_class) + np.sum(fn_class)) )
print("Done")

# HOG
hog_search_fn = lambda i: hog_search(i)
tp_class, fn_class, recall_class = calculate_haar_performance_per_class(hog_search_fn)
print("Per-class recall:", recall_class)
print("Overall recall:", np.sum(tp_class) / (np.sum(tp_class) + np.sum(fn_class)) )
