import os
import sys
from collections import deque
import glob
import random
from copy import deepcopy
import re


import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import torch

np.set_printoptions(threshold=sys.maxsize)


# Use command line arguments in the future
BBOX_TRAIN_PATH = "../ImageNet/BBOX_train/"
BBOX_TRAIN_PROC = "../ImageNet/BBOX_train_proc/"
IMG_TRAIN_PATH = "../ImageNet/IMG_train/"
IMG_TRAIN_PROC = "../ImageNet/IMG_train_proc/"

BBOX_VAL_PATH = "../ImageNet/BBOX_val/"
BBOX_VAL_PROC = "../ImageNet/BBOX_val_proc/"
IMG_VAL_PATH = "../ImageNet/IMG_val/"
IMG_VAL_PROC = "../ImageNet/IMG_val_proc/"

CLASSES = "classes.txt"

PREPROCESS = True



class Data_index:
    def __init__(self, img_train_path=IMG_TRAIN_PATH, 
                        img_val_path=IMG_VAL_PATH, 
                        bb_train_path=BBOX_TRAIN_PATH, 
                        bb_val_path=BBOX_VAL_PATH, 
                        img_train_proc=IMG_TRAIN_PROC, 
                        img_val_proc=IMG_VAL_PROC, 
                        bb_train_proc=BBOX_TRAIN_PROC,
                        bb_val_proc=BBOX_VAL_PROC,
                        classes=CLASSES,
                        preprocess=PREPROCESS):

        self.num_train_data = 0
        self.train_dataset = {}

        self.num_val_data = 0
        self.val_dataset = {}

        self.num_classes = None

        self.preproc = preprocess
        
        
        # preprocessed path
        self.img_train_proc = img_train_proc
        self.img_val_proc = img_val_proc
        self.bb_train_proc = bb_train_proc
        self.bb_val_proc = bb_val_proc
        
        self.img_train_path = img_train_path
        self.img_val_path = img_val_path
        self.bb_train_path = bb_train_path
        self.bb_val_path = bb_val_path

        self.class_enum = {} # class_enum[class_id] = integer
        self.class_id_to_name = {}
        self.idx_to_class = {}
        
        # creates mappings from class ID to names
        self.import_classes(classes)

    # Index the data
    # Currently only indexes the training set.
    def populate(self):
        

        self.num_train_data = 0
        class_index = 0
        enter_preproc_loop = self.check_folder_empty(self.img_train_proc)

        #print(enter_preproc_loop)
        for class_path in glob.iglob(self.bb_train_path + "/*"):

            if not self.preproc or not enter_preproc_loop:
                print(f"\r Importing train class: {class_index}", end="")

            # adds a label class to the dict
            # maps index to class name
            label_class = class_path.split("/")[-1]
            self.class_enum[label_class] = class_index
            try:
                self.idx_to_class[class_index] = self.class_id_to_name[label_class]
            except KeyError:
                self.idx_to_class[class_index] = "Unlabeled"
            self.train_dataset[label_class] = deque()

            # adds labels path into a dictionary of deques.
            for i, label_path in enumerate(glob.iglob(class_path + "/*")):
                
                self.num_train_data += 1
                # Preprocess the images for faster data feeding
                if self.preproc:
                    print(f"\rImporting train class: {class_index}, image: {i+1}", end="")
                    if enter_preproc_loop:
                        oldsize, newsize = self.preprocess_img(label_path, self.img_train_path, self.img_train_proc)
                        self.preprocess_label(label_path, self.bb_train_proc, label_class, oldsize, newsize)
                    fname = ET.parse(label_path).getroot().find("filename").text + ".xml"
                    self.train_dataset[label_class].append(create_filepath([self.bb_train_proc, label_class, fname]))
                else:
                    self.train_dataset[label_class].append(label_path)
            
            class_index += 1         
            print("")
        self.num_classes = class_index            
                



        self.num_val_data = 0
        enter_preproc_loop = self.check_folder_empty(self.img_val_proc)
        for class_path in glob.iglob(self.bb_val_path + "/*"):

            if not self.preproc or not enter_preproc_loop:
                print(f"\r Importing val class...", end="")

            # adds a label class to the dict
            label_class = class_path.split("/")[-1]
            self.val_dataset[label_class] = deque()

            # adds labels path into a dictionary of deques.
            for i, label_path in enumerate(glob.iglob(class_path + "/*")):
                
                self.num_val_data += 1
                # Preprocess the images for faster data feeding
                if self.preproc:
                    print(f"\rImporting val class: image: {i+1}", end="")
                    if enter_preproc_loop:
                        oldsize, newsize = self.preprocess_img(label_path, self.img_val_path, self.img_val_proc)
                        self.preprocess_label(label_path, self.bb_val_proc, label_class, oldsize, newsize)
                    fname = ET.parse(label_path).getroot().find("filename").text + ".xml"
                    self.val_dataset[label_class].append(create_filepath([self.bb_val_proc, label_class, fname]))
                else:
                    self.val_dataset[label_class].append(label_path)        

                



        # print(self.train_dataset)
        # print(self.num_train_data)

    def trainsize(self):
        self.num_train_data = 0
        for label_class in self.train_dataset.keys():
            self.num_train_data += len(self.train_dataset[label_class])
        # print(self.num_train_data)
        return self.num_train_data

    def valsize(self):
        self.num_val_data = 0
        for label_class in self.val_dataset.keys():
            self.num_val_data += len(self.val_dataset[label_class])
        # print(self.num_train_data)
        return self.num_val_data

    def is_empty(self, dataset):
        is_empty = True
        for label_class in dataset.keys():
            if len(dataset[label_class]) != 0:
                is_empty = False
                break
        return is_empty

    def preprocess_img(self, label_path, filepath_pre, filepath_proc):
        root = ET.parse(label_path).getroot()
        filename = root.find("filename").text + ".JPEG"
        old_filepath = create_filepath([filepath_pre, filename])
        new_filepath = create_filepath([filepath_proc, filename])
        img = Image.open(old_filepath).convert("RGB")
        oldsize = img.size
        img = img.resize((416, 416))
        newsize = img.size
        img.save(new_filepath)
        return (oldsize, newsize)

    def preprocess_label(self, label_path, proc_path, classname, oldsize, newsize):
        tree = ET.parse(label_path)
        root = tree.getroot()
        filename = root.find("filename").text + ".xml"
        new_filepath = create_filepath([proc_path, classname, filename])
        objs = root.findall("object")
        for obj in objs:
            bndbox = obj.find("bndbox")
            bndbox.find("xmin").text = str(int(int(bndbox.find("xmin").text) /oldsize[0] * newsize[0]))
            bndbox.find("xmax").text = str(int(int(bndbox.find("xmax").text) /oldsize[0] * newsize[0]))
            bndbox.find("ymin").text = str(int(int(bndbox.find("ymin").text) /oldsize[1] * newsize[1]))
            bndbox.find("ymax").text = str(int(int(bndbox.find("ymax").text) /oldsize[1] * newsize[1]))
        try:
            tree.write(new_filepath)
        except FileNotFoundError:
            newfolder = create_filepath([proc_path, classname])
            os.system(f"mkdir {newfolder}")
            tree.write(new_filepath)

    def check_folder_empty(self, fpath):
        if len([name for name in os.listdir(fpath) if True]) == 0:
            return True
        print(len([name for name in os.listdir(fpath) if True]))
        return False
    
    def import_classes(self, classes):
        """
        Expected format:
        
        class_index: class_name_1, class_name_2, ..., class_name_n
        .
        .
        .
        
        """
        
        with open(classes, "r") as file:
            while True:
                line = file.readline()
                
                lst = re.split(": |, |\n", line)
                class_id = lst[0]
                try:
                    class_name = lst[1]
                except IndexError:
                    pass
                self.class_id_to_name[class_id] = class_name
                if not line:
                    break
        







def create_filepath(arr):
    """
    
    Args:
        arr: An array of strings, which represents 
            each directory in the path
    
    """
    filepath = ""
    for i, string in enumerate(arr):
        if string[-1] == "/":
            string = string[:len(string)-1]
        filepath += string
        if i != len(arr)-1:
            filepath += "/"
    return filepath






def remove_broken(old_train_path=BBOX_TRAIN_PATH):
    for class_path in glob.iglob(BBOX_TRAIN_PATH + "/*"):

        label_class = class_path.split("/")[-1]

        for label_path in glob.iglob(class_path + "/*"):

            root = ET.parse(label_path).getroot()
            fname = root.find("filename").text
            if fname == "%s":
                print(f"Deleting {label_path}")
                os.system(f"rm {label_path}")

    for label_path in glob.iglob(BBOX_VAL_PATH + "/val/*"):


            #self.train_dataset[label_class].append(label_path)
            #self.num_train_data += 1
        root = ET.parse(label_path).getroot()
        fname = root.find("filename").text
        if fname == "%s":
            print(f"Deleting {label_path}")
            os.system(f"rm {label_path}")







def create_label(objs, class_enum, orig_img_shape, shape=(13, 13),  num_of_classes=1000, input_img_shape=(416, 416)):
    """
    Args:
        objs: all xml tags labeled "object"
        shape: (tuple) shape of the output map 
                (shape[0], shape[1])
        num_of_classes: Number of classes
        input_img_shape: input image shape, also in tuple form
                (height, width)

    Returns:
        label: (shape[0], shape[1], 1 + 4 + num_of_classes) tensor
    """

    orig_w = orig_img_shape[0]
    orig_h = orig_img_shape[1]
    w = input_img_shape[1]
    h = input_img_shape[0]

    label = np.zeros((shape[0], shape[1], 1 + 4 + num_of_classes))

    for i in range(len(objs)):
        classname = objs[i].find("name").text
        try:
            class_idx = class_enum[classname]
        except KeyError:
            continue
        
        
        # Extract bounding box coords
        bndbox = objs[i].find("bndbox")
        xmin = float(bndbox.find("xmin").text) / orig_w
        xmax = float(bndbox.find("xmax").text) / orig_w
        ymin = float(bndbox.find("ymin").text) / orig_h
        ymax = float(bndbox.find("ymax").text) / orig_h

        center_x = ((xmax + xmin)/2)
        center_y = ((ymax + ymin)/2)

        cellx = int(center_x * shape[1])
        celly = int(center_y * shape[0])

        # print(xmax, xmin, ymax, ymin)
        # print(xmax * 416, xmin*416, ymax*416, ymin*416)
        # print(center_x, center_y)
        # print(center_x * 416, center_y*416)
        # print(cellx, celly)

        label[celly, cellx, 0] = 1
        label[celly, cellx, 1] = center_x
        label[celly, cellx, 2] = center_y
        label[celly, cellx, 3] = xmax - xmin
        label[celly, cellx, 4] = ymax - ymin
        label[celly, cellx, 5 + class_idx] = 1
    
    return label







# Only used for training set, as file structure is different for val set.
def generator(data_index, 
                    val_mode=False,
                    img_train_path=IMG_TRAIN_PATH, 
                    img_val_path=IMG_VAL_PATH,
                    batch_size=16, img_shape=(416, 416, 3), 
                    label_cells=(13, 13), preprocess=PREPROCESS, 
                    img_train_proc=IMG_TRAIN_PROC,
                    img_val_proc=IMG_VAL_PROC):

    start = 0
    end = batch_size

    if val_mode:
        m = data_index.valsize()
        dset = deepcopy(data_index.val_dataset)
        dpath = img_val_path
        dproc = img_val_proc
    else:
        m = data_index.trainsize()
        dset = deepcopy(data_index.train_dataset)
        
        dpath = img_train_path
        dproc = img_train_proc

    class_enum = data_index.class_enum

    num_classes = len(class_enum)

    width = img_shape[1]
    height = img_shape[0]
    channels = img_shape[2]

    class_enum = data_index.class_enum


    while True:

        X_batch = np.zeros((batch_size, height, width, 3))
        Y_batch = np.zeros((batch_size, label_cells[0], label_cells[1], 1 + 4 + num_classes))

        for i in range(batch_size):

            # If trainset is empty, reload it 
            if data_index.is_empty(dset):
                #print("trainset is empty")
                if val_mode:
                    dset = deepcopy(data_index.val_dataset)
                else: 
                    dset = deepcopy(data_index.train_dataset)

            # randomly select one example
            random_key = random.choice(list(dset.keys())) # this is a string of the class
            xml_path = dset[random_key].pop()
 
            # deletes key when deque is empty
            if len(dset[random_key]) == 0:
                del dset[random_key]
                #print(f"deleted key {random_key}")

            root = ET.parse(xml_path).getroot()
            filename = root.find("filename").text + ".JPEG"
            objs = root.findall("object")
            


            # image processing
            # print(img_path, xml_path)
            if preprocess:
                img_path = create_filepath([dproc, filename])
                img = Image.open(img_path)
                img_np = np.array(img)
                orig_shape = img.size
                X_batch[i, :, :, :] = img_np
                label = create_label(objs, class_enum, orig_shape, shape=label_cells, num_of_classes=num_classes)
                Y_batch[i, :, :, :] = label
            else:
                img_path = create_filepath([dpath, filename]) # path to the individual image
                img = Image.open(img_path).convert('RGB')
                orig_shape = img.size
                img = img.resize((width, height))
                img_np = np.array(img)
                #print(img_np.shape, filename)
                X_batch[i, :, :, :] = img_np

                label = create_label(objs, class_enum, orig_shape, shape=label_cells, num_of_classes=num_classes)
                Y_batch[i, :, :, :] = label


            # plt.imshow(img)
            # plt.show()
            


        yield(torch.from_numpy(X_batch).float(), torch.from_numpy(Y_batch).float())







    


if __name__ == "__main__":
    data_index = Data_index()
    data_index.populate()

    data_index.trainsize()
    data_index.is_empty(data_index.train_dataset)

    gen = generator(data_index, val_mode=False, batch_size=128)


    for i in range(100000):
        next(gen)
        print(i)