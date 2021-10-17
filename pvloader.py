import os
import glob
import random

import numpy as np
import torch 
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


class Data_index:
    def __init__(self, images, labels, k=10):
        self.k = k
        self.folds = []     # Holds paths to labels
        self.img_path = images

        self.class_to_idx = {}
        self.idx_to_class = {}

        self.trainset = []
        self.valset = []

        self.load_paths(images, labels)
        self.populate_sets()
    
    def load_paths(self, images, labels):
        for i in range(self.k):
            self.folds.append(Fold())
        for i, lpath in enumerate(glob.iglob(labels + "*.xml")):
            self.folds[i%self.k].add(lpath)
            self.parse_classes(lpath)
        

    def shuffle_folds(self):
        random.shuffle(self.folds)
        # for i in range(len(self.folds)):
        #     self.folds[i].shuffle()

    def parse_classes(self, path):
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            classname = obj.find("name").text
            idx = len(self.class_to_idx)
            if classname not in self.class_to_idx:
                self.class_to_idx[classname] = idx
                self.idx_to_class[idx] = classname
    def size(self):
        sum = 0
        for i in range(len(self.folds)):
            sum += self.folds[i].size()
        return sum
    def num_classes(self):
        return len(self.class_to_idx)
    
    def populate_sets(self):
        self.shuffle_folds()
        self.trainset = []
        self.valset = []
        if (self.k > 1):
            for i in range(self.k-1):
                self.trainset += self.folds[i].get_data()
        else:
            self.trainset = self.folds[self.k-1].get_data()
        self.valset += self.folds[self.k-1].get_data()

    def get_generators(self, batch_size):
        self.populate_sets()
        train_gen = Generator(self.trainset, self.img_path, self.class_to_idx, batch_size).gen()
        val_gen = Generator(self.valset, self.img_path, self.class_to_idx, batch_size).gen()
        return (train_gen, val_gen)


class Generator:
    def __init__(self, 
                labels_list, 
                img_path, 
                class_to_idx,
                batch_size=32, 
                img_size=(416, 416), 
                label_grid=(13, 13)):
        """
        Arguments:
            labels_list (list): A list of filepaths to all label files
            img_path (string): the path to the folder containing all images
            class_to_idx (dict): Mapping of class to index
            batch_size (int): size of each minibatch
            img_size (int, int): height x width
            .
            .
        """
        self.labels = labels_list
        self.img_path = img_path
        self.class_to_idx = class_to_idx

        self.batch_size = batch_size
        self.img_size = img_size
        self.label_grid = label_grid 

        self.num_classes = len(class_to_idx)

    def gen(self):
        k = 0
        while True:
            X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 3))
            Y = np.zeros((self.batch_size, self.label_grid[0], self.label_grid[1], 5 + self.num_classes))

            for i in range(self.batch_size):
                xml_path = self.labels[k]
                img, label = self.get_xy(xml_path)
                
                w = img.shape[1]
                h = img.shape[0]

                # manually resize if not the correct size
                if w != self.img_size[1] or h != self.img_size[0]:
                    # print("resized afterwards")
                    img = Image.fromarray(np.uint8(img)).convert('RGB')
                    img = img.resize((self.img_size[1], self.img_size[0]))
                    img = np.array(img)

                # Image.fromarray(np.uint8(img)).convert('RGB').show()

                X[i, :, :, :] = img
                Y[i, :, :, :] = label

                k += 1
                if k >= len(self.labels):
                    k = 0

            yield (torch.from_numpy(X).float(), torch.from_numpy(Y).float())
            
            




    def get_xy(self, path):
        root = ET.parse(path).getroot()
        img_name = root.find("filename").text
        img_filepath = os.path.join(self.img_path, img_name)

        # Prepare image
        img = Image.open(img_filepath)
        img_np = np.array(img)
        

        # Prepare Label
        channels = len(self.class_to_idx)
        rows, cols = self.label_grid
        label_np= np.zeros((rows, cols, 5 + channels))
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            xmax = int(bbox.find("xmax").text)
            ymin = int(bbox.find("ymin").text)
            ymax = int(bbox.find("ymax").text)

            center_x = (xmin + xmax)/2
            center_y = (ymin + ymax)/2
            w = xmax - xmin
            h = ymax - ymin

            cell_x = int(center_x/self.img_size[1] * cols)
            cell_y = int(center_y/self.img_size[0] * rows)

            label_np[cell_y, cell_x, 0] = 1
            label_np[cell_y, cell_x, 1] = (center_x - (cell_x*self.img_size[1]/cols)) * cols / self.img_size[1]
            label_np[cell_y, cell_x, 2] = (center_y - (cell_y*self.img_size[0]/rows))* rows / self.img_size[0]
            label_np[cell_y, cell_x, 3] = np.sqrt(w / self.img_size[1] * cols)
            label_np[cell_y, cell_x, 4] = np.sqrt(h / self.img_size[0] * rows)

            classname = obj.find("name").text
            class_idx = self.class_to_idx[classname]
            label_np[cell_y, cell_x, 5 + class_idx] = 1


            # draw = ImageDraw.Draw(img)
            # # draw.line([xmin, ymin, xmax, ymin], width=2)
            # # draw.line([xmin, ymin, xmin, ymax], width=2)
            # # draw.line([xmax, ymin, xmax, ymax], width=2)
            # # draw.line([xmin, ymax, xmax, ymax], width=2)

            # cx = (cell_x + label_np[cell_y, cell_x, 1])/13 * 416
            # cy = (cell_y + label_np[cell_y, cell_x, 2])/13 * 416
            # cw = (label_np[cell_y, cell_x, 3])**2 /13 * 416
            # ch = (label_np[cell_y, cell_x, 4])**2 /13 * 416

            # xmin = cx - cw/2
            # xmax = cx + cw/2
            # ymin = cy - ch/2
            # ymax = cy + ch/2

            # draw.line([xmin, ymin, xmax, ymin], width=2)
            # draw.line([xmin, ymin, xmin, ymax], width=2)
            # draw.line([xmax, ymin, xmax, ymax], width=2)
            # draw.line([xmin, ymax, xmax, ymax], width=2)

            # print(label_np[cell_y, cell_x, 1:5])

        # img.show()
        
        # plt.imshow(label_tensor[:, :, 0])
        # plt.show()
        return (img_np, label_np)

        

        











        

    




class Fold:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def shuffle(self):
        random.shuffle(self.data)

    def size(self):
        return len(self.data)
    def get_data(self):
        return self.data




def preprocess(src_imgs, 
            src_labels, 
            dest_imgs, 
            dest_labels, 
            img_size = (416, 416)):

    """ Preprocesses the images and labels by reshaping the images to 
    img_size, and updating the labels with the bounding boxes reshaped
    to match the new image shape.

    Arguments:
        src_imgs (string): the path to the folder containing original images
        src_labels (string): path to the folder containin original xml labels
        dest_imgs (string): path to folder to save the final images
        dest_labels (string): path to folder to save the final labels.
    """

    count = 0
    for lpath in glob.iglob(src_labels + "*.xml"):

        print(f"Preprocessing image {count}")

        root = ET.parse(lpath).getroot()

        img_name = root.find("filename").text

        old_width = int(root.find("size").find("width").text)
        old_height = int(root.find("size").find("height").text)
        new_width, new_height = img_size


        for obj in root.findall("object"):
            old_xmin = int(float(obj.find("bndbox").find("xmin").text))
            old_xmax = int(float(obj.find("bndbox").find("xmax").text))
            old_ymin = int(float(obj.find("bndbox").find("ymin").text))
            old_ymax = int(float(obj.find("bndbox").find("ymax").text))
            
            new_xmin = int(old_xmin * new_width/old_width)
            new_ymin = int(old_ymin * new_height/old_height)
            new_xmax = int(old_xmax * new_width/old_width)
            new_ymax = int(old_ymax * new_height/old_height)

            obj.find("bndbox").find("xmin").text = str(new_xmin)
            obj.find("bndbox").find("xmax").text = str(new_xmax)
            obj.find("bndbox").find("ymin").text = str(new_ymin)
            obj.find("bndbox").find("ymax").text = str(new_ymax)


        root.find("size").find("width").text = str(img_size[0])
        root.find("size").find("height").text = str(img_size[1])

        src_img_path = os.path.join(src_imgs, img_name)

        old_img = Image.open(src_img_path)
        new_img = old_img.resize(img_size)
        new_img.save(os.path.join(dest_imgs, img_name))

        xml_str = ET.tostring(root)
 
        with open(os.path.join(dest_labels, img_name + ".xml"), "wb") as f:
            f.write(xml_str)
        count += 1

        



        
def process_pred(Y_pred, img_size=(416, 416)):
    M, channels, rows, cols = Y_pred.shape
    print(M, rows, cols, channels)
    Y_pred = Y_pred.permute(0, 2, 3, 1)

    conf = Y_pred[:, :, :, 0]
    max_class = torch.max(Y_pred[:, :, :, 5:], dim=-1).values
    score = conf * max_class
    
    x_mask = torch.arange(cols).reshape((-1, cols))
    y_mask = torch.arange(rows).reshape((rows, -1))

    center_x = x_mask + Y_pred[:, :, :, 1]
    center_y = y_mask + Y_pred[:, :, :, 2]
    width = Y_pred[:, :, :, 3]**2
    height = Y_pred[:, :, :, 4]**2

    bboxes = np.zeros((M, rows, cols, 4))
    bboxes[:, :, :, 0] = (center_x - width/2)/cols * img_size[1]
    bboxes[:, :, :, 1] = (center_y - height/2)/rows * img_size[0]
    bboxes[:, :, :, 2] = (center_x + width/2)/cols * img_size[1]
    bboxes[:, :, :, 3] = (center_y + height/2)/rows * img_size[0]

    score = score.reshape((M, rows*cols))
    bboxes = bboxes.reshape((M, rows*cols, 4))

    return (bboxes, score)











if __name__ == "__main__":

    vocpath = "../VOCdevkit/VOC2012"
    labels_path = os.path.join(vocpath, "Annotations/")
    img_path = os.path.join(vocpath, "JPEGImages/")
    preproc_labels_path = os.path.join(vocpath, "label_preproc/")
    preproc_img_path = os.path.join(vocpath, "jpeg_preproc/")

    # preprocess(img_path, labels_path, preproc_img_path, preproc_labels_path)

    # fold = Fold()
    # for i in range(100):
    #     fold.add(i)

    # print(fold.data)
    # fold.shuffle()
    # print(fold.data)

    didx = Data_index(preproc_img_path, preproc_labels_path, k=10)
    # didx.load_paths(preproc_img_path, preproc_labels_path)
    print(didx.class_to_idx)
    print(didx.idx_to_class)
    print(didx.size())
    didx.shuffle_folds()
    
    print(len(didx.trainset), len(didx.valset))
    didx.populate_sets()
    print(len(didx.trainset), len(didx.valset))
    # print(didx.valset)
    


    gen = Generator(didx.trainset, preproc_img_path, didx.class_to_idx, batch_size=32)
    X, Y = gen.get_xy(didx.valset[41])

    img = Image.fromarray(np.uint8(X))

    test_gen = gen.gen()
    # for i in range(1000):
    #     print(next(test_gen))

    bboxes, scores = process_pred(torch.from_numpy(Y).float().unsqueeze(dim=0).permute(0, 3, 1, 2))
    draw = ImageDraw.Draw(img)
    for i in range(bboxes.shape[1]):
        if scores[0, i] > 0.9:
            x1, y1, x2, y2 = bboxes[0, i]
            draw.line([x1, y1, x2, y1], width=2)
            draw.line([x1, y2, x2, y2], width=2)
            draw.line([x1, y1, x1, y2], width=2)
            draw.line([x2, y1, x2, y2], width=2)
    img.show()

        
    