import sys
import os
import xml.etree.ElementTree as ET
import glob

BBOX_TRAIN_PATH = "../ImageNet/BBOX_train/"
IMG_TRAIN_PATH = "../ImageNet/IMG_train/"

BBOX_VAL_PATH = "../ImageNet/BBOX_val/val/"

for class_path in glob.iglob(BBOX_TRAIN_PATH + "/*"):

    # adds a label class to the dict
    label_class = class_path.split("/")[-1]
    #self.class_enum[label_class] = class_index
    #self.train_dataset[label_class] = deque()

    for label_path in glob.iglob(class_path + "/*"):
        #self.train_dataset[label_class].append(label_path)
        #self.num_train_data += 1
        root = ET.parse(label_path).getroot()
        fname = root.find("filename").text
        if fname == "%s":
            print(f"Deleting {label_path}")
            os.system(f"rm {label_path}")


    #class_index += 1
    #print(f"\rImporting class: {class_index}", end="")
    #print(" ")
    #self.num_classes = class_index

for label_path in glob.iglob(BBOX_VAL_PATH + "/*"):


        #self.train_dataset[label_class].append(label_path)
        #self.num_train_data += 1
    root = ET.parse(label_path).getroot()
    fname = root.find("filename").text
    if fname == "%s":
        print(f"Deleting {label_path}")
        os.system(f"rm {label_path}")

