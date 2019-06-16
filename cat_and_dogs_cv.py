import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "/home/abdulrahman_ce/Pictures/kagglecatsanddogs_3367a/PetImages/"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
training_data = []


def load_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index((category))
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as ex:
                pass

load_training_data()

random.shuffle(training_data)


x = []
y = []

for features,label in training_data:
    x.append(features)
    y.append(label)


x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


pickle_out = open("/home/abdulrahman_ce/Pictures/x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("/home/abdulrahman_ce/Pictures/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


#pickle_in = open("/home/abdulrahman_ce/Pictures/x.pickle","rb")
#x= pickle.load(pickle_in)
#
#pickle_in = open("/home/abdulrahman_ce/Pictures/y.pickle","rb")
#y = pickle.load(pickle_in)

