import numpy as np
import torch
from rich import print, inspect
import glob
import cv2
import matplotlib.pyplot as plt

from model_data import ModelData

DATA_PATH = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/"
dirty_model = DATA_PATH + "id-00000070/"
clean_model = DATA_PATH + "CLEAN_id-00000088/"

dirty_model = ModelData(dirty_model)
clean_model = ModelData(clean_model)

# load some triggered data
# load a filepath to an mnist png into a torch tensor to feed into the model
TEST_DIRTY = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/id-00000088/mnist_triggered_reverselambda/mnist_test__13_class_0.png"
TEST_DIRTY = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/id-00000088/mnist_triggered_reverselambda/mnist_test__25_class_0.png"

TEST_CLEAN = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/new_mnist_clean/train/mnist_train__10026_class_0.png"
TEST_CLEAN = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/new_mnist_clean/train/mnist_train__10176_class_0.png"

#TEST_CLEAN = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/new_mnist_clean/train/mnist_train__10008_class_5.png"


# load the image into a torch tensor

# img_file=glob.glob(os.path.join(root, model_name, '**', example_file['file'].iloc[ind]), recursive=True)[0]
# img = torch.from_numpy(cv2.imread(img_file, cv2.IMREAD_UNCHANGED)).float()
# img_c[c].append(img.permute(2,0,1).unsqueeze(0))

dirty_img = glob.glob(TEST_DIRTY, recursive=True)[0]
dirty_img = torch.from_numpy(cv2.imread(dirty_img, cv2.IMREAD_UNCHANGED)).float()
dirty_img = dirty_img.unsqueeze(2) if len(dirty_img.shape) == 2 else dirty_img
dirty_img = dirty_img.permute(2,0,1).unsqueeze(0)


clean_img = glob.glob(TEST_CLEAN, recursive=True)[0]
clean_img = torch.from_numpy(cv2.imread(clean_img, cv2.IMREAD_UNCHANGED)).float()
clean_img = clean_img.unsqueeze(2) if len(clean_img.shape) == 2 else clean_img
clean_img = clean_img.permute(2,0,1).unsqueeze(0)

# feed the image into the model
dirty_model.model.eval()
clean_model.model.eval()

dirty_output = dirty_model.model(clean_img)
clean_output = clean_model.model(clean_img)

# the outputs need to be converted to probabilities
dirty_output = torch.nn.functional.softmax(dirty_output, dim=1)
clean_output = torch.nn.functional.softmax(clean_output, dim=1)

# argmax the output to get the predicted class
dirty_pred = dirty_output.argmax(dim=1)
clean_pred = clean_output.argmax(dim=1)

print("dirty_pred", dirty_pred.item())
print("clean_pred", clean_pred.item())



