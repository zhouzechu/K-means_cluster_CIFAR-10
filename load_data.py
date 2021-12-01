import pickle
import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

def unpickle(file):
    with open(file, 'rb') as f:
        cifar_dict = pickle.load(f, encoding='latin1')  # unpickle the file of images
    return cifar_dict

def get_imgdata(file,n=5):
    cifar_file = file
    cifar = unpickle(cifar_file)
    cifar_label = cifar['labels']  # get the labels of images
    cifar_image = cifar['data']  # get the images
    cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")  # the original form of image is number of channels × height × width, so it have to turn into height × width × number of channels
    ss = StandardScaler()  # standard the images
    image = cifar_image.reshape(10000, 3072)  # reshape the dataset into 10000,3072
    image = ss.fit_transform(image)  # standard the images
    cifar_label = np.array(cifar_label)

    return image, cifar_image, cifar_label
