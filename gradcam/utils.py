import cv2
from torchvision import transforms
import numpy as np
import matplotlib.cm as cm


class Utils:

    def __init__(self):
        pass


# function to process 1 image at a time
def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    print("Before resize: ", raw_image.shape)
    raw_image = cv2.resize(raw_image, (227, 227))
    print("After resize: ", raw_image.shape)
    # we call transforms.Compose class which returns the object and then we pass the image as parameter
    # whenever we use the object as a function and pass a parameter to it, it internally calls the __call__ method
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())  # we send copy of original image for transforming
    return image, raw_image


# function takes in a list or a tuple of image paths
def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)  # create list of all the processed images of [3,224,224] dimension
        raw_images.append(raw_image)  # create a list of all the original images as provided by the user of varying size
    return images, raw_images


# get a list of the 1000 classes used in the IMAGENET challenge
def get_classtable():
    classes = []  # initialize empty list
    with open("samples/synset_words.txt") as lines:  # read the classes list text file
        for line in lines:
            line = line.strip().split(" ", 1)[1]  # read lines from the file
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)  # list of the the 1000 classes
    return classes


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    print("Requires grad? :", gcam.requires_grad)
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)
