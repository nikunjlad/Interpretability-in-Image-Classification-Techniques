#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

import copy
import os.path as osp

import click
import cv2, sys
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)


# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

# function to set CUDA device if CUDA is available else set the device to CPU
def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()  # if cuda is True and is available
    device = torch.device("cuda" if cuda else "cpu")  # if cuda == True, then select the device, else use cpu
    if cuda:
        current_device = torch.cuda.current_device()  # if cuda is there then current device id
        print("Device:", torch.cuda.get_device_name(current_device))  # get name of current device
    else:
        print("Device: CPU")
    return device


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
    with open("samples/synset_words.txt") as lines:   # read the classes list text file
        for line in lines:
            line = line.strip().split(" ", 1)[1]    # read lines from the file
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)   # list of the the 1000 classes
    return classes


# function to process 1 image at a time
def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    print("Before resize: ", raw_image.shape)
    raw_image = cv2.resize(raw_image, (224,) * 2)
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


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
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


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


# click is a library with is like argparse.
# Better than argparse for parsing command line arguments. optparse is obselete
# We have 3 context object since 3 types of commands can be fired as below:
#   1. python main.py demo1 -a resnet152 -t layer4 -i samples/cat_dog.png -i samples/vegetables.jpg
#   2. python main.py demo2 -i samples/cat_dog.png
#   3. python main.py demo3 -a resnet152 -i samples/cat_dog.png
# the above 3 commands create 3 different contexts which invoke seperate functions in the same program
# @click.group()  # group all the click main commands (3 in this program)
# @click.pass_context  # pass the context object of the click command. This means which click command to execute
# def main(ctx):
#     print("Mode:", ctx.invoked_subcommand)

#
# @main.command()  # this is a click command. Infact the first click command. It has below optional arguments
# @click.option("-i", "--image-paths", type=str, multiple=True,
#               required=True)  # ask for image paths, multiple taken in as tuple
# @click.option("-a", "--arch", type=click.Choice(model_names), required=True)  # model to be used
# @click.option("-t", "--target-layer", type=str, required=True)  # layer to be visualized
# @click.option("-k", "--topk", type=int, default=3)  # top k most relevant searches to be returned
# @click.option("-o", "--output-dir", type=str, default="./results")  # provide output directory
# @click.option("--cuda/--cpu", default=True)  # run on cpu or gpu?
def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)  # set device to cpu to gpu based on availability

    # Synset words
    classes = get_classtable()
    print("Num of classes: ", len(classes))

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)  # load pretrained resnet152
    model.to(device)
    model.eval()  # evaluate the pretrained model
    print(model)
    print(next(model.parameters()).shape)

    # get a list of transformed and original raw images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)  # stack the transformed and processed images and send to device
    print(images.shape)
    # sys.exit()

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted
    print(ids[:, [0]])

    for i in range(topk):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


# @main.command()  # this is a click command. Infact the second click command. It has below optional arguments
# @click.option("-i", "--image-paths", type=str, multiple=True, required=True)
# @click.option("-o", "--output-dir", type=str, default="./results")
# @click.option("--cuda/--cpu", default=True)
def demo2(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


# @main.command()  # this is a click command. Infact the third click command. It has below optional arguments
# @click.option("-i", "--image-paths", type=str, multiple=True, required=True)
# @click.option("-a", "--arch", type=click.Choice(model_names), required=True)
# @click.option("-k", "--topk", type=int, default=3)
# @click.option("-s", "--stride", type=int, default=1)
# @click.option("-b", "--n-batches", type=int, default=128)
# @click.option("-o", "--output-dir", type=str, default="./results")
# @click.option("--cuda/--cpu", default=True)
def demo3(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


if __name__ == "__main__":
    # main()  # start program execution
    image_paths = (["samples/cat_dog.png"])
    target_layer = "layer4"
    arch = "resnet152"
    topk = 1
    output_dir = "./results"
    cuda = "cpu"
    stride = 1
    n_batches = 128
    demo1(image_paths, target_layer, arch, topk, output_dir, cuda)



