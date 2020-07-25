import os.path as osp
import click
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from utils import get_classtable

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
@click.group()  # group all the click main commands (3 in this program)
@click.pass_context  # pass the context object of the click command. This means which click command to execute
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


#
@main.command()  # this is a click command. Infact the first click command. It has below optional arguments
@click.option("-i", "--image-paths", type=str, multiple=True,
              required=True)  # ask for image paths, multiple taken in as tuple
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)  # model to be used
@click.option("-t", "--target-layer", type=str, required=True)  # layer to be visualized
@click.option("-k", "--topk", type=int, default=3)  # top k most relevant searches to be returned
@click.option("-o", "--output-dir", type=str, default="./results")  # provide output directory
@click.option("--cuda/--cpu", default=True)  # run on cpu or gpu?
def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """
    device = get_device(cuda)  # set device to cpu to gpu based on availability

    # Synset words
    classes = utils.get_classtable()
    # print("Num of classes: ", len(classes))

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)  # load pretrained resnet152
    model.to(device)
    model.eval()  # evaluate the pretrained model

    # get a list of transformed and original raw images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)  # stack the transformed and processed images and send to device

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

    gcam = GradCAM(model=model, target_layer=target_layer, gradcampp=True)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate()

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
                    "{}-{}-gradcampp-{}-{}.png".format(
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
                    "{}-{}-guided_gradcampp-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


@main.command()  # this is a click command. Infact the second click command. It has below optional arguments
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
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
        gcam.target_layer = target_layer
        regions = gcam.generate()

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


@main.command()  # this is a click command. Infact the third click command. It has below optional arguments
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
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
    main()  # start program execution
    # image_paths = (["samples/cat_dog.png"])
    # target_layer = "layer4"
    # arch = "resnet152"
    # topk = 1
    # output_dir = "./results"
    # cuda = "cpu"
    # stride = 1
    # n_batches = 128
    # demo1(image_paths, target_layer, arch, topk, output_dir, cuda)
