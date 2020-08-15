"""
Created on Wed Apr 29 16:11:20 2020

@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch, click, sys, os
import torch.nn.functional as F
from torchvision import models
from utility import save_class_activation_images, recreate_image, preprocess_image, get_classtable

# define torchvision models to choose from for during command line execution
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():

            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer

        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class ScoreCam:
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer, classes):
        self.model = model
        self.model.eval()
        self.classes = classes
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
            print("Predicted Class: {}".format(self.classes[target_class]))

        # Get convolution outputs
        target = conv_output[0]

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):

            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :], 0), 0)

            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(227, 227), mode='bilinear', align_corners=False)

            if saliency_map.max() == saliency_map.min():
                continue

            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image * norm_saliency_map)[1], dim=1)[0][target_class]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255
        return cam


@click.command()
@click.option("-i", "--image_path", type=str, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-l", "--target_layer", type=int, required=True)
@click.option("-o", "--output", type=str, default="./results")
def main(image_path, target_layer, arch, output):

    # 1. Read image
    original_image = Image.open(image_path).convert('RGB')

    # 2. Process image and get classes
    prep_img = preprocess_image(original_image)
    classes = get_classtable()

    # 3. Define model
    model = models.__dict__[arch](pretrained=True)

    # 4. Apply model to image with a given target layer
    score_cam = ScoreCam(model, target_layer=target_layer, classes=classes)

    # 5. create Score-CAM maps
    cam = score_cam.generate_cam(prep_img, target_class=None)

    # 6. convert image from Numpy to PIL before masking image with visualization
    prep_img = recreate_image(prep_img)

    # 7. create a filename where we need to export visualization
    img_name = image_path.split(".")[0].split("/")[-1]
    save_path = os.path.join(output, img_name)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_name_to_export = os.path.join(save_path, img_name)

    # 8. save the Score-CAM activation maps and the masked image with the heatmap
    save_class_activation_images(prep_img, cam, arch, target_layer, file_name_to_export)
    print('Score cam completed')
    sys.exit(1)


if __name__ == '__main__':
    main()
    # Get params
    # target_example = 0  # Snake
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = \
    #     get_example_params(target_example)
    # # Score cam
    # score_cam = ScoreCam(pretrained_model, target_layer=11)
    # # Generate cam mask
    # cam = score_cam.generate_cam(prep_img, target_class)
    # # print(original_image.size)
    # # print(original_image.mode)
    # # print(np.array(original_image).shape)
    # prep_img = recreate_image(prep_img)
    # print(type(prep_img))
    # # Save mask
    # save_class_activation_images(prep_img, cam, file_name_to_export)
    # print('Score cam completed')
