from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision import models
import torchvision, torch, cv2, random, click, os, sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")

"""
Instance Segmentation models

1. maskrcnn_resnet50_fpn

"""
instance_models = sorted(
    name
    for name in models.detection.__dict__
    if name.islower() and not name.startswith("__") and callable(models.detection.__dict__[name])
)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()  # if cuda is True and is available
    device = torch.device("cuda" if cuda else "cpu")  # if cuda == True, then select the device, else use cpu
    if cuda:
        current_device = torch.cuda.current_device()  # if cuda is there then current device id
        print("Device:", torch.cuda.get_device_name(current_device))  # get name of current device
    else:
        print("Device: CPU")
    return device


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class InstanceSegmentation:

    def __init__(self, img, model, threshold, arch="maskrcnn_resnet50_fpn", device="cuda"):
        self.img = img
        self.model = model
        self.threshold = threshold
        self.device = device
        self.arch = arch

    def generate_masks(self, image):
        colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
                   [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190], [128, 0, 0], [0, 128, 0],
                   [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                   [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                   [0, 64, 0],
                   [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
        coloured_mask = np.stack([r, g, b], axis=2)

        return coloured_mask

    def predict(self):
        # 1. open image using PIL
        img = Image.fromarray(np.uint8(self.img))

        # 2. define transformation
        transform = T.Compose([T.ToTensor()])

        # 3. apply transformation
        img = transform(img)

        # 4. apply model on the image. Image should be passed as a list
        pred = self.model([img])

        # 5. prediction scores are a list of dictionaries. we get the score, using the score tag
        pred_score = list(pred[0]['scores'].detach().numpy())

        # 6. loop over all the prediction scores and append all those in the list and
        # check all those which are above the threshold.
        pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold][-1]

        # 7. get all the masks value which are greater than 50% threshold, squeeze it by dropping dimension and numpyify
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

        # 8. create a list of coco labels predicted by the model
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]

        # 9. get the bounding box co-ordinates of all the objects detected in the image.
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

        # 10. get all the masks, prediction boxes and classes corresponding to those predictions which meet the
        # threshold
        masks = masks[:pred_t + 1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        return masks, pred_boxes, pred_class

    def segment(self, rect_th=1, text_size=0.30, text_th=1):  # 1    0.6   1
        # 1. get the masks, boxes and predicted classes
        masks, boxes, pred_cls = self.predict()

        # 3. loop over the masks
        for i in range(len(masks)):
            rgb_mask = self.generate_masks(masks[i])  # get the mask color
            self.img = cv2.addWeighted(self.img, 1, rgb_mask, 0.5, 0)  # overlay transperant mask on the object
            cv2.rectangle(self.img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(self.img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

        return self.img


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image_path", type=str, required=True, help="path to input image")
@click.option("-a", "--arch", type=click.Choice(instance_models), required=True,
              help="name of pretrained model to load")
@click.option("-o", "--output", type=str, default="results/", help="name of the directory to save output")
@click.option("-s", "--show_fig", type=bool, default=True, help="flag to display results or not")
@click.option("-t", "--threshold", type=float, default=0.5, help="threshold value to filter out predictions")
@click.option("--cuda/--cpu", default=True, help="flag to select device inference to be GPU or CPU")
def instance(image_path, arch, threshold, output, show_fig, cuda):
    # 1. configure device to be used
    device = get_device(cuda)
    image = cv2.imread(image_path)

    # aspect = image.shape[1] / image.shape[0]
    # if aspect > 0:
    #     image = cv2.resize(image, (227, int(800 / aspect)), interpolation=cv2.INTER_LANCZOS4)
    # else:
    #     image = cv2.resize(image, (600, int(600 / aspect)), interpolation=cv2.INTER_LANCZOS4)

    image = cv2.resize(image, (227,227), interpolation=cv2.INTER_LANCZOS4)

    # 2. load the model
    try:
        print("Loading model...")
        model = models.detection.__dict__[arch](pretrained=True).eval()

        i = InstanceSegmentation(image, model, threshold, arch, device)
        img = i.segment()
        # cv2.putText(img, "Instance Segmented using {}".format(arch), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.70, (0, 0, 255), 1)   #  (10, 30)      0.75     1

        img_name = image_path.split(".")[0].split("/")[-1]
        save_path = os.path.join(output, img_name)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_name = arch + "_" + img_name + "_.png"
        cv2.imwrite(os.path.join(save_path, save_name), img)

        if show_fig:
            print("Displaying image")
            cv2.imshow("Instance Segmentation", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(e)

    sys.exit(1)


if __name__ == "__main__":
    main()
