from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch, os, time, click, sys, cv2
import torchvision.transforms as T
import numpy as np
from parallel import AcquireVideo, DisplayVideo
from imutils.video import WebcamVideoStream

"""
Semantic Segmentation models

2. fcn_resnet101
4. deeplabv3_resnet101

"""
semantic_models = sorted(
    name
    for name in models.segmentation.__dict__
    if name.islower() and not name.startswith("__") and callable(models.segmentation.__dict__[name])
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


class SemanticSegmentation:

    def __init__(self, model, image=None, show_orig=True, dev="cuda", arch_name="fcn_resnet50",
                 stream=False):
        self.model = model
        self.image = image
        self.show_orig = show_orig
        self.dev = dev
        self.arch_name = arch_name
        self.stream = stream

    def decode_segmap(self, image, orig, nc=21):
        # color map to give to individual classes
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        # create zeros like array
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        a = np.zeros_like(image).astype(np.uint8)

        # for all the 21 classes assign a color to the corresponding pixel value
        for l in range(0, nc):
            idx = image == l
            print(np.any(idx))
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        alpha = 0.75  # alpha value of 0.8 for controlling foreground transperancy
        rgba = np.stack([r, g, b], axis=2)  # stack color channels
        original = cv2.resize(np.array(orig),
                              (rgba.shape[1], rgba.shape[0]))  # original image to feature map size
        # convert the original image from BGR to RGB (OpenCV (^-^))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cv2.addWeighted(rgba, alpha, original, 1 - alpha, 0, rgba)  # alpha blend segmented map on the original image

        return rgba, original

    def segment(self):
        # define the transformations we need to apply on the image
        trf = T.Compose([T.Resize(640),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # apply transformation on the input image and and add an additional dimension before computing
        inp = trf(self.image).unsqueeze(0).to(self.dev)
        out = self.model(inp)  # apply model on the input tensor
        print(out["aux"].shape)
        om = torch.argmax(out['out'].squeeze(), dim=0).detach().cpu().numpy()  # acquire output tensor and convert to numpy
        print(om.shape)

        # create segmentation map
        rgb, original = self.decode_segmap(om, orig=self.image)

        return rgb, original


class PerformanceAnalysis:

    def __init__(self, model1, model2, image_path, device):
        self.model1 = model1
        self.model2 = model2
        self.image_path = image_path
        self.device = device

    def infer_time(self, model):
        img = Image.open(self.image_path)
        trf = T.Compose([T.Resize(256),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        inp = trf(img).unsqueeze(0).to(self.device)

        st = time.time()
        out1 = model.to(self.device)(inp)
        et = time.time()

        return et - st

    def inference_analysis(self, mean_over, arch1, arch2):
        model1_infer_time_list = [self.infer_time(self.model1) for _ in range(mean_over)]
        model1_infer_time_avg = sum(model1_infer_time_list) / mean_over

        model2_infer_time_list = [self.infer_time(self.model2) for _ in range(mean_over)]
        model2_infer_time_avg = sum(model2_infer_time_list) / mean_over

        print('Inference time for first few calls for {}      : {}'.format(arch1, model1_infer_time_list[:10]))
        print('Inference time for first few calls for {} : {}'.format(arch2, model2_infer_time_list[:10]))

        print('The Average Inference time on {} is:     {:.2f}s'.format(arch1, model1_infer_time_avg))
        print('The Average Inference time on {} is: {:.2f}s'.format(arch2, model2_infer_time_avg))

        plt.bar([0.1, 0.2], [model1_infer_time_avg, model2_infer_time_avg], width=0.08)
        plt.ylabel('Time taken in Seconds')
        plt.xticks([0.1, 0.2], [arch1, arch2])
        plt.title('Inference time of' + arch1 + 'and ' + arch2 + 'on CPU')
        plt.show()

    @staticmethod
    def memory_analysis(base, arch1, arch2):
        # memory analysis
        resnet101_size = os.path.getsize('/root/.cache/torch/checkpoints/' + base + '-5d3b4d8f.pth')
        fcn_size = os.path.getsize('/root/.cache/torch/checkpoints/' + arch1 + '_coco-7ecb50ca.pth')
        dlab_size = os.path.getsize('/root/.cache/torch/checkpoints/' + arch2 + '_coco-586e9e4e.pth')

        fcn_total = fcn_size + resnet101_size
        dlab_total = dlab_size + resnet101_size

        print('Size of the' + arch1.split("_")[0] + ' model with ' + arch1.split("_")[
            1] + 'backbone is: {:.2f} MB'.format(fcn_total / (1024 * 1024)))
        print(
            'Size of the' + arch2.split("_")[0] + 'model with ' + arch2.split("_")[0] + 'backbone is: {:.2f} MB'.format(
                dlab_total / (1024 * 1024)))

        plt.bar([0, 1], [fcn_total / (1024 * 1024), dlab_total / (1024 * 1024)])
        plt.ylabel('Size of the model in MegaBytes')
        plt.xticks([0, 1], [arch1, arch2])
        plt.title('Comparison of the model size of ' + arch1 + ' and ' + arch2)
        plt.show()


@click.group()  # group all the click main commands (3 in this program)
@click.pass_context  # pass the context object of the click command. This means which click command to execute
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image_path", type=str, required=True)
@click.option("-a", "--arch", type=click.Choice(semantic_models), required=True)
@click.option("-s", "--show_fig", type=bool, default=True, help="flag to display results or not")
@click.option("-o", "--output", type=str, default="results/", help="name of the directory to save output")
@click.option("--cuda/--cpu", default=True)
def semantic(image_path, arch, show_fig, output, cuda):
    tic = time.time()
    img = Image.open(image_path)   # load image using PIL

    # lambda function to get model name
    mdl_name = lambda x: "FCN" if arch.split("_")[0] == "fcn" else "DeepLab-V3"

    device = get_device(cuda)
    model = models.segmentation.__dict__[arch](pretrained=True).eval()

    s = SemanticSegmentation(model, image=img, show_orig=False, dev=device, arch_name=arch)
    print("Segmenting image using {} network!".format(str(mdl_name(arch))))
    rgba, orig = s.segment()

    # overlay descriptive texts on the original and the segmented image
    cv2.putText(orig, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(rgba, "Semantic Segmented using {}".format(str(mdl_name(arch))), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out = np.vstack((orig, rgba))

    img_name = image_path.split(".")[0].split("/")[-1]
    save_path = os.path.join(output, img_name)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_name = str(mdl_name(arch)) + "_" + img_name + "_semantic_.png"
    cv2.imwrite(os.path.join(save_path, save_name), out)

    # display the semantic segmented map
    if show_fig:
        cv2.imshow("Semantic Segmented Image", rgba)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    toc = time.time()
    print("Inference time for {} network is {} secs!".format(str(mdl_name(arch)), str(toc - tic)))
    sys.exit(1)


@main.command()
@click.option("-i", "--image_path", type=str, required=True)
@click.option("-a", "--arch1", type=click.Choice(semantic_models), required=True)
@click.option("-a", "--arch2", type=click.Choice(semantic_models), required=True)
@click.option("-d", "--device", type=str, default="cuda")
@click.option("-t", "--analysis_type", type=str, default="time")
@click.option("-m", "--mean_over", type=int, default=100)
def analysis(image_path, arch1, arch2, device, mean_over, analysis_type):
    # load models to be compared
    model1 = models.__dict__[arch1](pretrained=True)
    model2 = models.__dict__[arch2](pretrained=True)

    # get the performance object
    p = PerformanceAnalysis(model1, model2, image_path, device)

    if analysis_type == "time":
        p.inference_analysis(mean_over, arch1, arch2)
    elif analysis_type == "memory":
        pass
        # p.memory_analysis()
    else:
        print("No analysis mentioned!")

    sys.exit(1)


@main.command()
@click.option("-a", "--arch", type=click.Choice(semantic_models), required=True)
@click.option("--cuda/--cpu", default=True)
def streaming(arch, cuda):
    device = get_device(cuda)
    model = models.segmentation.__dict__[arch](pretrained=True).eval()
    model.to(device)
    s = SemanticSegmentation(model, show_orig=False, dev=device, arch_name=arch, stream=True)
    vg = AcquireVideo(0).start()
    vs = DisplayVideo(vg.frame).start()

    while True:
        img = Image.fromarray(np.uint8(vg.read()))
        s.image = img
        vs.frame = s.segment()
        # cv2.imshow("real", rgb)

        if vg.stopped or vs.stopped:
            vg.stop()
            vs.stop()
            break

    cv2.destroyAllWindows()
    sys.exit(1)


if __name__ == "__main__":
    main()
