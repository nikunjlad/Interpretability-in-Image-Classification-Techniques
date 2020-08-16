

6. Observations
===============

.. toctree::
   :maxdepth: 2

.. role:: raw-html(raw)
   :format: html

As we mentioned earlier, it is not possible for us to have a Grad-CAM visualization for a segmentation network, since
it does not output probability scores like a classification network. What we will do now is a comparative analysis based
on our subjective observations having observed Grad-CAM and Grad-CAM++ visualization on classification networks. We then
observe the semantic and instance segmented maps on these images and see how good the inferences were.

.. warning::

   The Grad-CAM visualizations are from Classification models, and they have no direct co-relation with segmentation maps.
   We simply observe the interpretable nature of classification models and extend the conclusions on segmentation models.
   If you recollect, **Semantic Segmentation = Classification + Localization** and **Instance Segmentation = Semantic Segmentation + Object Detection**.
   Therefore, we argue that any interpretable results we acquire for classification gets carried forward to Segmentation too due to this
   inverse co-relation. This work is not a direct observations of Interpretable Segmentation models.


Before we jump into our observations, its important to note than we used pretrained models provided by PyTorch for our work.
More specifically, we used:

1. fcn_resnet101 (FCN network with ResNet-101 backbone) and deeplabv3_resnet101 (DeepLab-v3 Atrous Convolution based network again with ResNet-101 backbone)
for Semantic Segmentation tasks. These models were trained on subset of COCO 2017 PASCAL VOC dataset with just 20 categories. :raw-html:`<br />`
2. maskrcnn_resnet50_fpn (Feature Pyramid Pooling based Mask-RCNN network with ResNet-50 backbone) for Instance Segmentation tasks. This model was trained
on 80 categories of the COCO 2017 PASCAL VOC dataset. :raw-html:`<br />`
3. vgg-16, vgg-19, resnet-50, resnet-101 and resnet-152 based classification pretrained models for our Grad-CAM and Grad-CAM++ visualizations.
All the classification models were trained on IMAGENET-1000 dataset with 1000 classes. :raw-html:`<br />`

.. figure:: ../../_static/observations.png
   :align: center
   :width: 800px

   Grad-CAM and Grad-CAM++ visualizations of VGG-16 network for layer 43 along with Segmentation outputs :raw-html:`<br />`

Now, that we know how our models were trained on, we observed that our experiment set shrunk to just 20 categories. This is
because, semantic segmentation models were trained on only 20 categories and any image with a category not trained on the ones provided
was automatically segmented as background. For sanity purposes, we took images from only those categories for our tests.

From the image above, we show semantic and instance segmented masks for 5 image categories - cars, bottes, boats, seagulls and sofa.
We can observe that semantic fcn maps are coarse and do not fully segment the desired object in the image in contrast to deeplabv3 semantic maps.
However, instance segmented maps are very smooth in terms of their segmentation results. Not only are the individual instances of similar category
detected but masked uniquely. If we observe the Grad-CAM visualizations we see the desired object highlighted with the heatmap. However, Grad-CAM
maps are more focussed on highlighting single instances and fail in case of multiple instances. Grad-CAM++ fills this caveat and highlights or
attends to multiple instances of same object in the image. For instance, observe the car image in the first row. Grad-CAM++ clearly highlights
both cars while Grad-CAM fails to do so.

In case of the boat image, semantic FCN highlights the mast of the ship with a small red patch. Likewise, mask-rcnn based instance maps too
mask both the boats uniquely and entirely with great detail. This can be verified by the Grad-CAM++ heatmaps, where we can see the a significant
amount of focus being given to entire boat along with its mast not to forget the smaller boat as well.

