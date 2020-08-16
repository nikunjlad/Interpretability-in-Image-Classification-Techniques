
5. Interpretability in Computer Vision
======================================

Interpretability in Computer Vision requires algorithms which can help us understand the nuts and bolts of how deep
neural nets work. More specifically, we would like to visualize what the network seeing and how does it see. We discuss
3 algorithms which we explored as part of our effort to interpret image segmentation algorithms

.. admonition:: Note

   In this work, interpretability in Image Segmentation is done by comparing its predictions with heatmaps generated
   for the same image using classification networks. In short, given an image, we use the below mentioned algorithms
   for a classification task and then compare the observed heatmaps with the actual segmented results on the original
   image. This will give us an idea as to what parts are looked at the most for deciding the class given an image.

.. toctree::
   :maxdepth: 2

   gradcam/gradcam
   gradcampp/gradcampp
   scorecam/scorecam



