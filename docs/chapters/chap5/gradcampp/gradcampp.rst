

5.2 Grad-CAM++
==============

.. toctree::
   :maxdepth: 2

.. role:: raw-html(raw)
   :format: html

Grad-CAM++ is the extension of Grad-CAM which we observed earlier. Grad-CAM is not good at localizing multiple objects in images
belonging to the same class. For multiple object images, Grad-CAM do not capture the object in it’s entirety. This is required for better
recognition tasks and hence, Grad-CAM++ fills for these caveats.

.. figure:: ../../../_static/gradcampp.png
   :align: center
   :width: 600px

   Grad-CAM++ Architecture :raw-html:`<br />`

Grad-CAM++ provides pixel-wise weighting of the gradients of the output w.r.t. to the particular spatial position in the
final feature map towards overall decision of the CNN. This provides a measure of importance of each pixel in the feature map
towards overall decision of the CNN.

.. figure:: ../../../_static/gradcampp_intuit.png
   :align: center
   :width: 600px

   Grad-CAM++ Intuition :raw-html:`<br />`

.. admonition:: Note

   Below contains images of math and the underlying logic surrounding Grad-CAM++. Images are uploaded since writing such
   complex math and getting it rendered was difficult. Please bear my hand writing.

.. figure:: ../../../_static/page1.jpg
   :align: center
   :width: 700px

   Math Page 1 :raw-html:`<br />`


.. figure:: ../../../_static/page2.jpg
   :align: center
   :width: 700px

   Math Page 2 :raw-html:`<br />`

.. figure:: ../../../_static/page3.jpg
   :align: center
   :width: 700px

   Math Page 3 :raw-html:`<br />`


.. figure:: ../../../_static/page4.jpg
   :align: center
   :width: 700px

   Math Page 4 :raw-html:`<br />`


.. figure:: ../../../_static/gradcampp_vis.png
   :align: center
   :width: 700px

   Original Image vs Grad-CAM++ visualization for layer 43 of VGG-16 network :raw-html:`<br />`