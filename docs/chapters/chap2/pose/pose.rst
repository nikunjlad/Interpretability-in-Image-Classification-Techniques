

2.4 Pose Estimation
===================

.. toctree::
   :maxdepth: 2

.. role:: raw-html(raw)
   :format: html

2.4.1 What is Pose Estimation?
------------------------------

Pose estimation is a task that infers the pose of a person or object in an image or video. We can also think of pose
estimation as the process of determining the position and orientation of an object or a person at any given instant of time.
Pose estimation works by identifying, locating and tracking a number of keypoints in a person or an object.
It is also defined as a search for a specific pose in space of all articulated poses. With respect to
humans, the problem becomes **Human Pose Estimation** while with inanimate objects it's called **Rigid Pose Estimation**.

https://www.fritz.ai/pose-estimation/
https://nanonets.com/blog/human-pose-estimation-2d-guide/

.. figure:: ../../../_static/keypoint.png
   :align: center
   :width: 700px

   Pose estimated 17 Keypoints in human body  :raw-html:`<br />`
   *credits:* `nanonets <https://nanonets.com/blog/content/images/2019/04/Screen-Shot-2019-04-11-at-5.17.56-PM.png>`_

2.4.2 Applications of Pose Estimation Algorithms
------------------------------------------------

.. figure:: ../../../_static/homecourt.gif
   :align: center
   :width: 700px

   Homecourt app training for football  :raw-html:`<br />`
   *credits:* `medium <https://miro.medium.com/max/900/1*GBUW2y_aya34eC0eB1M5Yw.gif>`_

Pose Estimation algorithms have numerous applications and only recently has gained lot of popularity. For instance, an
application named `HomeCourt <https://www.homecourt.ai/>`_ uses pose estimation for training amateurs and professionals to improve and learn
various sports like basketball or football (as shown above) and also workout activities like cross-fit or high intensity training. Like wise, pose
estimation is also used in gaming industry for developing real time pose based games. They are also used in activity
recognition systems to figure what activity an individual is performing based on its pose in space.

.. figure:: ../../../_static/activity.gif
   :align: center
   :width: 700px

   Human Activity Recognition  :raw-html:`<br />`
   *credits:* `pythonawesome <https://pythonawesome.com/content/images/2019/10/recog_actions2.gif>`_

2.4.3 Challenges faced by Pose Estimation Algorithms
----------------------------------------------------

While we are able to achieve good estimation of pose given current state-of-the art algorithms and resources, there are
still some challenges faced by pose estimation algorithms. It is difficult to track and estimate small and barely visible
joints. The algorithm is yet not robust to occlusions and poor lighting conditions.