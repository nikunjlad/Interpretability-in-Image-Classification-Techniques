
3.3.1 Verify that the classifier works correctly
================================================

.. toctree::
   :maxdepth: 2

.. role:: raw-html(raw)
   :format: html

There are times when the classifier does not work as expected. There are times when inferences don't happen the way they
should. Most of the times these misclassifications or errors are non fatal. However, there are times when this can be fatal.
Yes, lack of interpretability can be fatal to humans. Let me give a few examples where AI had it's major failures. [1]_

.. raw:: html

  <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
     <iframe src="https://www.youtube.com/embed/LfmAG4dk-rU" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
  </div></br>
  <center>Tesla car slamming itself into a overturned truck in Taiwan</center>
  <center><i>credits: </i><a href="https://www.thedrive.com/news/33789/autopilot-blamed-for-teslas-crash-into-overturned-truck">thedrive</a></center>


During the early June of 2020, a 53-year old driver in Taiwan was driving his self-driving Model 3 Tesla car. The driver
failed to spot an overturned truck on the highway which almost occupied 2 lanes. [2]_ Since the car was on autopilot mode and the
driver was supposedly watching a movie, the car slammed itself into the truck thereby killing the driver. This is not the only
instance when AI was fatal. Back in 2018, an Uber self-driving car killed a jaywalking 45-year old woman after it failed to
detect the person. [3]_ Both these incidents are wake up calls for industry practitioners of AI to consciously check not only how the
self-driving systems or any other complex AI systems work internally but also ask the question why they are predicting the things
they are predicting. It is necessary to have interpretability backed with explainability to really understand the nature of these
accidents and to prevent such fatal situations in future.

.. raw:: html

  <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
     <iframe src="https://www.youtube.com/embed/RASBcc4yOOo" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
  </div></br>
  <center>Uber self-driving car fails to detect jaywalking woman at night</center>
  <center><i>credits: </i><a href="https://www.theguardian.com/technology/2018/mar/22/video-released-of-uber-self-driving-crash-that-killed-woman-in-arizona">theguardian</a></center>

The failures are not just limited to the self driving cars. In mid 2018, IBM Watson supercomputer recommended **unsafe and incorrect**
cancer treatments. Slide decks from June and July 2017, which included customer assessments of Watson Oncology claimed to often
produce consistently inaccurate predictions. [4]_ This could have been fatal had the doctors or physicians not verified the treatment
suggestions given by the algorithm. A.I. is slowly treading it's path into healthcare space and as we become more and more dependent
on AI for it's decision making in healthcare, interpreting its recommendations with reasonable explanations is important. [5]_

Sometimes, even though the classifier is trained with all the samples properly and is generalized to handle the majority of the
scenarios, there are corner cases which are missed out. These corner cases account to ~1% of our usual cases. This statistic becomes
significant especially if we intend to use AI in medical and self driving applications.

.. rubric:: Citations

.. [1] `https://medium.com/syncedreview/2018-in-review-10-ai-failures-c18faadf5983 <https://medium.com/syncedreview/2018-in-review-10-ai-failures-c18faadf5983>`_
.. [2] `https://www.thedrive.com/news/33789/autopilot-blamed-for-teslas-crash-into-overturned-truck <https://www.thedrive.com/news/33789/autopilot-blamed-for-teslas-crash-into-overturned-truck>`_
.. [3] `https://www.theguardian.com/technology/2018/mar/22/video-released-of-uber-self-driving-crash-that-killed-woman-in-arizona <https://www.theguardian.com/technology/2018/mar/22/video-released-of-uber-self-driving-crash-that-killed-woman-in-arizona>`_
.. [4] `https://www.beckershospitalreview.com/artificial-intelligence/ibm-s-watson-recommended-unsafe-and-incorrect-cancer-treatments-stat-report-finds.html <https://www.beckershospitalreview.com/artificial-intelligence/ibm-s-watson-recommended-unsafe-and-incorrect-cancer-treatments-stat-report-finds.html>`_
.. [5] `https://gizmodo.com/ibm-watson-reportedly-recommended-cancer-treatments-tha-1827868882 <https://gizmodo.com/ibm-watson-reportedly-recommended-cancer-treatments-tha-1827868882>`_


