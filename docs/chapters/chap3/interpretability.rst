

3. Understanding Interpretability
=================================

.. toctree::
   :maxdepth: 2

.. role:: raw-html(raw)
   :format: html

3.1 What is Interpretability?
-----------------------------

An important question to ask while pursuing research or in general is - **How?**. How does the car move? How does the earth rotate?
How does an algorithm predict this? How does a self driving car operate? If we get a reasonable explanation and answer to the
above questions related to how, we can say we have an interpretable system at hand. Let me give you a couple of fictional yet
probable scenarios where we might ask the question how.

Say you are travelling in a subway and all of a sudden you find the police narrowing down on the person sitting
next to you. You wonder what just happened and why was that person arrested. Turns out the face recognition software in
the camera mounted in the subway recognized the person next to you as a wanted criminal and notified the police with his
real time location. While everyone is happy and feeling safe that a criminal is arrested, how often do you ask a question -
*"How did the camera figure out the person was criminal?"*.

Let me give you another example. Say, you are fond of plants and you are a tree lover. You have a garden of your own and
since you are new to growing plants and you are not aware of plant species, you happen to download an android application
to identify plant species. This application gives you information of plant characteristics, the weather best suitable for
its growth, the frequency at which you have to give it fertilizers and pesticides or water to keep it healthy. You point your
phone camera to a plant you just bought and the app gives a notification to water it twice everyday and to keep it in
sunlight. Moreover, it also suggests you to cut out the weed growing around the other plants of yours when you capture
the dying leaves on those plants which are slowing turning yellowish. Wouldn't you be surprised by how wizardly this
app is to let you insights of your plants and potentially save you your money? *"How does it figure out the plant species?"*.

.. figure:: ../../_static/plant.jpg
   :align: center
   :width: 600px

   A person using a plant identification app :raw-html:`<br />`
   *credits:* `303magazine <https://images.303magazine.com/uploads/2020/05/unnamed-1.jpg>`_


So coming back to the question at hand -

|  **What is interpretability?**

According to Wikipedia and the formal definition as available on the internet -

| **Interpretability is a relation between formal theories that expresses the possibility of interpreting or translating one into the other.**

Meh! That doesn't give a good intuition. I am pretty sure most of you must be wondering what does this mean in non-scientific terms.
So, let me explain it to you in non-scientific terms.

| **Interpretability is the quality of any system, person or entity to be self explanatory in terms of its actions without knowing about the reasons of those actions.**

It is about being able to discern the mechanics of a situation, system or an algorithms decisions, without necessarily knowing **why**.

An accident occured in your neighbourhood; a person was injured. It seems that when he was riding a bicycle, another car came out from the
other side and hit the bicycle which inturn led to the accident. The accident is interpretable in nature since we know exactly what happened
at every instant of time; in short the event was interpretable. Again, we need not worry about the **why** behind the entire incident as long as we
can clearly discern about the **how** of the situation.


Understanding Explainability and its relation with Interpretability
-------------------------------------------------------------------

3.2 Why Interpretability is necessary?
--------------------------------------

Now that we know what interpretability is, why is it required? Why is the need for interpretability so important lately, especially now
that we have an exponential rise in A.I. related systems? Turns out, the more complex the system, the harder for it to be
self interpretable. For instance, consider a bicycle. It is a fairly interpretable system. You have pedals which require
physical energy by a person. This physical energy inturn helps deliver the energy required to turn the wheels via a chain which
runs through from the pedals to the back tyres of the cycle. A fairly simple system. Quite interpretable. What about a system which
is complex. Like a car for instance. How does the internal combustion engine work? How does it work the way it works? As mentioned
earlier, the more complex a system is, the harder it turns out to be self explanatory in nature aka. less interpretable. And so, while
these complex systems do give out efficient results, helps us save time and delivers loads of profits, at times, they can turn out to
be quite bizzare in terms of their actions.

.. figure:: ../../_static/bicycle.gif
   :align: center
   :width: 600px

   A bicycle mechanism :raw-html:`<br />`
   *credits:* `stringbike <https://www.stringbike.com/_img/magic-stringdrive-01.gif>`_


Some important reasons why interpretability is required in Computer Vision:

3.2.1 Verify that the classifier works correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are times when the classifier does not work as expected. There are times when inferences don't happen the way they
should. Most of the times these misclassifications or errors are non fatal. However, there are times when this can be fatal.
Yes, lack of interpretability can be fatal to humans. Let me give a few examples where AI had it's major failures.

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/LfmAG4dk-rU" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

link: https://www.thedrive.com/news/33789/autopilot-blamed-for-teslas-crash-into-overturned-truck

During the early June of 2020, a 53-year old driver in Taiwan was driving his self-driving Model 3 Tesla car. The driver
failed to spot an overturned truck on the highway which almost occupied 2 lanes. Since the car was on autopilot mode and the
driver was supposedly watching a movie, the car slammed itself into the truck thereby killing the driver. This is not the only
instance when AI was fatal. Back in 2018, an Uber self-driving car killed a jaywalking 45-year old woman after it failed to
detect the person. Both these incidents are wake up calls for industry practitioners of AI to consciously check not only how the
self-driving systems or any other complex AI systems work internally but also ask the question why they are predicting the things
they are predicting. It is necessary to have interpretability backed with explainability to really understand the nature of these
accidents and to prevent such fatal situations in future.

link: https://www.nbcnews.com/video/self-driving-uber-crash-that-killed-pedestrian-in-tempe-arizona-caught-on-camera-1191726659979


2. Improve the classifier
3. Learn from the algorithm itself about it's decisions
4. Get insights

