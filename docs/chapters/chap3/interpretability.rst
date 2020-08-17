

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
the camera mounted in the subway recognized the person next to you and notified the police with his
real time location. While everyone usually ignore the situation and carry on with their lives, considering it to be
just another arrest of some random individual, how often do they ask a question -
*"How was the police notified about the person next to you?"*.

Let me give you another example. Say, you are fond of plants and you are a tree lover. You have a garden of your own and
since you are new to growing plants and you are not aware of plant species, you happen to download an android application
to identify plant species. This application gives you information of plant characteristics, the weather best suitable for
its growth, the frequency at which you have to give it fertilizers and pesticides or water to keep it healthy. You point your
phone camera to a plant you just bought and the app gives a notification to water it twice everyday and to keep it in
sunlight. Moreover, it also suggests you to cut out the weed growing around the other plants of yours when you capture
the dying leaves on those plants which are slowly turning yellowish. Wouldn't you be surprised by how wizardly this
app is, to let you insights of your plants and potentially save you your money? *"How does it figure out the plant species?"*.

.. figure:: ../../_static/plant.jpg
   :align: center
   :width: 600px

   A person using a plant identification app :raw-html:`<br />`
   *credits:* `303magazine <https://images.303magazine.com/uploads/2020/05/unnamed-1.jpg>`_


So coming back to the question at hand -

|  **What is interpretability?**

According to Wikipedia and the formal definition as available on the internet [1]_ -

| **Interpretability is a relation between formal theories that expresses the possibility of interpreting or translating one into the other.**

Meh! That doesn't give a good intuition. I am pretty sure most of you must be wondering what does this mean in non-scientific terms.
So, let me explain it to you in non-scientific terms.

| **Interpretability is the quality of any system, person or entity to be self explanatory in terms of its actions without knowing about the reasons of those actions.**

It is about being able to discern the mechanics of a situation, system or an algorithms decisions, without necessarily knowing **why**.
More specifically it is an extent to which a cause and effect can be observed within a system. [2]_ It is an extent to which you can *predict*
what is going to happen given a change in input or algorithmic parameters. It defines the amount of consistency in predicting a model's
result without trying to know the reasons behind the prediction. [3]_

An accident occured in your neighbourhood; a person was injured. It seems that when he was riding a bicycle, another car came out from the
other side and hit the bicycle which inturn led to the accident. The accident is interpretable in nature since we know exactly what happened
at every instant of time; in short the event was interpretable. Again, we need not worry about the **why** behind the entire incident as long as we
can clearly discern about the **how** of the situation. But, can we explain *why* the accident happened?


3.2 Understanding Explainability and its relation with Interpretability
-----------------------------------------------------------------------

While we are on the topic of unraveling the truth of what goes inside deep neural networks, it is important to note that
**interpretability** is not **explainability**. Both the terms are used interchangeably often and it's necessary to understand
the difference between the two. We explained what interpretability is earlier. What about explainability?

If we revisit the earlier example of the accident, we came to know how the accident occurred, that the car hit the bicycle from
the blind spot of the cyclist. But **why** did the accident happen? Did the cyclist deliberately break the traffic signal and tried
crossing the road amid speeding cars? Was there water on the road which caused the car to slid and hit the bicycle? Did the breaks of
the bicycle fail or more worse did the breaks of the car fail? Was the visibility poor or signal broken?, etc etc..! There are endless
possible reasons why an accident can occur, while only **one** definite method / explaination exists as to **how** it occurred.

Let me take a fresh example. Say, you are interested in star gazing. You figure out the best possible location with least ambient
light to observe the constellation of stars. You have a friend who is an astronomer and who owns a very powerful yet complex telescope.
While your friend knows how to set the telescope so as to clearly observe the stars, the milky way or the meteor shower along with the
reasons as to why it works the way it works, you are fairly new to operating the telescope. You intend to learn to operate the telescope,
and so your friend explains to you how to operate. Turns out, you need to set the magnification factor, the azimuth mount settings and
whole lot of other configurations until you finally can see the stars or the planets. You really don't know **why** you can see the stars
so clearly, except for the fact that you know **how** to operate the telescope. Essentially, you are able to **interpret** the working of
the telescope, but you do not know **why** telescope works and **why** do we need exactly those set of configurations as explained to you
by your friend.

A telescope has a very specific scientific principle governing its operation. The laws of physics are constant and the
**explanability** factor of your telescope with depend upon the situation and external factors. If the location is not a plateau, your
telescope mount needs to have its azimuth settings aligned accordingly and for some inclined region, it has to be aligned in a different way.
If you observe, irrespective of the location and the external weather conditions, we always have to configure the azimuth settings, the
magnification factor, etc. The steps are defined. The **how** is constant so as to **reproduce** the same result of observing the stars.
The process is **consistent**. The system is interpretable! On the other side, various factors affected the magnitude and the nature by
which you configure your telescope settings. Knowing **why** to configure those settings requires domain knowledge, and as mentioned in
the previous example, the reasons are possibly endless. Explainability, yet again has many variables.


3.2.1 Important Properties Of Explainability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Portability**: It defines the range of machine learning models where the explanation method can be used.
2. **Expressive Power**: It defines as the structure of an explanation that a method is able to generate.
3. **Translucency**: This describes as to how much the method of explanation depends on the machine learning model. Low translucency methods tend to have higher portability.
4. **Algorithmic Complexity**: It defines the computational complexity of a method where the explanations are generated.
5. **Fidelity**: High fidelity is considered as one of the important properties of an explanation as low fidelity lacks in explaining the machine learning model.

.. warning::

   Although Interpretablility and Explainability have different purposes, for the sake of simplicity, we use Interpretability
   to denote both the concept of Interpretability and Explainability. From here on, the use of the word Interpretability will
   give us answer to both the **how** and the **why** questions, instead of just the **how** question. Do note, that both these
   terms are not meant to be confused and should ideally be treated with their unique representations.

3.3 Why Interpretability is necessary?
--------------------------------------

Now that we know what interpretability is, why is it required, why is the need for interpretability so important lately, especially now
that we have an exponential rise in A.I. related systems? Turns out, the more complex the system, the harder for it to be
self interpretable. For instance, consider a bicycle. It is a fairly interpretable system. You have pedals which require
physical energy by a person. This physical energy inturn helps deliver the energy required to turn the wheels via a chain which
runs through from the pedals to the back tyres of the cycle. A fairly simple system. Quite **interpretable**!

.. figure:: ../../_static/bicycle.gif
   :align: center
   :width: 600px

   A bicycle mechanism :raw-html:`<br />`
   *credits:* `stringbike <https://www.stringbike.com/_img/magic-stringdrive-01.gif>`_

What about a system which is complex. Like a car for instance. How does the internal combustion engine work? Why does it work the way it works? As mentioned
earlier, the more complex a system is, the harder it turns out to be self explanatory in nature aka. **less interpretable**. And so, while
these complex systems do give out efficient results, saves us time and delivers loads of profits, at times, they can turn out to
be quite bizzare in terms of their actions.

Some important reasons why interpretability is required in Computer Vision:

.. toctree::
   :maxdepth: 2

   verify_classifier
   improve_classifier
   learn_from_algorithm
   get_insights

.. rubric:: Citations

.. [1] `https://en.wikipedia.org/wiki/Interpretability <https://en.wikipedia.org/wiki/Interpretability>`_
.. [2] `https://www.kdnuggets.com/2018/12/machine-learning-explainability-interpretability-ai.html <https://www.kdnuggets.com/2018/12/machine-learning-explainability-interpretability-ai.html#:~:text=Interpretability%20is%20about%20the%20extent,be%20observed%20within%20a%20system.&text=Explainability%2C%20meanwhile%2C%20is%20the%20extent,be%20explained%20in%20human%20terms.>`_
.. [3] `https://analyticsindiamag.com/explainability-vs-interpretability-in-artificial-intelligence-and-machine-learning/ <https://analyticsindiamag.com/explainability-vs-interpretability-in-artificial-intelligence-and-machine-learning/>`_




