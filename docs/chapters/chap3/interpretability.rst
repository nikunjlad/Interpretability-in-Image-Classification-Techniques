

Understanding Interpretability
==============================

.. toctree::
   :maxdepth: 2

.. role:: raw-html(raw)
   :format: html

What is Interpretability?
-------------------------

An important question to ask while pursuing research or in general is - **How?**.
How does the car move? Likewise, What is a CNN and how does algorithm predict this? If we get a reasonable explanation and answer to each
of the above question related to how, we can say we have an interpretable system at hand. Well, what does that
exactly mean to have an interpretable system? Let me give you a couple of fictional yet probable scenarios.

Say you are travelling in a subway and all of a sudden you find the police narrowing down on the person sitting
next to you. You wonder what just happened and why was that person arrested. Turns out the face recognition software in
the camera mounted in the subway recognized the person next to you as a wanted criminal and notified the police with his
real time location. While everyone is happy and feeling safe that a criminal is arrested, how often do you ask a question -
"How did the camera figure out the person was criminal?".

https://i.insider.com/5db5c65d045a314bf82efda2?width=1100&format=jpeg&auto=webp

Let me give you another example. Say, you are fond of plants and you are a tree lover. You have a garden of your own and
since you are new to growing plants and you are not aware of plant species, you happen to download an android application
to identify plant species. This application gives you information of plant characteristics, the weather best suitable for
its growth, the frequency at which you have to give it fertilizers and pesticides or water to keep it healthy. You point your
phone camera to a plant you just bought and the app gives a notification to water it twice everyday and to keep it in
sunlight. Moreover, it also suggests you to cut out the weed growing around the other plants of yours when you capture
the dying leaves on those plants which are slowing turning yellowish. Wouldn't you be surprised by how wizardly this
app is to let you insights of your plants and potentially save you your money? "How does it figure out the plant species?".

https://images.303magazine.com/uploads/2020/05/unnamed-1.jpg

So coming back to the question at hand:

|  **What is interpretability?**

According to Wikipedia and the formal definition as available on the internet,

| *Interpretability is a relation between formal theories that expresses the possibility of interpreting or translating one into the other.*

Meh! That doesn't give a good intuition. I am pretty sure most of you must be wondering what does this mean in non-scientific terms.
So, let me explain it to you in non-scientific terms.

| *Interpretability is the quality of any system, person or entity to be self explanatory in terms of its actions without knowing about the reasons of those actions.*

Interpretability is about being able to discern the mechanics of a situation, system or an algorithms decisions, without necessarily knowing why.

An accident occured in your neighbourhood. A person was injured. It seems that when he was riding a bicycle, another car came out from the
other side and hit the bicycle which inturn led to the accident. The accident is interpretable in nature since we know exact what happened
at every instant of time, in short the event was interpretable. Again, we need not worry about **why** behind the entire incident as long as we
can clearly discern about the **how** of the situation.

Why Interpretability is necessary?
----------------------------------

Now that we know what interpretability is, why is it required? Why is the need for interpretability so important lately, now
that we have an exponential rise in A.I. related systems? Turns out, the more the complex the system, the harder for it to be
self interpretable. For instance, consider a bicycle. It is a fairly interpretable system. You have pedals which require a
physical energy by a person. This physical energy inturn helps deliver the energy required to turn the wheels via a chain which
runs through from the pedals to the back tyres of the cycle. A fairly simple system. Quite interpretable. What about a system which
is complex. Like a car for instance. How does the internal combustion engine work? How does it work the way it works? As mentioned
earlier, the more complex a system is, the harder it turns out to be self explanatory in nature aka. less interpretable. And so, while
these complex systems do give out efficient results, helps us save time and delivers loads of profits, at times, they can turn out to
be quite bizzare in terms of their actions.

https://www.stringbike.com/_img/magic-stringdrive-01.gif

Some important reasons why interpretability is required in Computer Vision:

1. Verify that the classifier works correctly.
2. Improve the classifier
3. Learn from the algorithm itself about it's decisions
4. Get insights



Understanding Explainability and its relation with Interpretability
-------------------------------------------------------------------
