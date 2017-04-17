# Week 5 Lecture - Object Recognition

## 5a - Why Object Recognition Is Difficult

* Segmentation
* Lighting
* Deformation
* Affordances
* Viewpoint

## 5b - Ways To Achieve Viewpoint Invariance

It's hard to appreciate how difficult it is.  It's the main difficulty in making computers perceive. We are still lacking generally accepted solutions

### Today's Approaches For Viewpoint Invariance

* Use redundant invariant features
* Put a box around the object and use normalized pixels
* Lecture 5c: use replicated features with pooling: "convolutional neural nets"
* Use a hierarchy of parts that have explicit poses relative to the camera \(will be described later in course\)

### The Invariant Feature Approach

* Extract a large, redundant set of features that are invariant under transformations
  * e.g. a pair of roughly parallel lines with a red dot between them
  * this is what baby herring gulls use to know where to peck for food
* With enough invariant features, there is only one way to assemble them into an object
  * We don't need to represent the relationships between the features directly because they are captured by other features.
* For recognition, _we must avoid forming features from parts of different objects_

### The Judicious Normalization Approach

* Put a box around the object and use it as a coordinate frame for a set of normalized pixels.
  * This solves the dimension hopping problem. If we choose the box correctly, the same part of an object always occurs on the same normalized pixels.
  * The box can provide invariance to many degrees of freedom: **translation, rotation, scale, shear, stretch ...**
* Choosing the box is difficult because of: **segmentation errors, occlusion, unusual orientations**
* We need to recognize the shape to get the box right!

We recognize the letter before doing mental rotation to decide if it's a mirror image:

![](/assets/hinton-lec5b-rotatedR.png)

### The Brute Force Normalization Approach

* When training the recognizer, use well-segmented, upright images to fit the correct box.
* At test time try all possible boxes in a range of positions and scales.
  * Widely used for detecting upright things like faces and house numbers in _unsegmented images_.
  * It is much more efficient if the recognizer can cope with some variation in position and scale so that we can use a coarse grid when trying all possible boxes.

## 5c - Convolutional Neural Networks For Hand-Written Digit Recognition

### The replicated feature approach

\(currently the dominant approach for neural networks\)

### Backpropagation with weight constraints

tbd

### What does replicating the feature detectors achieve?

tbd

### Pooling the outputs of replicated feature detectors

tbd

### Le Net

Yann LeCun

### ![](/assets/hinton-lec5c-lecun-minst-architecture.png)The 82 errors made by LeNet5

### ![](/assets/hinton-lec5c-leNet-errors.png)Priors and Prejudice

tbd

### The brute force approach

tbd

### The errors made by the Ciresan et. al. net

### tbd ![](/assets/hinton-lec5c-Ciresan-errors.png)How to detect a significant drop in the error rate

tbd

## 5d - Convolutional Neural Networks For Object Recognition

TBD

# Week 5 Quiz

TBD

# Week 5 FAQ

* TBD



