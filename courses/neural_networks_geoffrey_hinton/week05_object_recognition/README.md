# Week 5 Lecture - Object Recognition

## 5a01 - Why Object Recognition Is Difficult

* Segmentation
* Lighting
* Deformation
* Affordances
* Viewpoint

## 5b01 - Ways To Achieve Viewpoint Invariance

It's hard to appreciate how difficult it is.  It's the main difficulty in making computers perceive. We are still lacking generally accepted solutions

### 5b02 - Some ways to achieve viewpoint invariance

* Use redundant invariant features
* Put a box around the object and use normalized pixels
* Lecture 5c: use replicated features with pooling: "convolutional neural nets"
* Use a hierarchy of parts that have explicit poses relative to the camera \(will be described later in course\)

### 5b03 - The Invariant Feature Approach

* Extract a large, redundant set of features that are invariant under transformations
  * e.g. a pair of roughly parallel lines with a red dot between them
  * this is what baby herring gulls use to know where to peck for food
* With enough invariant features, there is only one way to assemble them into an object
  * We don't need to represent the relationships between the features directly because they are captured by other features.
* For recognition, _we must avoid forming features from parts of different objects_

### 5b04 - The Judicious Normalization Approach

* Put a box around the object and use it as a coordinate frame for a set of normalized pixels.
  * This solves the dimension hopping problem. If we choose the box correctly, the same part of an object always occurs on the same normalized pixels.
  * The box can provide invariance to many degrees of freedom: **translation, rotation, scale, shear, stretch ...**
* Choosing the box is difficult because of: **segmentation errors, occlusion, unusual orientations**
* We need to recognize the shape to get the box right!

We recognize the letter before doing mental rotation to decide if it's a mirror image:

![](/assets/hinton-lec5b-rotatedR.png)

### 5b05 - The Brute Force Normalization Approach

* When training the recognizer, use well-segmented, upright images to fit the correct box.
* At test time try all possible boxes in a range of positions and scales.
  * Widely used for detecting upright things like faces and house numbers in _unsegmented images_.
  * It is much more efficient if the recognizer can cope with some variation in position and scale so that we can use a coarse grid when trying all possible boxes.

## 5c01 - Convolutional Neural Networks For Hand-Written Digit Recognition

CNNs originated in 1980s. It was possible on computers then, and they worked really well.

### 5c02 - The replicated feature approach

\(currently the dominant approach for neural networks\)

Use many different copies of the same feature detector with different positions.

* We could replicate across scale and orientation, but that's tricky and expensive.
* Replication greatly reduces the number of free parameters to be learned.

Al

### 5c03 - Backpropagation with weight constraints

* it's easy to modify the backpropagation algorithm

### 5c04 - What does replicating the feature detectors achieve?

* many people claim replicating translation invariance
  * that's not true
* they achieve equivariance, not invariance
* **equivariant activities**: replicated features do not make the neural activities invariant to translation. the activities are equivariant.
* \[images here\]
* **Invariant knowledge**: 

"We are achieving equivariance in the activities and invariance in the weights."

### 5c04 - Pooling the outputs of replicated feature detectors

Get a small amount of translational invariance at each level of the net by averaging _four neighboring replicated detectors_ to give a single output to the next level.

* reduces the number of inputs to the next layer of feature extraction, which allows us to have many more _different_ feature maps
* taking the max of four works slightly better \(_than averaging? works better for what?_\)

**Problem**: after several levels of pooling, we lose info about positions of things

* consequence: makes it impossible to use precise spatial relationships

### 5c - Question

5:53

Consider the following convolutional neural network. The input is a 3 x 3 image and each hidden unit has a 2 x 2 _weight filter_ that connects to a localized region of this image. This is shown in the following diagram \(note that this represents **one** _filter map_ and some lines are dashed just to avoid clutter\):

![](/assets/hinton-lec5c-question-pool.png)

In the image shown to the network, black pixels have a value of 1 while white pixels have a value of 0, so only two pixels in this image have a value of 1. Suppose we wanted to pool the output of each hidden unit using _max pooling_. If $$y_1=2,y_2=0,y_3=1,y_4=0$$ and max-pooling is defined as $$y_{pool}=\max_iy_i$$, then what will be the value of $$y_{pool}$$ in this example?

1. 2
2. 0
3. 1
4. 3

#### 5c - Question Work

I was not expecting that pooling would reuse pixels. If $$y_{pool}$$ is the maximum of any output $$y_i$$, then it should be \#1: 2.

#### 5c - Question Answer

_Correct_. Max pooling takes the output of each hidden unit in the map and picks the maximum value among these. In this case, the maximum value is 2 and it comes from hidden unit $$h_1$$. The neurons on the layer above therefore only see that some hidden unit produced an output of 2. They do not know which hidden unit among the 4 caused this, and therefore they lose the ability to distinguish where interesting things happen in the image. They only know that something interesting happened somewhere.

In general, we don't pool over the entire image, but instead we pool over regions. For example, we might pool over $$h1, h2 \text{ and } h3, h4$$ to create two outputs. This lets the network build up progressively more invariant features with each successive layer.

### 5c05 - Le Net

5:55

Yann LeCun

### ![](/assets/hinton-lec5c-lecun-minst-architecture.png)5c06 - The 82 errors made by LeNet5

### ![](/assets/hinton-lec5c-leNet-errors.png)5c07 - Priors and Prejudice

tbd

### 5c08 - The brute force approach

tbd

### 5c09 - The errors made by the Ciresan et. al. net

tbd ![](/assets/hinton-lec5c-Ciresan-errors.png)How to detect a significant drop in the error rate

tbd

## 5d01 - Convolutional Neural Networks For Object Recognition

TBD

### 5d02 - From hand-wriHen digits to 3-D objects

TBD

### 5d03 - The ILSVRC-2012 competition on ImageNet

TBD

### 5d04 - Examples from the test set \(with the network’s guesses\)

TBD

### 5d05 - Error rates on the ILSVRC-2012 competition

TBD

### 5d06 - A neural network for ImageNet

TBD

### 5d07 - Tricks that significantly improve generalization

TBD

### 5d08 - The hardware required for Alex’s net

TBD

### 5d09 - Finding roads in high-resolution images

![](/assets/hinton-lec5d-roadfinder.jpg)

# Week 5 Quiz

TBD

# Week 5 Vocab

_weight filter \[matrix, applied to CNN, used in 5c question\]_: asdf

_filter map \[used in 5c question\]_: asdf

_convolutional neural network_: asdf

# Week 5 FAQ

* TBD



