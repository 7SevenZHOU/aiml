# Week 3

[Link to Week in Coursera](https://www.coursera.org/learn/neural-networks/home/week/3)

## 3a - Linear Neuron Weight Learning

### 3a Intro - Why Not Perceptron?

* every step, weights use averaging to get closer to "generously feasible" good weights; weights always get closer to good set of weights
* but in some networks, an average of two good solutions is a worse solution
* \(as a result?\) "multi-layer" NNs don't use perceptron

* Alternative approach: get output values to move closer to target

  * works for non-convex problems
  * where there exists several independent sets of good weights
  * where averaging good weights may give bad set of weight

* Simplest example: linear neuron with squared error

### 3a Intro - Linear Neurons

### ![](/assets/linear-neurons-def1.png) {#linear-neuron-def1}

* also called _Linear Filters_

* error: squared difference between target and produced output

  * different from perceptron because error there was for weights

* we'll take iterative method rather than solving using calculus because it's easier to generalize to n leveled problems and because it's probably  more like how it happens biologically

### 3a Example - Fish, Chips Ketchup

* Fish, chips, ketchup. Every day you order relatively random amounts for lunch and find out total price at end.

* Eventually, you will figure out  price of fish, chips, ketchup individually by knowing how much of each you ordered and the  amount that the total changed.

* You order \(2, 5, 3\) for \(fish, chips, ketchup\), which costs 850, the target. The correct prices are \(150, 50, 100\), but you don't know that.

* First, you guess \(50, 50, 50\) for the prices / weights. This would give a total of \(50, 50, 50\) \* \(2, 5, 3\) = 500. This means that the _residual error_ is 800 - 500 = 350.

* _Delta rule_ for learning prices / weights is $$\Delta w_i = \varepsilon x_i(t-y)$$, where i is the index of the lunch ingredient, $$\varepsilon$$ is the learning rate of 1/35 \(chosen by you\), $$x_i$$ is the count of the number of the $$i$$th things you order, $$t$$ is the target price of the entire meal, and $$y$$ is the output generated in this iteration of the model given by \(50, 50, 50\) here.

* This gives new weights of \(1/35\)\(2, 5, 3\)\(50, 50, 50\)\(350\) = 10\*\(2, 5, 3\)\(50, 50, 50\) = \(70, 100, 80\)

  * weight for chips got worse - this is different than perceptron

### 3a Iterative Learning Procedure - Behavior

* There may not be a perfect solution.
* By making learning rate small, we can get closer to good enough solution. 
* If all dimensions are highly correlated, it will take much longer. 
* vs. Perceptrons
  * in perceptrons, we change the weight vector by input vector _only when an error of a certain size is made \(&gt;= generously feasible\)_
  * in _online_ version of delta-rule, we continually change weight vector by _error scaled by learning rate_. 

## 3b - Linear Neuron Error Surface

Visualizing simple linear network error surface:

![](/assets/error-surface-linear-neuron.png)

* horizontal axis for _each_ weight \(why not?\)
* vertical error axis
* plotting horizontal axis against vertical axis makes a bowl
* plotting weights against each other makes elliptical contours; inner ellipse is at bottom of error bowl

For multi-layer, non-linear nets the error surface is more complicated than this single bowl.

### 3b question

* Suppose we have a dataset of two training points.

  $$\begin{align*} x_1 &= (1, -1) & t_1&=0 \\ x_2 &= (0, 1) & t_2&=1 \end{align*}$$

* Consider a network with two input units connected to a linear neuron with weights $$w=(w_1, w_2)$$. What is the equation for the error surface when using a squared error loss function?

* Hint: The squared error is defined as $$\frac{1}{2}(w^T x_1 - t_1)^2 + \frac{1}{2}(w^T x_2 - t_2)^2$$

  * A: $$E = \frac{1}{2}\left(w_1^2 + 2w_2^2 - 2w_1w_2 -2w_2 + 1\right)$$

  * B: $$E = \frac{1}{2}\left((w_1 - w_2 - 1)^2 + w_2^2\right)$$

  * C: $$E = \frac{1}{2}\left(w_1^2 - 2w_2^2 + 2w_1w_2 + 1\right)$$

  * D: $$E = \frac{1}{2}(w_1 - w_2)^2$$

* First guess:

  * Squared error loss function is the type applicable to the error surface just described.

  * Here, we're given two data points and we have to pick the equation of the error surface given the relationship between them.

  * 0.5\(\(w1, w2\)\(1, -1\) - 0\)^2 + 0.5\(\(w1, w2\)\(0, 1\) -1\)^2

  * 0.5\(1w1, -1w2\)^2 + 0.5\(0w1-1, 1w2-1\)^2

  * 0.5\(w1^2, w2^2\) + 0.5\(-1^2, \(w2-1\)^2\)

  * 0.5\(w1^2, w2^2\) + 0.5\(1, w2^2 - 2w2 + 1\)

  * 0.5\(w1^2+1, 2w2^2 - 2w2 + 1\)

  * My vector arithmetic is rusty, but I don't understand how you can use the squared error equation to arrive at a scalar

  * If one were able to add the two vector components, it would be: 0.5\(w1^2 + 2w2^2 - 2w2 + 2\), which is very close to answer A

* Correct Answer: A

  > The error summed over all training cases is  
  > $$E = \frac{1}{2}(w_1  -w_2 - 0)^2 + \frac{1}{2}(0w_1 + w_2 - 1)^2 = \frac{1}{2}\left(w_1^2 + 2w_2^2 - 2w_1w_2 -2w_2 + 1\right)$$
  >
  > Note the quadratic form of the energy surface. For any fixed value of E, the contours will be ellipses. For fixed values of w1, it's a parabolic relation between E and w2. Similarly for fixed values of w2."

* Somehow, you are able to just add w1 to w2 to arrive at E

### 3b - Online vs batch learning

Simplest _batch learning_ - do _steepest descent_, traveling perpendicular to the contour lines to the bottom of the bowl

Simplest _online learning_ - zig-zag around direction of steepest descent:![](/assets/linear-neuron-online-zigzag-constraints.png)

* It looks like paths are determined by movements that are perpendicular to constraints
* I'm not sure where the constraints come from in this case. I believe you triangulate them by plotting the intersection from two points on the same contour.

## 3c - Logistic Output Neuron Weight Learning

## 3d - Backpropagation Algorithm

## 3e - Using Backpropagation Algorithm Derivatives



