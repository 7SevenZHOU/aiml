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

* I made a mistake between step 1 and step 2 when I moved -1 inside the parens on the right - I duplicated the -1, which is why I ended up with +2 instead of +1 `#offByOneError`

### 3b - Online vs Batch Learning

Simplest _batch learning_ - do _steepest descent_, traveling perpendicular to the contour lines to the bottom of the bowl. "We get the gradient, summed over all training cases."

Simplest _online learning_ - zig-zag around direction of steepest descent:![](/assets/linear-neuron-online-zigzag-constraints.png)

* _online learning_ - after each training case, change the weights in proportion to the error gradient for that single training case
* The change in the weights moves us towards a constraint plane. In picture above, there are two training cases: each of the blue lines. Training case one is at upper right.
* Start at outer red dot and compute the gradient on first training case using delta rule. This moves us perpendicularly towards the first constraint plane. 
* If we alternate between the training cases, we'll zigzag backwards and forwards between the two constraint planes until we intersect.

#### Learning Speed

* If picking a random starting point, it's ideal if the error space cross sections are more like circles than elongated ellipse. 
* If cross sections are circular, then the chances of picking a bad starting point are lower.
* If cross sections are elongated ellipses, it's possible for the direction to a constraint to lead away from the bottom of the error surface.
  * This happens when lines that correspond to training cases are almost parallel
  * "Nasty property" - gradient is big in direction we do not want to learn, and small in the direction we do want to learn, which is the bottom of the error bowl
  * The way I picture this is to review the unit circle and the graph of $$tan(\theta)$$
  * Here is [a Kahn Academy link to that](https://youtu.be/FK6-tZ5D7xM?t=387).
  * If the angle/slope between the two training cases is parallel, the amount of incorrect descent goes to infinity
  * If the angle/slope between the two training cases is perpendicular, the amount of incorrect descent is zero. This only happens when the error contours are circular.

#### Learning Speed Question:

_If our initial weight vector is _$$(w_1, 0)$$_ for some real number _$$w_1$$_, then on which of the following error surfaces can we expect steepest descent to converge poorly? Check all that apply._

| A: ![](/assets/error_surface_bad1.png) | B: ![](/assets/error_surface_nice.png) |
| :--- | :--- |
| C: ![](/assets/error_surface_nice2.png) | D: ![](/assets/error_surface_bad2.png) |

I guessed that A and D would converge poorly, since they were elongated ellipticals.

I do not know what the significance of the axis is or what the position of the gradient in the axis represents.

After submitting, I see that A was a correct answer but D was marked as incorrect.

The explanation for A is:

> _The first one is similar to the picture shown in the lectures. It is a diagonally oriented ellipse and steepest descent will still tend to zig-zag on this error surface. Even though the second and third **have different minima locations** and scalings, the steepest descent direction will still take you very close to the minimum with the appropriate learning rate. The last case is tricky: even though the shape is an ellipse, the **initial weight vector starts off somewhere along the x axis**, and so again the steepest descent direction points directly toward the minimum. In other words, **there is zero gradient along the vertical axis** and therefore we are simply minimizing a parabola along one dimension from that point to get to the minimum._

There are several things that I am lost on here:

* What are the "different minima locations" referred to in B and C?  
* When discussing D, he says "the initial weight vector starts somewhere along the x axis." Where is the initial weight vector? 
  * Going back to the problem statement, it is \(w1, 0\), so we know that w2 is zero in the initial point.
* I assume that the shape of the ellipse comes from two training cases, as in the lectures.

## 3c - Logistic Output Neuron Weight Learning

![](/assets/logistic-neuron-def1.png)Logistic neurons have an output that is a monotonically increasing function of its input. The output of the neuron is \(0, 1\) on \(-inf, inf\). Once the sum of the weights multiplied by the inputs approaches a certain threshold, the neuron's output continuously and quickly changes from zero to one.

The neuron's output is a function of the _logit z_, which is the biased sum of the product of the wights and inputs. The output y = 1/\(1 + e^-z\).

### Logistic Neuron Derivatives

Derivatives of the logit z with respect to the inputs and weights are simple: dz/dw\__i = x\_i; dz/dx\_i = w\_i. _

The derivative of whole logistic neuron w.r.t. the logit is simple: $$dy/dz=y(1-y)$$. The reason why:

$$y=\frac{1}{1+e^{-z}}=(1+e^{-z})^{-1}$$ by definition of negative power. So using the [power rule](https://www.khanacademy.org/math/ap-calculus-ab/basic-differentiation-ab/power-rule-ab/v/power-rule), $$\frac{dy}{dz}=-1(1+e^{-z})^{-2}=-1(1+\frac{1}{(1+e^{-z})^{2}})=\frac{-1e^{-z}}{(1+e^{-z})^{2}}-1$$. Using fraction addition, we end up with:

$$\frac{dy}{dz}=\frac{-1(-e^{-z})}{(1+e^{-z})^{2}}=(\frac{1}{1+e^{-z}})(\frac{e^{-z}}{1+e^{-z}})=y(1-y)$$

because $$\frac{e^{-z}}{1+e^{-z}}=\frac{(1+e^{-z})-1}{1+e^{-z}}=\frac{(1+e^{-z})}{1+e^{-z}}\frac{-1}{1+e^{-z}}=1-y$$

#### Learn Weights Using Chain Rule

* To learn the weights, we need derivative of output w.r.t. each weight:
  * $$\frac{\delta y}{\delta w_i}=\frac{\delta z}{\delta w_i}\frac{dy}{dz}=x_iy(1-y)$$, multiplying the derivative with respect to z by the derivative with respect to the ith weight, which is just x\_i
  * The derivative of the total error with respect to an individual weight,$$\frac{\delta E}{\delta w_i}$$ is equal to the sum of the derivatives of the outputs on the nth neuron w.r.t. its ith input multiplied by the derivative of the total error by the output of the nth neuron: 
  * $$ \sum_n \frac{\delta y^{n}}{\delta w_i}\frac{\delta E}{\delta y^{n}} $$, which is equal to the negative sum of \(x\)\(y\(1-y\)\)\(t-y\), 
    * _x_ refers to the _i_th input of the _n_th neuron 
    * _y_ refers to the output of the _n_th neuron
    * _t_ refers to the target output of the _n_th neuron
    * x\(t-y\) is the delta rule and y\(1-y\) is the extra term, which represents the slope of the logistic

## 3d - Backpropagation Algorithm

## 3e - Using Backpropagation Algorithm Derivatives



