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

* * _Delta rule_ for learning prices / weights is $$\Delta w_i = \varepsilon x_i(t-y)$$, where i is the index of the "lunch ingredient," $$\varepsilon$$ is the learning rate of 1/35 \(chosen by you\), $$x_i$$ is the count of the number of the $$i$$th things you order, $$t$$ is the target price of the entire meal, and $$y$$ is the output generated in this iteration of the model given by \(50, 50, 50\) here.
  * This gives new weights of \(1/35\)\(2, 5, 3\)\(50, 50, 50\)\(350\) = 10\*\(2, 5, 3\)\(50, 50, 50\) = \(70, 100, 80\)
    * weight for chips got worse - this is different than perceptron

## 3b - Linear Neuron Error Surface

## 3c - Logistic Output Neuron Weight Learning

## 3d - Backpropagation Algorithm

## 3e - Using Backpropagation Algorithm Derivatives



