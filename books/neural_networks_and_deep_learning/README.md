# Notes for Neural Networks and Deep Learning

## [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)

## [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

### [Warm up: a fast matrix-based approach to computing the output from a neural network](http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output _from_a_neural_network)

### [The two assumptions we need about the cost function](http://neuralnetworksanddeeplearning.com/chap2.html#the_two_assumptions_we_need_about_the_cost_function)

### [The Hadamard product](http://neuralnetworksanddeeplearning.com/chap2.html#the_hadamard_product_$s_\odot_t$)
$$s \odot t$$:
  * the *elementwise* product of two vectors $$ s $$ and $$ t $$
  * called *Hadamard product* or *Schur product*

### [The four fundamental equations behind backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation)

* measure of error
  * $$
      \delta^l_j =
      \frac{\partial C}{\partial z^l_j}
    $$ - here delta is the "inexact derivative" or incremental amount changed in 
    hidden neuron $$ j $$ at layer $$ l $$, and it's set to be the partial 
    derivative of the Cost function w.r.t the logistic logit at that neuron.

### [Proof of the four fundamental equations (optional)](http://neuralnetworksanddeeplearning.com/chap2.html#proof_of_the_four_fundamental_equations_(optional))

### [The backpropagation algorithm](http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm)

### [The code for backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html#the_code_for_backpropagation)

### [In what sense is backpropagation a fast algorithm?](http://neuralnetworksanddeeplearning.com/chap2.html#in_what_sense_is_backpropagation_a_fast_algorithm)

### [Backpropagation: the big picture](http://neuralnetworksanddeeplearning.com/chap2.html#backpropagation_the_big_picture)


## [Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html)

## [A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)

## [Why are deep neural networks hard to train?](http://neuralnetworksanddeeplearning.com/chap5.html)

## [Deep learning](http://neuralnetworksanddeeplearning.com/chap6.html)



