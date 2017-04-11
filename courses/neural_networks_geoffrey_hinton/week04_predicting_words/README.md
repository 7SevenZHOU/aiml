# Week 4 - Predicting Words

## Week 4 - Links

[Lecture Slides](https://d18ky98rnyall9.cloudfront.net/_4bd9216688e0605b8e05f5533577b3b8_lec4.pdf?Expires=1491955200&Signature=YYhlbLG4XsdPuiceHDrXNMJfTdzGApJK11GhS1Tkbq1nIvVv~0G4ZVtvnfSE-LfAOBmQ0R29P8zJt7qpxY5OdRv7ynlO~sht6h~Ah5uz7PwIcwXYNRURkfC1~zKlZsBLh2v~K7Iu8-joqGmdVtlg-5YwCF7-n4cchMtVOexxBWU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

[Week In Coursera](https://www.coursera.org/learn/neural-networks/lecture/68Koq/another-diversion-the-softmax-output-function-7-min)

## 4a - Learning To Predict The Next Word

## 4b - A Brief Diversion Into Cognitive Science

## 4c - The Softmax Output Function

Squared error cost function has drawbacks:

* There is almost no gradient if desired output is 1 and actual output is 0.00000001
* If we are trying to pick between mutually exclusive classes and we know probabilities should sum to one, no way to encode that

Softmax is a cost function that is an alternative to squared error

It is applied to a group of neurons called _a softmax group_

Inputs to neurons in a softmax group are called _logits_

In the group, the outputs all sum to one

Outputs given by:


$$
y_i = \frac{e^{z_i}}{\sum_{j \in group} e^{z_j}}
$$


Simple output derivatives like derivative of logistic unit, though not as trivial to derive:


$$
\frac{\delta y_i}{\delta z_i}=y_i(1-y_i)
$$


### Cross-Entropy Is Cost Function To Use With Softmax

Question: if we're using a softmax group for the outputs, what's the cost function?

Answer: one that gives the negative log probability of right answer


$$
C=-\sum_{j} t_j \log{y_j}
$$


Here, $$t_j$$ is the target value.

This is called the cross-entropy cost function

* C has very big gradient when target value is zero and output is almost zero
* value of 0.000001 \(1/M\) is much better with this cost function than 0.000000001 \(1/B\), unlike squared error
* nice property: cost function _C _has very steep derivative when answer is wrong. 
* output wrt input is flat when answer is wrong. When you multiply the two together, using the chain rule, giving how fast the cost function changes times how fast the output of a unit changes as you change zi, the result is the actual input minus the target output.
* the steepness of the derivative of the cost function with respect to the neuron's output, dC/dy balances the flatness of the derivative of the output with respect to the logit, dy/dz.


$$
\frac{\delta C}{\delta z_i}=\sum_j \frac{\delta C}{\delta y_j}\frac{\delta y_j}{\delta z_i}=y_i-t_i
$$


* For a given neuron _i_, the derivative of the cost function with respect to the logit is just the difference between the output and the target output at that neuron
* When actual target outputs are very different, that has a slope of one or minus one, and slope is never bigger than one or minus one. 
* Slope never gets small until two things are pretty much the same, in other words, you're getting pretty much the right answer.

## 4d - Neuro-Probabilistic Language Models

## 4e - Dealing With Many Possible Outputs In Neuro-Probab. Lang. Models



