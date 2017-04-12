# Week 4 - Predicting Words

## Week 4 - Links

[Lecture Slides](https://d18ky98rnyall9.cloudfront.net/_4bd9216688e0605b8e05f5533577b3b8_lec4.pdf?Expires=1491955200&Signature=YYhlbLG4XsdPuiceHDrXNMJfTdzGApJK11GhS1Tkbq1nIvVv~0G4ZVtvnfSE-LfAOBmQ0R29P8zJt7qpxY5OdRv7ynlO~sht6h~Ah5uz7PwIcwXYNRURkfC1~zKlZsBLh2v~K7Iu8-joqGmdVtlg-5YwCF7-n4cchMtVOexxBWU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

[Week In Coursera](https://www.coursera.org/learn/neural-networks/lecture/68Koq/another-diversion-the-softmax-output-function-7-min)

## 4a - Learning To Predict The Next Word

## 4b - A Brief Diversion Into Cognitive Science

## 4c - The Softmax Output Function

Squared error cost function has drawbacks:

* There is almost no gradient if desired output is 1 and actual output is 0.00000001
  * it's way out on a plateau where the slope is almost exactly horizontal
  * it will take a long time to adjust, even though it's a big mistake
* If we are trying to pick between mutually exclusive classes and we know probabilities should sum to one, no way to encode that
  * there is no reason to deprive network of this information

_Softmax_ is a different cost function applied to a group of neurons called _a softmax group_. It is a "soft, continuous version of the maximum function."

$$z_i$$ is the total input from layer below, called the _logit._

$$y_i$$ is the output of each unit, which depends on the logits of the other units in the softmax group. In the group, the outputs all sum to one, because they are given by a formula that takes into account all of the logits of the group:


$$
y_i = \frac{e^{z_i}}{\sum_{j \in group} e^{z_j}}
$$


Essentially, the output _yi_ is the _zi_ over the sum over all _zi_s in the softmax group, except where each is expressed as a power function of _e_. So _yi_ is always between zero and one.

Softmax has simple output derivatives, though not that trivial to derive:


$$
\frac{\delta y_i}{\delta z_i}=y_i(1-y_i)
$$


#### Question

If $$\mathbf{z} = (z_1, z_2, \ldots z_k)$$ is the input to a k-way softmax unit, the output distribution is $$\mathbf{y}=(y_1, y_2, \ldots y_k)$$, where $$y_i = \dfrac{\exp(z_i)}{\sum_j\exp(z_j)}$$, which of the following statements are true?

1. The output distribution would still be the same if the input vector was _c_**z**_ _for any positive constant _c_. 
2. The output distribution would still be t he same if the input vector was _c _+ **z** for any positive constant _c_. 
3. Any probability distribution _P_ over discrete states $$P(x) > 0  \ \ \forall x$$ can be represented as the output of a softmax unit for some inputs.
4. Each output of a softmax unit always lies in \(0,1\).

#### Work

1. If you scale z, then you change the denominator much more than the numerator, so that will change the distribution. False. _Correct_
2. If you add a constant to each term, that should not affect the distribution. True. 
   1. _Correct. _Let's say we have two z's: z1=2, z2=-2. Now let's take a softmax over them:_ _$$\frac{\exp(z_1)}{\exp(z_1) + \exp(z_2)}=\frac{\exp(2)}{\exp(2)+\exp(-2)}$$. If we add some positive constant _c_ to each $$z_i$$ then this becomes:
      $$\frac{\exp(2+c)}{\exp(2+c) + \exp(-2+c)}=\frac{\exp(2)\exp(c)}{(\exp(2)+\exp(-2))\exp(c)}=\frac{\exp(2)}{\exp(2)+\exp(-2)}$$.
      Multiplying each $$z_i$$ by _c_ gives:
      $$\frac{\exp(2c)}{\exp(2c) + \exp(-2c)}=\frac{\exp(2)^c}{\exp(2)^c + \exp(-2)^c} \neq \frac{\exp(2)}{\exp(2)+\exp(-2)}$$
3. I see no reason why any probability distribution over discrete states could not be represented by softmax. True.
4. True. 

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
* nice property: cost function _C_ has a very steep derivative when answer is wrong. 
* output w.r.t. input is flat when answer is wrong. 
* the steepness of the derivative of the cost function with respect to the neuron's output, dC/dy balances the flatness of the derivative of the output with respect to the logit, dy/dz.


$$
\frac{\delta C}{\delta z_i}=\sum_j \frac{\delta C}{\delta y_j}\frac{\delta y_j}{\delta z_i}=y_i-t_i
$$


* For a given neuron _i_, the derivative of the cost function with respect to the logit is just the difference between the output and the target output at that neuron
* When actual target outputs are very different, that has a slope of one or minus one, and slope is never bigger than one or minus one. 
* Slope never gets small until two things are pretty much the same, in other words, you're getting pretty much the right answer.

## 4d - Neuro-Probabilistic Language Models

## 4e - Dealing With Many Possible Outputs In Neuro-Probab. Lang. Models



