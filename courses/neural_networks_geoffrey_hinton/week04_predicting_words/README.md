# Week 4 - Predicting Words

## Week 4 - Links

[Lecture Slides](https://d18ky98rnyall9.cloudfront.net/_4bd9216688e0605b8e05f5533577b3b8_lec4.pdf?Expires=1491955200&Signature=YYhlbLG4XsdPuiceHDrXNMJfTdzGApJK11GhS1Tkbq1nIvVv~0G4ZVtvnfSE-LfAOBmQ0R29P8zJt7qpxY5OdRv7ynlO~sht6h~Ah5uz7PwIcwXYNRURkfC1~zKlZsBLh2v~K7Iu8-joqGmdVtlg-5YwCF7-n4cchMtVOexxBWU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

[Week In Coursera](https://www.coursera.org/learn/neural-networks/lecture/68Koq/another-diversion-the-softmax-output-function-7-min)

## 4a - Learning To Predict The Next Word

### Family Tree Example Intro

![](/assets/4a-family-tree.png)alternative way to express family tree is to make a set of triples using 12 relationships connecting entities: son, daughter, nephew, niece, father, mother, uncle, aunt, brother, sister, husband, wife

\(colin has-father james\) & \(colin has-mother victoria\) → \(james has-wife victoria\)

\(charlotte has-brother colin\) & \(victoria has-brother arthur\) → \(charlotte has-uncle arthur\)

relational learning task: given a large set of triples that come from some family trees, figure out the regularities.

![](/assets/course-hinton-4a-family-tree-nn.png)![](/assets/course-hinton-4a-family-tree-nn-result.png)

The six hidden units in the _bottleneck_ connected to the input representation of person 1 learn _nationality_, _generation_, _branch of family tree_

These features are only useful if the other _bottlenecks_ use similar representations and the central layer learns how features predict other features: \(Input person is of generation 3\) & \(relationship requires answer to be one generation up\) → \(Output person is of generation 2\)

If trained on eight of the relationship types, then tested on the remaining four, it gets answers 3/4 correct, which "is good for a 24-way choice."

* How is 24 computed here?

On "much bigger" datasets we can train on "a much smaller fraction" of the data.

Suppose we have millions of relational facts in form \(A R B\).

* We can train a neural net to discover a feature vector representations of the terms that allow the third term to be predicted from the first two.
* We can use the trained net to find very unlikely triples. These are good candidates for errors in the database.
* Instead of predicting third term we could use all three as input and predict the probability that the fact is correct. To do this we would need a good source of false facts.  

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


Essentially, the output _yi_ is the _zi_ over the sum over all _zi\_s in the softmax group, except where each is expressed as a power function of \_e_. So _yi_ is always between zero and one.

Softmax has simple output derivatives, though not that trivial to derive:


$$
\frac{\delta y_i}{\delta z_i}=y_i(1-y_i)
$$


#### Question

If $$\mathbf{z} = (z_1, z_2, \ldots z_k)$$ is the input to a k-way softmax unit, the output distribution is $$\mathbf{y}=(y_1, y_2, \ldots y_k)$$, where $$y_i = \dfrac{\exp(z_i)}{\sum_j\exp(z_j)}$$, which of the following statements are true?

1. The output distribution would still be the same if the input vector was _c_**z**_ \_for any positive constant_ c\_. 
2. The output distribution would still be t he same if the input vector was _c _+ **z** for any positive constant _c_. 
3. Any probability distribution _P_ over discrete states $$P(x) > 0  \ \ \forall x$$ can be represented as the output of a softmax unit for some inputs.
4. Each output of a softmax unit always lies in \(0,1\).

#### Work

1. If you scale z, then you change the denominator much more than the numerator, so that will change the distribution. False. _Correct_
2. If you add a constant to each term, that should not affect the distribution. True. 
   1. _Correct. \_Let's say we have two z's: z1=2, z2=-2. Now let's take a softmax over them: _$$\frac{\exp(z_1)}{\exp(z_1) + \exp(z_2)}=\frac{\exp(2)}{\exp(2)+\exp(-2)}$$. If we add some positive constant \_c\_ to each $$z_i$$ then this becomes:
      $$\frac{\exp(2+c)}{\exp(2+c) + \exp(-2+c)}=\frac{\exp(2)\exp(c)}{(\exp(2)+\exp(-2))\exp(c)}=\frac{\exp(2)}{\exp(2)+\exp(-2)}$$.
      Multiplying each $$z_i$$ by _c_ gives:
      $$\frac{\exp(2c)}{\exp(2c) + \exp(-2c)}=\frac{\exp(2)^c}{\exp(2)^c + \exp(-2)^c} \neq \frac{\exp(2)}{\exp(2)+\exp(-2)}$$
3. I see no reason why any probability distribution over discrete states could not be represented by softmax. True. _Correct_
4. True. _Correct_

### Cross-Entropy Is Cost Function To Use With Softmax

Question: if we're using a softmax group for the outputs, what's the cost function?

Answer: one that gives the negative log probability of right answer

Cross-entropy cost function for _each_ output assumes that the for the unit under consideration, the target value is one, and all other target values for _j_ are zero. Then take the sum of the targets times the outputs for _all_ outputs.


$$
C=-\sum_{j} t_j \log{y_j}
$$


$$t_j$$ is the target value under consideration.

This is called the cross-entropy cost function

C has very big gradient when target value is zero and output is almost zero

* value of 0.000001 \(1/M\) is much better than 0.000000001 \(1/B\)
* cost function _C_ has a very steep derivative when answer is very wrong 
* the steepness of the derivative of the cost function with respect to the neuron's output, dC/dy balances the flatness of the derivative of the output with respect to the logit, dy/dz.


$$
\frac{\delta C}{\delta z_i}=\sum_j \frac{\delta C}{\delta y_j}\frac{\delta y_j}{\delta z_i}=y_i-t_i
$$


* For a given neuron _i_, the derivative of the cost function with respect to the logit is just the difference between the output and the target output at that neuron
* When actual target outputs are very different, that has a slope of one or minus one, and slope is never bigger than one or minus one. 
* Slope never gets small until two things are pretty much the same, in other words, you're getting pretty much the right answer.

## 4d - Neuro-Probabilistic Language Models

We're use our understanding to hear the correct words

* "We do this unconsciously when we wreck a nice beach."

### Trigram Method

$$\frac{p(w_3=c \mid w_2=b, w_1=a)}{p(w_3=d \mid w_2=b, w_1=a)}=\frac{count(abc)}{count(abd)}$$

* count freq of all triples of words in a corpus
* use freqs to make bets on relative prob of words given two previous words
* was state of art until recently
* cannot use much bigger context because too many possibilities to store and counts would mostly be zero
  * too many - really?
* Dinosaur pizza ... then what? 
  * "back-off" to digrams when count for trigram is too small

### Trigram Limitations

* example: “the cat got squashed in the garden on friday”
  * should help us predict words in “the dog got flattened in the yard on monday”
* does not understand similarities between ...
  * cat/dog; squashed/flattened; garden/yard; friday/monday

### Yoshua Bengio Next Word Prediction![](/assets/bengio-predicting-next-word.png)

* put in candidate for third word, then get output score for how good that candidate word is in the context.

* run forwards through net many times, one for each candidate.

* input context to the big hidden layer is the same for each candidate word.

### The Problem With 100k Output Words

* Each unit layer A above has 100k outgoing weights
  * so can't afford to have many hidden units
    * unless we have huge num training cases
      * _why?_
  * We could make the last hidden layer small, but then it's hard to get the 100k probabilities correct; small probabilities are often important

## 4e - Dealing With Many Possible Outputs In Neuro-Probab. Lang. Models

Someone on the board recommended reading [Minih and Hinton, 2009](http://www.cs.toronto.edu/~hinton/absps/andriytree.pdf) for this part of the lecture

### Serial Architecture

![](/assets/4e-serial-arch-for-word-discovery.png)Trying to predict next word or middle word of string of words.

Put the candidate in with the its context words as before.

Go forwards through net, and then give score for how good that vector is in the net.

After computing logit score for each candidate word, use all logits in a softmax to get word probabilities.

The difference between the word probabilities and their target probabilities gives cross-entropy error derivatives.

* The derivatives try to raise the score of the correct candidate and lower the scores of its high-scoring rivals.

We can save tie if we only use a small set of candidates suggested by some other predictor, for example, revising the probabilities of the words the trigram model thinks are likely \(a second pass\).

### Predicting Path Through Probability Tree

Based on Minih and Hinton, 2009

![](/assets/minh-and-hinton-08.png)Arrange all words in binary tree with words as leaves

Use previous context to generate _prediction vector, _**v**.

Compare v with a learned vector u at each node.

Apply logistic function to scalar product of **u** and **v** to predict probabilities of taking the two branches of the tree.

### Simpler Way To Learn Features For Words

Collobert and Weston, 2008

![](/assets/4e-collobert-and-weston-2008.png)By displaying on 2-D map, we can get idea of quality of learned feature vectors

Easy substitutes area clustered near one another. Example Image:

![](/assets/4e-turian-t-sne-sample.png)

Multi-scale method **t-sne** displays similar clusters near each other, too

* no extra supervision
* information is all in the context; some people think we learn words this way
* "She scrommed him with the frying pan" - does it mean bludgeoned or does it mean impressed him with her cooking skills?



