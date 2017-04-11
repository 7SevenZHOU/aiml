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

$$y_i = \frac{e^{z_i}}{\sum_{j \in group} e^{z_j}}$$

Output Derivatives:

$$\frac{\delta y_i}{\delta z_i}=y_i(1-y_i)$$

This is not trivial to derive. 

## 4d - Neuro-Probabilistic Language Models

## 4e - Dealing With Many Possible Outputs In Neuro-Probab. Lang. Models



