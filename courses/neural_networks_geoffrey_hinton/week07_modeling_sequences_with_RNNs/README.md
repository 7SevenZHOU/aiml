# Week 7 - Modeling Sequences with RNNs

## 7a - Modeling sequences: A brief overview
### 7a-02 - Getting targets when modeling sequences
### 7a-03 - Memoryless models for sequences
### 7a-04 - Beyond memoryless models
### 7a-05 - Linear Dynamical Systems (engineers love them!)
### 7a-06 - Hidden Markov Models (computer scientists love them!)
### 7a-07 - A fundamental limitation of HMMs
### 7a-08 - Recurrent neural networks
### 7a-09 - Do generative models need to be stochastic?
### 7a-10 - Recurrent neural networks


## 7b - Training RNNs with backpropagation
### 7b-02 - The equivalence between feedforward nets and recurrent nets
### 7b-03 - Reminder: Backpropagation with weight constraints
### 7b-04 - Backpropagation through time
### 7b-05 - An irritating extra issue
### 7b-06 - Providing input to recurrent networks
### 7b-07 - Teaching signals for recurrent networks


## 7c - A toy example of training an RNN
### 7c-02 - A good toy problem for a recurrent network
### 7c-03 - The algorithm for binary addition
### 7c-04 - A recurrent net for binary addition
### 7c-05 - The connectivity of the network
### 7c-06 - What the network learns


## 7d - Why it is difficult to train an RNN
### 7d-02 - The backward pass is linear
### 7d-03 - The problem of exploding or vanishing gradients
### 7d-04 - Why the back-propagated gradient blows up
### 7d-05 - Four effective ways to learn an RNN


## 7e - Long term short term memory
### 7e-02 - Long Short Term Memory (LSTM)
### 7e-03 - Implementing a memory cell in a neural network
### 7e-04 - Backpropagation through a memory cell
* At initial time, let's assume that keep gate is 0 and write gate is 1
* value of 1.7 from rest of NN is set to 1.7. 

### 7e-05 - Reading cursive handwriting
* natural task for RNN
* usually, input is sequence of (x, y, p) coordinates of the tip 
  of the pen, where p indicates pen up or pen down
* output is sequence of characters.

### 7e-06 - A demonstration of online handwriting recognition by an RNN with Long Short Term Memory
* from Alex Graves
* [movie demo](https://www.youtube.com/watch?v=mLxsbWAYIpw):
* **Row 1**: This shows when the characters are recognized.
  – It never revises its output so difficult decisions are more delayed.
* **Row 2**: This shows the states of a subset of the memory cells.
  – Notice how they get reset when it recognizes a character.
* **Row 3**: This shows the writing. The net sees the x and y coordinates.
  – Optical input actually works a bit better than pen coordinates.
* **Row 4**: This shows the gradient backpropagated all the way to the x and
  - y inputs from the currently most active character.
  – This lets you see which bits of the data are influencing the decision.


# Week 7 Quiz

TBD

# Week 7 Vocab

* *Attenuation*:
* *Attractors*: 

# Week 7 FAQ

TBD

# Week 7 Other

## Papers

TBD

## Week 7 Links
TBD

## Week 7 People

### Hochreiter 

### Schmidhuber

### Alex Graves
* Graves & Schmidhuber (2009) showed that RNNs with LSTM
  are currently the best systems for reading cursive writing.
