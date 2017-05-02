# Week 7 - Modeling Sequences with RNNs

## 7a 
### 7a-02 - 
### 7a-03 - 
### 7a-04 - 
### 7a-05 - 
### 7a-06 - 
### 7a-07 - 

## 7b - 
### 7b-02 -
### 7b-03 - 
### 7b-04 - 
### 7b-05 - 
### 7b-06 - 
### 7b-07 - 
### 7b-08 - 


## 7c - 
### 7c-02 -
### 7c-03 - 
### 7c-04 - 
### 7c-05 - 
### 7c-06 - 


## 7d - 
### 7d-02 -
### 7d-03 - 
### 7d-04 - 


## 7e - Long term short term memory
### 7e-02 - Long Short Term Memory (LSTM)
### 7e-03 - Implementing a memory cell in a neural network
### 7e-04 - Backpropagation through a memory cell
* At initial time, let's asume that keep gate is 0 and write gate is 1
* value of 1.7 from rest of NN is set to 1.7. 
* rest of NN has to set the 

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
