# Neural Networks with Geoffrey Hinton

## Octave Notes

* [Octave Language Reference](https://www.gnu.org/software/octave/doc/interpreter/)

### Matrices - Basic

* from [A Programmer's Guide to Octave](http://www.i-programmer.info/programming/other-languages/4779-a-programmers-guide-to-octave.html?start=1)
```octave
% row vector
A=[1,2,3]

% column vector - think of semicolons as newlines
A=[1;2;3]

% matrix multiplication
A=[1,2,3;4,5,6]
B=[7,8;9,10;11,12]
A*B

% scalar multiplication
C=10
C*A

% element by element multiplication-put a dot before the operator.
% For example
A.*B
% performs an element-by-element multiplication of the two matrices 
% and not a matrix multiplication i.e. a_{ij}*b_{ij}.

% single quote to "complex transpose" a matrix
% following is a column matrix:
A=[1,2,3]'

You can use the inverse function to find the inverse of any square non-singular matrix. For example
A=[1,2;3,4]
B=inverse[A]
A*B
displays the identity matrix.

The expression  x\y is the left division of y by x and is equivalent to
inverse(x)*y
The advantage of using this notation is that the inverse isn't actually used in the calculation.

The expression x/y is the right division of x by y and it is equivalent to
x*inverse(y)
Again the inverse matrix is never computed and generalized inverses are used if necessary

A(1,2)
is the value in row 1 column 2. You can remember this because that's 
how they are addressed in math: 2x3 matrix means two rows of three columns.

You can assign a new value to a single element e.g.
A(1,2)=3

A vector of indexes just picks out the combined set of elements that 
each index would pick out. For example:
A([1,2],1)
picks out A(1,1) and A(2,1) and the result is a column vector because 
you have specified part of a column of the original matrix.
```
* *Vector Index* to a matrix:
  > A vector of indexes just picks out the combined set of elements 
  > that each index would pick out. For example, `A([1,2],1)`
  > picks out `A(1,1)` and `A(2,1)` and the result is a column vector 
  > because you have specified part of a column of the original matrix.
* *Range Index* to a matrix: 
  > In general a range is specified as `start:increment:end`
  > and if you leave out the increment it is assumed to be 1 and the range is
  > `start:end`. The increment can be negative.


