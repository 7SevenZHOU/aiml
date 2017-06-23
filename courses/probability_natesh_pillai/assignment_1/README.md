# Assignment 1 Questions

## Problem 1
(Chapter 1, Problem 12) Four players, named A, B, C, and D, are
playing a card game. A standard, well-shuffled deck of cards is
dealt to the players (so each player receives a 13-card hand).

a. How many possibilities are there for the hand that player A will get?
(Within a hand, the order in which cards were received doesn’t matter.)

b. How many possibilities are there overall for what hands
everyone will get, assuming that it matters which player
gets which hand, but not the order of cards within a hand?

Explain intuitively why the answer to Part (b) is
not the fourth power of the answer to Part (a).

## Problem 2
(Chapter 1, Problem 13) A certain casino uses 10 standard decks
of cards mixed together into one big deck, which we will call a
superdeck. Thus, the superdeck has 52 · 10 = 520 cards, with 10
copies of each card. How many different 10-card hands can be dealt
from the superdeck? The order of the cards does not matter, nor
does it matter which of the original 10 decks the cards came from.
Express your answer as a binomial coefficient. Hint: Bose-Einstein.

## Problem 3
(Chapter 1, Problem 17) Give a story proof that

$$ \sum_{k=1}^{n} k \binom{n}{k}^2 = n \binom{2n-1}{n-1} $$

for all positive integers n. Hint: Consider choosing a committee
of size n from two groups of size n each, where only one
of the two groups has people eligible to become president.

## Problem 4
(Chapter 1, Problem 20) The Dutch mathematician R.J. Stroeker remarked:
Every beginning student of number theory surely must have marveled
at the miraculous fact that for each natural number n the sum of the
first n positive consecutive cubes is a perfect square. Furthermore,
it is the square of the sum of the first n positive integers! That is,

$$ 1^3 + 2^3 + ... + n^3 = ( 1 + 2 + ... + n)^2 $$

Usually this identity is proven by induction, but that does not give
much insight into why the result is true, nor does it help much if
we wanted to compute the left-hand side but didn’t already know this
result. In this problem, you will give a story proof of the identity.

a. Give a story proof of the identity   
   $$ 1 + 2 + · · · + n = \binom{n + 1}{2} $$.

b. Give a story proof of the identity
   $$ 
   1^3 + 2^3 + ··· + n^3 
   = 6 \binom{n + 1}{4} + 6 \binom{n + 1}{3} + \binom{n + 1}{2} 
   $$
   
It is then just basic algebra (not required for
this problem) to check that the square of the
right-hand side in (a) is the right-hand side in (b).

Hint: Imagine choosing a number between 1 and n and
then choosing 3 numbers between 0 and n smaller than
the original number, with replacement. Then consider
cases based on how many distinct numbers were chosen.

## Problem 5
(Chapter 1, Problem 30) Four cards are face down on a table. You are
told that two are red and two are black, and you need to guess which
two are red and which two are black. You do this by pointing to the two
cards you’re guessing are red (and then implicitly you’re guessing that
the other two are black). Assume that all configurations are equally
likely, and that you do not have psychic powers. Find the probability
that exactly j of your guesses are correct, for j = 0, 1, 2, 3, 4.

## Problem 6
(Chapter 1, Problem 43) Let A and B be events. The symmetric
difference $$ A \Delta B $$ is defined to be the set of all elements
that are in A or B but not both. In logic and engineering, this
event is also called the XOR (exclusive or) of A and B. Show that

$$ P( A \Delta B ) = P(A) + P(B) - 2P( A \cap B ) $$

## Problem 7
(Chapter 1, Problem 50) A certain class has 20 students, and
meets on Mondays and Wednesdays in a classroom with exactly 20
seats. In a certain week, everyone in the class attends both
days. On both days, the students choose their seats completely
randomly (with one student  per seat). Find the probability
that no one sits in the same seat on both days of that week.

## Problem 8
(Chapter 1, Problem 56) A widget inspector inspects 12 widgets
and finds that exactly 3 are defective. Unfortunately, the
widgets then get all mixed up and the inspector has to find
the 3 defective widgets again by testing widgets one by one.

(a) Find the probability that the inspector
will now have to test at least 9 widgets.

(b) Find the probability that the inspector
will now have to test at least 10 widgets.

## Problem 9 - Challenge Problem
Eight players P1 − P8 are playing in a tournament. Whenever
$$ P_i $$ plays $$ P_j $$ for $$ i > j $$, $$ P_i $$ wins. 
The tournament consists of 3 rounds. At every round, the 
players are chosen randomly. What is the probability that 
$$ P_1 $$ and $$ P_4 $$ play in the finals.

## R Exercises

### R Exercise 1
(R language basics) Given 
$$ x = (7.3, 6.8, 0.005, 9, 12, 2.4, 18.9, 0.9) $$,

(a) Calculate the mean of x.  
(b) Remove the mean from x.  
(c) Calculate the square root of each element in x.  
(d) Find the element in x which is larger than its square root.  
(e) Round the square root of each element in x to two decimals.

2. (Data frame and plot) We use data state.x77 in R for this problem.

(a) Select the country with highest Income.  
(b) Sort the data with respect to
variable Population in descending order.
(c) Give the scatter plot: Illiteracy VS Murder.
Please label the names of states on the plot.
(d) In the previous plot, text "high" on the
states with income higher than the 75% quantile.

3. (Function) Write a function to calculate the sample variance of a
vector $$ x = (x_1, · · · , x_n) $$. The formula is given as follows:

$$ \text{Var}(x) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \hat x)^2 $$