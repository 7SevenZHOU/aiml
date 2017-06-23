# Assignment 1

## Problem 1
(Chapter 1, Problem 12) Four players, named A, B, C, and D, are playing a card game. A standard, well-shuffled deck of 
cards is dealt to the players (so each player receives a 13-card hand).

a. How many possibilities are there for the hand that player A will get? (Within a hand, the order in which cards were 
received doesn’t matter.)
b. How many possibilities are there overall for what hands everyone will get, assuming that it matters which player 
gets which hand, but not the order of cards within a hand?

Explain intuitively why the answer to Part (b) is not the fourth power of the answer to Part (a).

## Problem 2
(Chapter 1, Problem 13) A certain casino uses 10 standard decks of cards mixed together into one big deck, which we 
will call a superdeck. Thus, the superdeck has 52 · 10 = 520 cards, with 10 copies of each card. How many different 
10-card hands can be dealt from the superdeck? The order of the cards does not matter, nor does it matter which of 
the original 10 decks the cards came from. Express your answer as a binomial coefficient. Hint: Bose-Einstein.

## Problem 3
(Chapter 1, Problem 17) Give a story proof that

$$ \sum_{k=1}^{n} k \binom{n}{k}^2 = n \binom{2n-1}{n-1} $$

for all positive integers n.
Hint: Consider choosing a committee of size n from two groups of size n each, where only one of the two groups has 
people eligible to become president.

## Problem 4
(Chapter 1, Problem 20) The Dutch mathematician R.J. Stroeker remarked: Every beginning student of number theory 
surely must have marveled at the miraculous fact that for each natural number n the sum of the first n positive 
consecutive cubes is a perfect square. Furthermore, it is the square of the sum of the first n positive integers! 
That is, 

$$ 1^3 + 2^3 + ... + n^3 = ( 1 + 2 + ... + n)^2 $$

Usually this identity is proven by induction, but that does not give much insight into why the result is true, 
nor does it help much if we wanted to compute the left-hand side but didn’t already know this result. In this 
problem, you will give a story proof of the identity.

a. Give a story proof of the identity   
   $$ 1 + 2 + · · · + n = \binom{n + 1}{2} $$.

b. Give a story proof of the identity
   $$ 1^3 + 2^3 + ··· + n^3 = 6 \binom{n + 1}{4} + 6 \binom{n + 1}{3} + \binom{n + 1}{2} $$
   
It is then just basic algebra (not required for this problem) to check that the square of the right-hand side in

(a) is the right-hand side in (b).

Hint: Imagine choosing a number between 1 and n and then choosing 3 numbers between 0 and n smaller 
than the original number, with replacement. Then consider cases based on how many distinct numbers were
chosen.

## Problem 5
(Chapter 1, Problem 30) Four cards are face down on a table. You are told that two are red and two are black,
and you need to guess which two are red and which two are black. You do this by pointing to the two cards
you’re guessing are red (and then implicitly you’re guessing that the other two are black). Assume that all
configurations are equally likely, and that you do not have psychic powers. Find the probability that exactly
j of your guesses are correct, for j = 0, 1, 2, 3, 4.