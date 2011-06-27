.. |observations| replace:: :math:`O = O_1, O_2, ..., O_T`
.. |states| replace:: :math:`S = \{S_1, S_2, ..., S_n\}`

Hidden Markov Model (HMM)
===========================

Consider a system which may be described at any time as being in one of a set
of :math:`n` distinct states, |states|. At regularly spaced
discrete times, the system undergoes a change of state (possibly back to the same
state) according to a set of probabilities associated with the state. We denote
the time instants associated with state changes as :math:`t = 1, 2, ...`, and
we denote the actual state at time :math:`t` as :math:`q_t`. A full probabilistic
description of the above system would, in general, require specification of the
current state (at time :math:`t`), as well as the predecessor states. For the
special case of a discrete, first order, Markov chain, this probabilistic
description is truncated to just the current and the predecessor state, i.e.

.. math::

   P(q_t=S_j|q_{t-1}=S_i, q_{t-2}=S_k) = P(q_t=S_j|q_{t-1}=S_i)

Furthermore, we only consider those processes in which the right-hand side of
the above is independent of time, thereby leading to the set of set of state
transition probabilities :math:`a_{ij}` of the form

.. math::

   a_{ij} = P(q_t=S_j|q_{t-1}=S_i) \quad 1 \le i, j \le n

with the state transition coefficients having the properties

.. math::

   a_{ij} \ge 0

.. math::

   \sum_{j=1}^{n}{a_{ij}} = 1

since they obey standard stochastic constraints.

Markov Model: An example
-------------------------

The above stochastic process could be called an observable Markov model since
the output of the process is the set of states at each instant of time, where
each each state corresponds to a physical (observable) event. To set ideas,
consider a simple 3-state Markov model of the weather. We assume that once a
day (e.g. at noon), the weather is observed as being one of the following:

   State 1: rain or (snow)

   State 2: cloudy

   State 3: sunny.

We postulate that the weather on day :math:`t` is characterized by a single one
of the three states above, and that the matrix :math:`A` of state transition
probabilities is

.. math::

   A = \{a_{ij}\} = \begin{pmatrix}
                   0.4 & 0.3 & 0.3 \\
                   0.2 & 0.6 & 0.2 \\
                   0.1 & 0.1 & 0.8 \\
                   \end{pmatrix}

Given that the weather on day 1 (:math:`t=1`) is sunny (state 3), we can ask
the question: What is the probability (according to the model) that the weather
for the next 7 days will be "sun-sun-rain-rain-sun-cloudy-sun"? Stated more
formally, we define the observation sequence :math:`O` as :math:`O = \{S_3, S_3, S_3, S_1, S_1, S_3, S_2, S_3\}`
corresponding to :math:`t=1, 2, ..., 8`, and we wish to determine the probability
of :math:`O`, given the model. This probability can be expressed (and evaluated)
as

.. math::

   p(O|Model) & = p(S_3, S_3, S_3, S_1, S_1, S_3, S_2, S_3|Model) \\
              & = p(S_3)p(S_3|S_3)p(S_3|S_3)p(S_1|S_3)p(S_1|S_1)p(S_3|S_1)p(S_2|S_3)p(S_3|S_2) \\
              & = \pi_{3} a_{33} a_{33} a_{31} a_{11} a_{13} a_{32} a_{23} \\
              & = 1 * (0.8) * (0.8) * (0.1) * (0.4) * (0.3) * (0.1) * (0.2) \\
              & = 1.536 \times 10^{-4}

where we use the notation

.. math::

   \pi_{i} = p(q_{1}=S_{i})

to denote the initial state probabilities.

Another interesting question we can ask (and answer using the model) is: Given
that the model is in a known state, what is the probability it stays in that
state for exactly :math:`d` days? This probability can be evaluated as the
probability of the observation sequence

.. math::

   O = \{ S_{i_{1}}, S_{i_{2}}, S_{i_{3}}, ..., S_{i_{d}}, ... S_{i_{d+1}} \}

given the model, which is

.. math::

   p(O|Model, q_1 = S_{i}) = (a_{ii})^{d-1}(1 - a_{ii}) = p_{i}(d)

The quantity :math:`p_{i}(d)` is the (discrete) probability density function of
duration :math:`d` in state :math:`i`. This exponential duration density is
characteristic of the state duration in a Markov chain. Based on :math:`p_{i}(d)`,
we can readily calculate the expected number of observations (duration) in a
state, conditioned on starting in that state as

.. math::

   \bar{d}_{i} & = \sum_{d=1}^{\infty}{dp_{i}(d)} \\
               & = \sum_{d=1}^{\infty}{d(a_{ii})^{d-1}(1-a_{ii})} = \frac{1}{1 - a_{ii}} \\

Thus the expected number of consecutive days of sunny weather, according to
the model, is :math:`1/(0.2) = 5`; for cloudy it is 2.5; for rain it is 1.67.

Extension to Hidden Markov Model
---------------------------------

So far we have considered Markov models in which each state corresponded to an
observable event. This model is too restrictive to be applicable to many problems
of interest. In this section we extend the concept of Markov models to include
the case where the observation is a probabilistic function of the state - i.e.,
the resulting model (which is called a hidden Markov model) is a doubly embedded
stochastic process with an underlying stochastic process that is not observable
(it is hidden), but can only be observed through another set of stochastic
processes that produce the sequence of observations.

Elements of an HMM
--------------------

An HMM is characterized by the following:

1. :math:`n`, the number of states in the model. ALthough the states are hidden,
   for many practical applications there is often some physical significance
   attached to the states or to sets of states of the model. Generally the states
   are interconnected in such a way that any state can be reached from any other
   state; however, we will see later in this paper that other possible
   interconnections of states are often of interest. We denote the individual
   states as |states|, and the state at time :math:`t`
   as :math:`q_t`.
2. :math:`m`, the number of distinct observation symbols per state, i.e., the
   discrete alphabet size. The observation symbols correspond to the physical
   output of the system being modeled. We denote the individual symbols as
   :math:`V = \{v_1, v_2, ..., v_m\}`.
3. The state transition probability distribution :math:`A = \{a_{ij}\}` where

   .. math::

      a_{ij} = p(q_{t+1}=S_{j}|q_{t}=S_{i}) \quad 1 \le i, j \le n

   For the special case where any state can reach any other state in a single
   step, we have :math:`a_{ij}>0` for all :math:`i,j`. For other types of HMMs,
   we would have :math:`a_{ij}=0` for one or more :math:`(i,j)` pairs.
4. The observation symbol probability distribution in state :math:`j, B= \{b_{j}(k)\}`,
   where

   .. math::

      b_{j}(k) = p(v_{k_{t}}|q_{t}=S_{j}) \quad 1 \le j \le n, 1 \le k \le m
5. The initial state distribution :math:`\pi = \{\pi_{i}\}` where

   .. math::

      \pi_{i} = p(q_{1}=S_{i}) \quad 1 \le i \le n

Given appropriate values of :math:`n, m, A, B, \pi`, the HMM can be used as a
generator to give an observation sequence :math:`O = O_1, O_2, ..., O_T`
(where each observation :math:`O_t` is one of the symbols from :math:`V`,
and :math:`T` is the number of observations in the sequence) as follows:

1. Choose an initial state :math:`q_1=S_i` according to the initial state
   distribution :math:`\pi`.
2. Set :math:`t=1`.
3. Choose :math:`O_t=v_k` according to the symbol probability distribution
   in state :math:`S_i`, i.e. :math:`b_i(k)`.
4. Transit to a new state :math:`q_{t+1}=S_{j}` according to the state transition
   probability distribution for state :math:`S_i`, i.e., :math:`a_{ij}`.
5. Set :math:`t=t+1`; return to step 3 if :math:`t<T`; otherwise terminate
   the procedure.

It can be seen from the above that a complete specification of an HMM requires
specification of two model parameters (:math:`n` and :math:`m`), specification
of observation symbols, and the specification of the three probability measures
:math:`A, B, \pi`. For convenience, we use the compact notation

.. math::

   \lambda = (A, B, \pi)

to indicate the complete parameter set of the model.

Thr three basic problems for HMMs
------------------------------------

*Problem 1*
   Given the observation sequence |observations|, and a model
   :math:`\lambda = (A, B, \pi)`, how do we efficiently compute :math:`p(O|\lambda)`,
   the probability of the observation sequence, given the model?
*Problem 2*
   Given the observation sequence |observations|, and the model
   :math:`\lambda`, how do we choose a corresponding state sequence
   :math:`Q = q_1, q_2, ... q_T` which is optimal in some meaningful sense
   (i.e., best "explains" the observations)?
*Problem 3*
   How do we adjust the model parameters :math:`\lambda = (A, B, \pi)` to
   maximize :math:`p(O|\lambda)`?

Solutions to the three basic problems of HMM
---------------------------------------------

Solution to Problem 1
**********************
We wish to calculate the probablility of the observation sequence |observations|,
given the model :math:`\lambda`, i.e., :math:`p(O|\lambda)`. The most straightforward
way of doing this is through enumerating every possible state sequence of length
:math:`T` (the number of observations). Consider one such fixed state sequence

.. math::

   Q = q_1, q_2, ..., q_T

where :math:`q_1` is the initial state. The probability of the observation
sequence :math:`O` for the state sequence of th above is

.. math::

   p(O|Q, \lambda) = \prod_{t=1}^{T}{p(O_t|q_t,\lambda)}

where we have assumed statistical independence of observations. Thus we get

.. math::

   p(O|Q,\lambda) = b_{q_{1}}(O_{1})b_{q_{2}}(O_{2})...b_{q_{T}}(O_{T})

The probability of such a state sequence :math:`Q` can be written as

.. math::

   p(Q|\lambda) = \pi_{q_{1}}a_{q_{1}q_{2}}a_{q_{2}q_{3}}...a_{q_{T-1}q_{T}}

The joint probability of :math:`O` and :math:`Q`, i.e., the probability that
:math:`O` and :math:`Q` occur simultaneously, is simply the produce of the
above two terms, i.e.,

.. math::

   p(O,Q|\lambda) = p(O|Q,\lambda)p(Q|\lambda)

The probability of :math:`O` (given the model) is obtained by summing this
joint probability over all possible state sequences :math:`q` giving

.. math::

   p(O|\lambda) & = \sum_{all\,Q}{p(O|Q,\lambda)p(Q|\lambda)} \\
                & = \sum_{q_{1},q_{2},...,q_{T}}{\pi_{q_{1}}b_{q_{1}}(O_{1})a_{q_{1}q_{2}}b_{q_{2}}(O_{2})...a_{q_{T-1}q_{T}}b_{q_{T}}(Q_{T})}

The interpretation of the computation in the above equation is the following.
Initially (at time :math:`t=1`) we are in state :math:`q_1` with probability
:math:`\pi_{q_{1}}`, and generate the symbol :math:`O_1` (in this state) with
probability :math:`b_{q_{1}}(O_{1})`. The clock changes from time :math:`t` to
:math:`t+1 (t=2)` and we make a transition to state :math:`q_2` from state
:math:`q_1` with probability :math:`a_{q_{1}q_{2}}`, and generation symbol
:math:`O_2` with probability :math:`b_{q_{2}}(O_{2})`. This process continues in
this manner until we make the list transition (at time :math:`T`) from state
:math:`q_{T-1}` to state :math:`q_T` with probability :math:`a_{q_{T-1}q_{T}}`
and generate symbol :math:`O_T` with probaiblity :math:`b_{q_{T}}(O_{T})`.

The *Forward-Backword Procedure*: Consider the forward variable :math:`\alpha_{t}(i)`
defined as

.. math::

   \alpha_{t}(i) = p(O_1, O_2, ..., O_t, q_t = S_i|\lambda)

i.e., the probability of the partial observation sequence, :math:`O_1, O_2, ..., O_t`,
(until time :math:`t`) and state :math:`S_i` at time :math:`t`, given the model
:math:`\lambda`. We can solve for :math:`\alpha_t(i)` inductively, as follows

1) Initialization:

   .. math::

      \alpha_1(i) = \pi_{i}b_{i}(O_{1}) \quad 1 \le i \le n
2) Induction:

   .. math::

      \alpha_{t+1}(j) = \left[\sum_{i=1}^{n}{\alpha_{t}(i)a_{ij}}\right]b_{j}(O_{t+1})
3) Termination:

   .. math::

      p(O|\lambda) = \sum_{i=1}^{n}{\alpha_T(i)}

In a similar manner, we can consider a backward variable :math:`\beta_{t}(i)`
defined as

.. math::

   \beta_{T}(i) = p(O_{t+1}, O_{t+2}, ..., O_{T}|q_{t}=S_{i},\lambda)

i.e., the probability of the partial observation sequence from :math:`t+1` to
the end, given state :math:`S_i` at time :math:`t` and the model :math:`\lambda`.
Again we can solve for :math:`\beta_{t}(i)` inductively, as follows:

1) Initialization:

   .. math::

      \beta_{T}(i) = 1 \quad 1 \le i \le n
2) Induction:

   .. math::

      \beta_{t}(i) = \sum_{j=1}^{n}{a_{ij}b_{j}(O_{t+1})\beta_{t+1}(j)} \quad t = T-1, T-2, ..., 1, 1 \le i \le n


Solution to Problem 2
**********************

Unlike Problem 1 for which an exact solution can be given, there are several
possible ways of solving Problem 2, namely finding the "optimal" state sequence
associated with the given observation sequence. The difficulty lies with the
definition of the optimal state sequence; i.e., there are several possible
optimal criteria. For example, one possible optimality criterion is to choose
the states :math:`q_{t}` which are *individually* most likely. This optimality
criterion maximizes the expected number of correct individual states. To implement
this solution to Problem 2, we define the variable

.. math::

   \gamma_{t}(i) = p(q_{t}=S_{i}|O,\lambda)

i.e., the probability of being in state :math:`S_{i}` at time :math:`t`, given
the observation sequence :math:`O`, and the model :math:`\lambda`. The above
equation can be expressed simply in terms of the forward-backward variables,
i.e.,

.. math::

   \gamma_{t}(i) = \frac{\alpha_{t}(i)\beta_{t}(i)}{p(O|\lambda)} = \frac{\alpha_{t}(i)\beta_{t}(i)}{\sum_{i}^{n}{\alpha_{t}(i)\beta_{t}(i)}}

since :math:`\alpha_t(i)` accounts for the partial observation sequence
:math:`O_1 O_2 ... O_t` and state :math:`S_1` at :math:`t`, while :math:`\beta_t(i)`
accounts for the remainder of the observation sequence :math:`O_{t+1}O_{t+2}...O_{T}`
given state :math:`S_i` at :math:`t`. The normalization factor :math:`p(O|\lambda)=\sum_{i=1}^{n}{\alpha_t(i)\beta_t(i)}`
makes :math:`\gamma_t(i)` a probability measure so that

.. math::

   \sum_{i=1}^{n}{\gamma_t(i)} = 1

Using :math:`\gamma_t(i)`, we can solve for the individually most likely state
:math:`q_t` at time :math:`t`, as

.. math::

   q_t = \operatorname*{arg\,max}_{1\le i\le n}[\gamma_t(i)]

Although the above maximizes the expected number of correct states, there could
be some problems with the resulting state sequence. For example, when the HMM
has state transitions which have zero probability (:math:`a_{ij}=0` for some
:math:`i` and :math:`j`), the "optimal" state sequence may, in fact, not event be
a valid state sequence. This is due to the fact that the solution of the above
equation simply determines the most likely state at every instant, without
regard to the probability of occurrence of sequences of states.

One possible solution to the above problem is to modify the optimality criterion.
For example, one could solve for the state sequence that maximizes the expected
number of correct pairs of states :math:`(q_t, q_{t+1})`, or triples of states
:math:`(q_t, q_{t+1}, q_{t+2})`, etc. Although these criteria might be reasonable
for some applications, the most widely used criterion is to find the single best
state sequence (path), i.e., to maximize :math:`p(Q|O,\lambda)` which is
equivalent to maximizing :math:`p(Q, O|\lambda)`. A formal technique for finding
this single best state sequence exists, based on dynamic programming methods,
and is called the Viterbi algorithm.

.. rubric:: Viterbi algorithm

To find the single best state sequence, :math:`Q=\{q_1,q_2,...,q_T\}`, for
the given observation sequence :math:`O=\{O_1,O_2,...,O_T\}`, we need to
define the quantity

.. math::

   \delta_{t}(i) = \max_{q_1,q_2,...,q_{t-1}}{p(q_{1}q_{2}...q_{t}=i, O_{1}O_{2}...O_{t}|\lambda)}

i.e., :math:`\delta_{t}(i)` is the best score (highest probability) along a
single path, at time :math:`t`, which accounts for the first :math:`t` observations
and ends in state :math:`S_i`. By induction we have

.. math::

   \delta_{t+1}(j) = [\max_{i}{\delta_{t}(i)a_{ij}}]\cdot b_{j}(O_{t+1})

To actually retrieve the state sequence, we need to keep track of the argument
which maximized the above equation, for each :math:`t` and :math:`j`. We do
this via the array :math:`\psi_{t}(j)`. The complete procedure for finding the
best state sequence can now be stated as follows

1. Initialization:

   .. math::

      \begin{array}{l l}
      \delta_{1}(i) = \pi_{i}b_{i}(O_{1}) & \quad 1 \le i \le n \\
      \psi_{1}(i) = 0 & \\
      \end{array}

2. Recursion:

   .. math::

      \begin{array}{l l}
      \delta_{t}(j) = \max_{1\le i\le n}[\delta_{t-1}(i)a_{ij}]b_{j}(O_{t}) & \quad 2 \le t \le T \\
                                                                            & \quad 1 \le j \le n \\
      \psi_{t}(j) = \max_{1\le i\le n}[\delta_{t-1}(i)a_{ij}]               & \quad 2 \le t \le T \\
                                                                            & \quad 1 \le j \le n \\
      \end{array}

3. Termination:

   .. math::

      p^{*} = \max_{1\le i\le n}[\delta_{T}(i)]

      q_{T}^{*} = \operatorname*{arg,\max}_{1\le i\le n}[\delta_{T}(i)]

4. Path (state sequence) backtracking:

   .. math::

      q_{t}^{*} = \psi_{t+1}(q_{t+1}^{*}) \quad t = T - 1, T - 2, ..., 1

Solution to Problem 3
**********************


