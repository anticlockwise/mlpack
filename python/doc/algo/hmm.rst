Hidden Markov Model (HMM)
===========================

Consider a system which may be described at any time as being in one of a set
of :math:`n` distinct states, :math:`\{S_1, S_2, ..., S_n\}`. At regularly spaced
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


