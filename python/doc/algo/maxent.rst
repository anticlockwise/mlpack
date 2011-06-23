Maximum Entropy Model (Maxent)
===============================

Let's consider a random process which produces an output value :math:`y`, a
member of a finite set :math:`\Phi`. For example, in machine translation, the
word *in* can be translated, hence the value of :math:`y` could take on any
word in the set :math:`\{dans, en, \grave{a}, au\hspace{2pt}cours\hspace{2pt}de, pendant\}`.
In generating :math:`y`, the process can be influenced by some contextual
information :math:`x`, a membef of finite set :math:`\chi`. In the machine
translation example, this information could include the words surrounding
*in*.

The task here is to construct a stochastic model that accurately represents
the behaviour of the random process. Such a model is a method of estimating
the conditional probability that, given a context :math:`x`, the process will
output :math:`y`

Training data
--------------

To study the process, we observe the behaviour of the random process for
some time, collecting a large number of samples
:math:`(\textbf{x}_1, y_1), (\textbf{x}_2, y_2), ... (\textbf{x}_N, y_N)`.
We can summarise the training sample in terms of empirical probability
distribution :math:`\tilde{p}(x, y)`, defined by

.. math::

   \tilde{p}(x, y) = \frac{1}{N} \times \text{number of times that (\textit{x, y}) occurs in the sample}

Features
---------

Our goal is to construct a statistical model of the process which generated
the training sample :math:`\tilde{p}(x, y)`. The building blocks of this model
will be a set of statistics of the training sample. Particular statistics of
the model depends on the conditioning information :math:`x`. For instance, in
the translation example, we might notice that, in the training sample, if
*April* is the word following *in*, then the translation of *in* is *en* with
a higher probability.

To express the event that *in* translates as *en* when *April* is the following
word, we can introduce the indicator function

.. math::

   f(x, y) = \left\{
   \begin{array}{l l}
     1 & \quad \text{if \textit{y = en} and \textit{April} follows \textit{in}} \\
     0 & \quad \text{otherwise} \\
   \end{array} \right.

The expected value of :math:`f` with respect to the empirical distribution :math:`\tilde{p}(x, y)`
is exactly the statistic we are interested in. We denote this expected value by

.. math::

   \tilde{p}(f) = \sum_{x, y}{\tilde{p}(x, y)f(x, y)}

We can express any statistic of the sample as the expected value of an
appropriate binary-valued indicator function :math:`f`. We call such function a *feature
function* or *feature* for short.

When we discover a statistic that we feel is useful, we can acknowledge its
importance by requiring that our model accord with it. We do this by constraining
the expected value that the model assigns to the corresponding feature function
:math:`f`. The expected value of :math:`f` with respect to the model
:math:`p(y|x)` is

.. math::

   p(f) = \sum_{x, y}{\tilde{p}(x)p(y|x)f(x, y)}

Where :math:`\tilde{p}(x)` is the empirical distribution of :math:`x` in the
training sample. We constrain this expected value to be the same as the expected
value of :math:`f` in the training sample. Therefore, we require

.. math::

   p(f) = \tilde{p}(f)

Combining the above three equations, we yield a more explicit equation

.. math::

   \sum_{x, y}{\tilde{p}(x)p(y|x)f(x, y)} = \sum_{x, y}{\tilde{p}(x, y)f(x, y)}

We call the above requirement a *constraint equation* or simple a *constraint*.
By restricting attention to those models :math:`p(y|x)` for which the constraint
holds, we are eliminating from consideration those models which do not agree
with the training sample on how often the output of the process should exhibit
the feature :math:`f`.

The maxent principle
---------------------

Suppose that we are given :math:`n` features :math:`f_i`, which determine
statistics we feel are important in modelling the process. We would like our
model to accord with the statistics. That is, we would like :math:`p` to lie
in the subset :math:`C` of :math:`P` defined by

.. math::

   C = \{ p \in P | p(f_i) = \tilde{p}(f_i)\quad \text{for i} \in \{1, 2, ..., n\} \}

Here :math:`P` is the space of all (unconditional) probability distributions on
:math:`3` points, sometimes called a *simplex*. If we impose no constraints,
then all probability models are allowable. Imposing one linear constraint :math:`C_1`
restricts us to those :math:`p\in P` which lie on the region defined by :math:`C_1`.
A second linear constraint could determine :math:`p` exactly, if the two
constraints are satisfiable, where the intersection of :math:`C_1` and :math:`C_2`
is non-empty. Alternatively, a second linear constraint could be inconsistent
with the first. However, the linear constraints in the present setting are
extracted from the training sample and cannot, by construction, be inconsistent.
Furthermore, the linear constraints in our applications will not event come close
to determining :math:`p\in P` uniquely. Instead, the set :math:`C = C_1 \cap C_2 \cap ... \cap C_n`
of allowable models will be infinite.

Among the models :math:`p \in C`, the maximum entropy philosophy dictates that
we select the distribution which is most uniform. But now we face a question left
open earlier: what does "uniform" mean?

A mathematical measure of the uniformity of a conditional distribution :math:`p(y|x)`
is provided by the conditional entropy

.. math::

   H(p) = -\sum_{x, y}{\tilde{p}(x)p(y|x)\log{p(y|x)}}

The entropy is bounded from below by zero, the entropy of a model with no
uncertainty at all, and from above by :math:`\log{|Y|}`, the entropy of the
uniform distribution over all possible :math:`|Y|` values of :math:`y`.
With this definition in hand - To select a model from a set :math:`C` of allowed
probability distributions, choose the model :math:`p^{*} \in C` with maximum
entropy :math:`H(p)`:

.. math::

   p^{*} = \operatorname*{arg\,max}_{p\in C} H(p)

It can be shown that :math:`p^{*}` is always well-defined, that is, there is always
a unique model :math:`p^{*}` with maximum entropy in any constrained set :math:`C`.

Exponential form
------------------

To solve the constraint and address the general maximum entropy problem, we apply
the method of Lagrange multipliers from the theory of constrained optimization.

The constrained optimization problem at hand is

.. math::

   p^{*} & = \operatorname*{arg\,max}_{p\in C} H(p)\\
         & = \operatorname*{arg\,max}_{p\in C} \left(-\sum_{x, y}({\tilde{p}(x)p(y|x)\log{p(y|x)}}\right)\\

We refer to this as the primal problem; it is a succint way of saying that we
seek to maximize :math:`H(p)` subject to the following constraints:

1. :math:`p(y|x) \ge 0` for all :math:`x, y`.
2. :math:`\sum_{y}{p(y|x)} = 1` for all :math:`x`. This and the previous condition gaurantee that :math:`p` is a conditional probability distribution.
3. :math:`\sum_{x, y}{\tilde{p}(x)p(y|x)f(x, y)} = \sum_{x, y}{\tilde{p}(x, y)f(x, y)} \text{for i}\in \{i, 2, ... n\}`. In other words, :math:`p\in C`, and so satisfies the active constraints :math:`C`.

To solve this optimization problem, we introduce the Lagrangian

.. math::

   \xi(p, \Lambda, \gamma) & \equiv -\sum_{x, y}{\tilde{p}(x)p(y|x)\log{p(y|x)}}\\
                           & +      \sum_{i}{\lambda_{i}\left(\sum_{x, y}{\tilde{p}(x,y)f_{i}(x,y) - \tilde{p}(x)p(y|x)f_{i}(x,y)}\right)}\\
                           & +      \gamma\sum_{y}{p(y|x)} - 1\\

The real-valued parameters :math:`\gamma` and :math:`\Lambda = \{\lambda_1, \lambda_2, ... \lambda_n\}`
correspond to the :math:`1+n` constraints imposed on the solution.

The following strategy yields the optimal value of :math:`p (p^{*})`: first hold
:math:`\gamma` and :math:`\Lambda` constant and maximize the Lagrangian
with respect to :math:`p`. This yields an expression for :math:`p` in terms of
the (still unsolved-for) parameters :math:`\gamma` and :math:`\Lambda`. Now
substitute this expression back into the equation, this time solving for the
optimal values of :math:`\gamma` and :math:`\Lambda` (:math:`\Lambda^*` and
:math:`\gamma^*`, respectively).

Proceeding this manner, we hold :math:`\Lambda`, :math:`\gamma` fixed and compute
the unconstrained maximum of the :math:`\xi(p, \Lambda, \gamma)` over all
:math:`p \in P`:

.. math::

   \frac{\partial\xi}{\partial p(y|x)} = -\tilde{p}(x)(1 + \log{p(y|x)}) - \sum_{i}{\lambda_{i}\tilde{p}(x)f_{i}(x,y)} + \gamma

Equating this expression to zero and solving for :math:`p(y|x)`, we find that at
its optimum, :math:`p` has the parametric form

.. math::

   p^{*}(y|x) = \exp{\left(\sum_{k=1}^{n}{\lambda_{i}f_{i}(x,y)}\right)}\exp{\left(-\frac{\gamma}{\tilde{p}(x)} - 1\right)}

We have thus found the parametric form of :math:`p^*`, and so we now take up
the task of solving for the optimal values :math:`\gamma^*, \Lambda^*`.
Recognizing that the second factor in this equation is the factor corresponding
to the second of the constraints listed above, we can rewrite the above as

.. math::

   p^{*}(y|x) = \frac{1}{Z(x)}\exp{\left(\sum_{i}{\lambda_{i}f_{i}(x,y)}\right)}

where :math:`Z(x)`, the normalizing factor, is given by

.. math::

   Z(x) = \sum_{y}{\exp{\left(\sum_{i}{\lambda_{i}f_{i}(x,y)}\right)}}

We have found :math:`\gamma^*` but not yet :math:`\Lambda^*`. Towards this end
we introduce some further notation. Define the dual function :math:`\Psi(\Lambda)`
as

.. math::

   \Psi(\Lambda) = \xi(p^*, \Lambda, \gamma^*)

and the dual optimization problem as

.. math::

   \text{Find}\,\Lambda^* = \operatorname*{arg\,max}_{\Lambda}\Psi(\Lambda)

Since :math:`p^*` and :math:`\gamma^*` are fixed, the righthand side of the
above has only the free variables :math:`\Lambda = \{\lambda_1, \lambda_2, ... \lambda_n\}`.

It is far from obvious that the :math:`p^*` with :math:`\Lambda = \Lambda^*`
given by above is in fact the solution to the constrained optimization problem
we set out to find. But in fact this is due to a fundamental principle in the
theory of Lagrange multipliers, called generically the Kuhn-Tucker theorem,
which asserts that (under suitable assumptions, which are satisfied here) the
primal and dual problems are closely related.

Maximum likelihood
-------------------

The log-likelihood :math:`L_{\tilde{p}}(p)` of the empirical distribution
:math:`\tilde{p}` as predicted by a model :math:`p` is defined by

.. math::

   L_{\tilde{p}}(p) = \log{\prod_{x,y}{p(y|x)^{\tilde{p}(x,y)}}} = \sum_{x, y}{\tilde{p}(x,y)\log{p(y|x)}}

It is easy to check that the dual function :math:`\Psi(\Lambda)` is, in fact,
just the log-likelihood for the exponential model :math:`p`; that is

.. math::

   \Psi(\Lambda) = L_{\tilde{p}}(p)

where :math:`p` has the parametric form of :math:`p^*(y|x)`. With this
interpretation, the result of the previous section can be rephrased as:

   The model :math:`p^* \in C` with maximum entropy is the model in the parametric family :math:`p(y|x)` that maximized the likelihood of the training sample :math:`\tilde{p}`.

This result provides an added justification for the maximum entropy principle:
if the notion of selecting a model :math:`p^*` on the basis of maximum entropy
isn't compelling enough, it so happens that this same :math:`p^*` is also the
model which, from among all models of the same parametric form :math:`p^{*}(y|x) = ^1/_{Z(x)}\times\exp{\left(\sum_{i}{\lambda_{i}f_{i}(x,y)}\right)}`,
can best account for the training sample.

Computing the parameters
------------------------

For all but the most simple problems, the :math:`\Lambda^*` that maximize
:math:`\Psi(\Lambda)` cannot be found analytically. Instead, we must resort to
numerical methods. From the perspective of numerical optimization, the function
:math:`\Psi(\Lambda)` is well behaved, since it is smooth and convex-:math:`\cap`
in each :math:`\lambda`. Consequently, a variety of numerical methods can be
used to calculate :math:`\Lambda^*`. One simple method is coordinate-wise
ascent, in which :math:`\Lambda^*` is computed by iteratively maximizing
:math:`\Psi(\Lambda)` one coordinate at a time. When applied to the maximum
entropy problem, this technique yields the popular Brown algorithm. Other
general purpose methods that can be used to maximize :math:`\Psi(\Lambda)`
include gradient ascent and conjugate gradient.

An optimization method specifically tailored to the maximum entropy problem
is the iterative scaling algorithm of Darroch and Ratcliff. The algorithm
is applicable whenever the feature functions :math:`f_{i}(x, y)` are non-negative:

.. math::

   f_{i}(x,y) \ge 0 \quad \text{for all}\,i, x, \text{and}\,y.

.. rubric:: Algorithm 1 Improved Iterative Scaling

.. parsed-literal::

   Input:  Feature functions :math:`f_1, f_2, ... f_n`; emperical distribution :math:`\tilde{p}(x,y)`
   Output: Optimal parameter values :math:`\Lambda_{i}^{*}`; optimal model :math:`p^*`

   1. Start with :math:`\lambda_{i} = 0 \text{for all}\,i=\{1, 2, ..., n\}`
   2. Do for each :math:`i \in \{1, 2, ..., n\}`:
      a. Let :math:`\Delta\lambda_{i}` be the solution to :math:`\sum_{x,y}{\tilde{p}(x)p(y|x)f_{i}(x,y)\exp{\left(\Delta\lambda_{i}f^{\#}(x,y)\right)}} = \tilde{p}(f_{i})` where :math:`f^{\#}(x,y)\equiv\sum_{i=1}^{n}f_{i}(x,y)(*)`
      b. Update the value of :math:`\lambda_i` according to: :math:`\lambda_i \leftarrow \lambda_i + \Delta\lambda_i`
   3. Go to step 2 if not all the :math:`\lambda_i` have converged

The key step in the algorithm is step (2a), the computation of the increments :math:`\Delta\lambda_i` that solve :math:`*`.
Notice that this equation contains the term :math:`p(y|x)`, which changes
as :math:`\lambda_i` becomes updated in step (2b). For this reason, the algorithm
is iterative, and requires a pass through the entire set of :math:`(x,y)` pairs
in the empirical sample :math:`\tilde{p}(x,y)` for each iteration.

Equation :math:`*` merits a brief comment. In words, :math:`f^{\#}(x,y)` is
the number of features :math:`f_{i}(x,y)` which are "active" (:math:`f_{i}(x,y)=1`)
for :math:`x,y`. In some cases, :math:`f^{\#}(x,y)` may be a constant (
:math:`f^{\#}(x,y) = M` for all :math:`x,y`, say), in which case :math:`\Delta\lambda_i`
is given explicitly as

.. math::

   \Delta\lambda_i = \frac{1}{M}\log{\frac{\tilde{p}(f_i)}{p_{\lambda}(f_i)}}

However, in practive it is typically the case that different numbers of features
will apply at different :math:`x,y`. If :math:`f^{\#}(x,y)` is not constant,
then the :math:`\Delta\lambda_i` must be computed numerically. A simple and
effective way of doing this is by Newton's method. This method computes the
solution :math:`\alpha_{*}` of an equation :math:`g(\alpha_{i})=0` iteratively
by the recurrence

.. math::

   \alpha_{n+1} = \alpha_{n} - \frac{g(\alpha_{n})}{g^{'}(\alpha_n)}

with an appropriate choice for :math:`\alpha_{0}` and suitable attention paid
to the domain of :math:`g`.
