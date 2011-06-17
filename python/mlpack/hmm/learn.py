from mlpack.hmm.model import *
import math
import copy
import numpy
import itertools
import sys

ALPHA = 0
BETA  = 1

class Cluster(object):
    def __init__(self, e=None):
        self.elements = []
        self.centroid = None
        if e is not None:
            self.elements.append(e)
            self.centroid = e.factor()

    def add(self, e):
        if self.centroid is None:
            self.centroid = e.factor()
        else:
            self.centroid.reeval_add(e, self.elements)
        self.elements.append(e)

    def remove(self, i):
        self.centroid.reeval_remove(self.elements[i], self.elements)
        del self.elements[i]

class KMeansCalculator(object):
    def __init__(self, k, elements):
        self.clusters = []
        len_elem = len(elements)
        cluster_nb, elem_nb = 0, 0

        while elem_nb < len_elem and cluster_nb < k\
                and len_elem - elem_nb > k - cluster_nb:
            elem = elements[elem_nb]
            added = False
            for i in range(cluster_nb):
                cluster = self.clusters[i]
                if cluster.centroid.distance(elem) == 0.0:
                    cluster.add(elem)
                    added = True
                    break

            elem_nb += 1
            if added:
                continue

            self.clusters.append(Cluster(elements[elem_nb]))
            cluster_nb += 1

        while cluster_nb < k and elem_nb < len_elem:
            self.clusters.append(Cluster(elements[elem_nb]))
            elem_nb, cluster_nb = elem_nb+1, cluster_nb+1

        while cluster_nb < k:
            self.clusters.append(Cluster())
            cluster_nb += 1

        while elem_nb < len_elem:
            elem = elements[elem_nb]
            self.nearest_cluster(elem).add(elem)

        terminated = False
        while not terminated:
            terminated = True
            for i, c in enumerate(self.clusters):
                elist = c.elements
                for j, e in enumerate(elist):
                    if c.centroid.distance(e) > 0.0:
                        nearest = self.nearest_cluster(e)
                        if c != nearest:
                            nearest.add(e)
                            c.remove(j)
                            terminted = False

    def nearest_cluster(self, elem):
        distance = sys.maxint
        cluster = None
        for i, c in enumerate(self.clusters):
            d = c.centroid.distance(elem)
            if distance > d:
                distance, cluster = d, c
        return cluster

    def cluster(self, index):
        return self.clusters[index].elements

class ForwardBackwardCalculator(object):
    def __init__(self, oseq, hmm, flags):
        if ALPHA in flags:
            self.compute_alpha(hmm, oseq)
        if BETA in flags:
            self.compute_beta(hmm, oseq)

        self.compute_prob(oseq, hmm, flags)

    def compute_alpha(self, hmm, oseq):
        nb_states = hmm.nb_states()
        self.alpha = numpy.zeros((len(oseq), nb_states))
        for i in range(nb_states):
            self.alpha[0][i] = hmm.get_pi(i) * hmm.get_opdf(i).probability(oseq[0])
        for i, o in enumerate(oseq[1:]):
            for j in range(nb_states):
                self.compute_alpha_step(hmm, o, i, j)

    def compute_alpha_step(self, hmm, o, t, j):
        nb_states = hmm.nb_states()
        s = 0.0
        for i in range(nb_states):
            s += alpha[t-1][i] * hmm.get_aij(i, j)
        self.alpha[t][j] = s * hmm.get_opdf(j).probability(o)

    def compute_beta(self, hmm, oseq):
        nb_states = hmm.nb_states()
        len_seq = len(seq)
        self.beta = numpy.zeros((len_seq, nb_states))
        self.beta[len_seq-1] = numpy.array([1.0 for i in range(nb_states)])
        for t in range(len_seq-2, -1, -1):
            for i in range(nb_states):
                self.compute_beta_step(hmm, oseq[t+1], t, i)

    def compute_beta_step(self, hmm, o, t, i):
        nb_states = hmm.nb_states()
        s = 0.0
        for j in range(nb_states):
            s += self.beta[t+1][j] * hmm.get_aij(i, j) * hmm.get_opdf(j).probability(o)
        self.beta[t][i] = s

    def get_alpha(self, i, j):
        return self.alpha[i][j]

    def get_beta(self, i, j):
        return self.beta[i][j]

    def get_prob(self):
        return self.probability

    def compute_prob(self, hmm, oseq, flags):
        self.probability = 0.0
        nb_states = hmm.nb_states()
        if ALPHA in flags:
            probability += numpy.sum(self.alpha, axis=1)[-1]
        else:
            for i in range(nb_states):
                probability += hmm.get_pi(i) * hmm.get_opdf(
                        i).probability(oseq[0]) * self.beta[0][i]

class Clusters(object):
    def __init__(self, k, observations):
        self.cluster_hash = {}
        self.clusters = []

        kmc = KMeansCalculator(k, observations)

        for i in range(k):
            cluster = kmc.clusters[i]
            self.clusters.append(cluster)
            for element in cluster:
                cluster_hash[element] = i

    def put(self, o, cluster_nb):
        self.cluster_hash[o] = cluster_nb
        self.clusters[cluster_nb].add(o)

    def remove(self, o, cluster_nb):
        self.cluster_hash[cluster_nb] = -1
        self.clusters[cluster_nb].remove(o)

    def is_in_cluster(self, o, cluster_nb):
        return self.cluster_nb(o) == cluster_nb

    def cluster_nb(self, e):
        return self.cluster_hash[e]

    def cluster(self, i):
        return self.clusters[i]

class ViterbiCalculator(object):
    def __init__(self, oseq, hmm):
        nb_states = hmm.nb_states()
        len_seq = len(oseq)

        self.delta = numpy.zeros((len_seq, nb_states))
        self.psy = numpy.zeros((len_seq, nb_states))
        self.state_seq = numpy.zeros((len_seq,))

        for i in range(nb_states):
            delta[0][i] = -math.log(hmm.get_pi(i)) - math.log(hmm.get_opdf(i).probability(oseq[0]))
            psy[0][i] = 0

        for t, o in enumerate(oseq[1:]):
            for i in range(nb_states):
                self.compute_step(hmm, o, t+1, i)

        self.ln_probability = sys.maxint
        for i in range(nb_states):
            prob = self.delta[-1][i]
            if self.ln_probability > prob:
                self.ln_probability = prob
                self.state_seq[-1] = i
        self.ln_probability = -self.ln_probability

        for t in range(len_seq-2, -1, -1):
            self.state_seq[t] = self.psy[t+1][self.state_seq[t+1]]

    def compute_step(self, hmm, o, t, j):
        nb_states = hmm.nb_states()
        min_delta, min_psy = sys.maxint, 0
        for i in range(nb_states):
            delta = self.delta[t-1][i] - math.log(hmm.get_aij(i, j))
            if min_delta > delta:
                min_delta = delta
                min_psy = i

        self.delta[t][j] = min_delta - math.log(hmm.get_opdf(j).probability(o))
        self.psy[t][j] = min_psy

class KMeansLearner(object):
    def __init__(self, nb_states, opdf_factory, sequences):
        self.sequences = sequences
        self.opdf_factory = opdf_factory
        self.nb_states = nb_states

        observations = self.flat(sequences)
        self.clusters = Clusters(nb_states, observations)
        self.terminated = False

    def iterate(self):
        hmm = HmmModel(self.nb_states, self.opdf_factory)
        self.learn_pi(hmm)
        self.learn_aij(hmm)
        self.learn_opdf(hmm)

        self.terminated = self.optimize_cluster(hmm)

        return hmm

    def learn(self):
        hmm = None
        while not self.terminated:
            hmm = self.iterate()
        return hmm

    def learn_pi(self, hmm):
        pi = numpy.zeros((self.nb_states,))
        len_seq = len(self.sequences)
        for sequence in self.sequences:
            pi[self.clusters.cluster_nb(sequence[0])] += 1.0
        for i in range(self.nb_states):
            hmm.set_pi(i, pi[i] / len_seq)

    def learn_aij(self, hmm):
        nb_states = hmm.nb_states()
        hmm.a = numpy.zeros((nb_states, nb_states))
        for seq in self.sequences:
            if len(seq) < 2:
                continue

            second_state = self.clusters.cluster_nb(seq[0])
            for i, seq in enumerate(sequences):
                first_state = second_state
                second_state = self.clusters.cluster_nb(seq[i])

                hmm.set_aij(first_state, second_state,
                        hmm.get_aij(first_state, second_state)+1.0)

        sums = numpy.sum(hmm.a, axis=1)
        for i in range(nb_states):
            s = sums[i]
            if s == 0.0:
                for j in range(nb_states):
                    hmm.set_aij(i, j, 1.0 / nb_states)
            else:
                for j in range(nb_states):
                    hmm.set_aij(i, j, hmm.get_aij(i, j) / s)

    def learn_opdf(self, hmm):
        nb_states = hmm.nb_states()
        for i in range(nb_states):
            obsseq = self.clusters.cluster(i)
            if len(obsseq) == 0:
                hmm.set_opdf(i, self.opdf_factory.factor())
            else:
                hmm.get_opdf(i).fit(obsseq)

    def optimize_cluster(self, hmm):
        modif = False
        for seq in self.sequences:
            vc = ViterbiCalculator(seq, hmm)
            states = vc.state_seq
            for i, state in enumerate(states):
                o = seq[i]
                if self.clusters.cluster_nb(o) != state:
                    modif = True
                    self.clusters.remove(o, self.clusters.cluster_nb(o))
                    self.clusters.put(o, state)

        return not modif

    @classmethod
    def flat(self, sequences):
        return list(itertools.chain.from_iterable(sequences))

class BaumWelchLearner(object):
    def __init__(self, nb_iterations=9):
        self.nb_iterations = nb_iterations

    def iterate(self, hmm, sequences):
        nhmm = hmm.copy()

        len_seq = len(sequences)
        nb_states = hmm.nb_states()
        all_gamma = numpy.zeros((len_seq,1,1))
        aij_num   = numpy.zeros((len_seq, len_seq))
        aij_den   = numpy.zeros((len_seq,))

        for g, obs_seq in enumerate(sequences):
            fbc = ForwardBackwardCalculator(obs_seq, hmm, set([ALPHA, BETA]))
            xi = self.estimate_xi(obs_seq, fbc, hmm)
            gamma = all_gamma[g] = self.estimate_gamma(xi, fbc)

            for i in range(nb_states):
                for t in range(len_seq):
                    aij_den[i] += gamma[t][i]

                    for j in range(nb_states):
                        aij_num[i][j] += xi[t][i][j]

        for i in range(nb_states):
            if aij_den[i] == 0.0:
                for j in range(nb_states):
                    nhmm.set_aij(i, j, hmm.get_aij(i, j))
            else:
                for j in range(nb_states):
                    nhmm.set_aij(i, j, aij_num[i][j] / aij_den[i])

        for i in range(nb_states):
            nhmm.set_pi(i, 0.0)
        for o in range(len_seq):
            for i in range(nb_states):
                nhmm.set_pi(i, nhmm.get_pi(i) + all_gamma[o][0][i] / len_seq)

        for i in range(nb_states):
            observations = KMeansLearner.flat(sequences)
            len_obsseq = len(observations)
            weights = numpy.zeros((len_obsseq,))
            s, j = 0.0, 0
            for o, seq in enumerate(sequences):
                len_seq = len(seq)
                for t in range(len_seq):
                    weight[j] = all_gamma[o][t][i]
                    s += weight[j]
                    j += 1

            j -= 1
            while j >= 0:
                weights[j] /= s
                j -= 1

            opdf = nhmm.get_opdf(i)
            opdf.fit(observations, weights)

        return nhmm

    def learn(self, initial_hmm, sequences):
        hmm = initial_hmm
        for i in range(self.nb_iterations):
            hmm = self.iterate(hmm, sequences)
        return hmm

    def estimate_xi(self, sequence, fbc, hmm):
        len_seq = len(sequence)
        nb_states = hmm.nb_states()
        xi = numpy.zeros((len_seq, nb_states, nb_states))
        probability = fbc.get_prob()

        for t, o in enumerate(sequence[1:]):
            for i in range(nb_states):
                for j in range(nb_states):
                    xi[t][i][j] = fbc.get_alpha(t, i) * hmm.get_aij(i, j)\
                            * hmm.get_opdf(j).probability(o)\
                            * fbc.get_beta(t+1, j) / probability

        return xi

    def estimate_gamma(self, xi, fbc):
        len_seq, nb_states, _ = xi.shape
        gamma = numpy.zeros((len_seq+1, nb_states))
        for t in range(len_seq):
            for i in range(nb_states):
                for j in range(nb_states):
                    gamma[t][i] += xi[t][i][j]
        for j in range(nb_states):
            for i in range(nb_states):
                gamma[len_seq][j] += xi[-1][i][j]

        return gamma
