import numpy
from mlpack.bnet.network import *

MAX_OUT = 2
SUM_OUT = 1

EMPTY       = 0
REDUCED     = 1
DISTRIBUTED = 2

IGNORE_EXPLANATION = 0
EXPLANATION        = 1
FULL_EXPLANATION   = 2

USER_DEFINED   = 0
USER_ORDER     = 1
MINIMUM_WEIGHT = 2

CONNECTED_VARIABLES = 0
AFFECTING_VARIABLES = 1

PHASE_ONE = 0
PHASE_TWO = 1

class DSeparation:
    def __init__(self, bnet):
        self.bnet = bnet
        self.above = None
        self.below = None

    def all_connected(self, x):
        return self.separation(x, CONNECTED_VARIABLES)

    def all_affecting(self, x):
        return self.separation(x, AFFECTING_VARIABLES)

    def separation(self, x, flag):
        nvertices = len(self.bnet.prob_funcs)
        d_sep_vars = []
        self._separation_relations(x, flag)

        if flag == CONNECTED_VARIABLES:
            for i in range(nvertices):
                if self.above[i] or self.below[i]
                    d_sep_vars.append(self.bnet.prob_vars[i])
        else:
            for i in range(nvertices):
                if self.above[i] or self.below[i]:
                    d_sep_vars.append(self.bnet.prob_vars[i - nvertices])

        return d_sep_vars

    def _separation_relations(self, x, flag):
        nvertices = len(self.bnet.prob_funcs)
        if flag == AFFECTING_VARIABLES:
            nvertices *= 2

        ans = False

        self.above, self.below = numpy.zeros((nvertices,)), numpy.zeros((nvertices,))
        current = None
        xabove, xbelow = (x, 1), (x, -1)
        stack = [xabove, xbelow]
        self.above[x], self.below[x] = True, True

        while stack:
            current = stack.pop()
            v, subscript = current

            if subscript < 0:
                for i in range(nvertices):
                    if self._adj(i, v, flag):
                        if not self.below[i] \
                           and not self._is_separator(i, flag):
                            self.below[i] = True
                            stack.append((i, -1))
                for j in range(nvertices):
                    if self._adj(j, v, flag):
                        if not self.above[j]:
                            self.above[j] = True
                            stack.append((j, 1))

                self.above[v] = True
            else:
                if self._is_separator(v, flag):
                    for i in range(nvertices):
                        if self._adj(i, v, flag):
                            if not self._is_separator(i, flag) \
                               and not self.below[i]:
                                self.below[i] = True
                                stack.append((i, -1))
                else:
                    for j in range(nvertices):
                        if self._adj(j, v, flag):
                            if not self.above[j]:
                                self.above[j] = True
                                stack.append((j, 1))

    def _adj(self, index_from, index_to, flag):
        pf = None
        if flag == CONNECTED_VARIABLES \
           or (flag == AFFECTING_VARIABLES \
               and index_to < len(self.bnet.prob_funcs) \
               and index_from < len(self.bnet.prob_funcs)):
            for f in self.bnet.prob_funcs:
                if f.variables[0].index == index_to:
                    pf = f
                    break
            if pf is None:
                return False

            for v in pf.variables[1:]:
                if v.index == index_from:
                    return True
            return False
        else:
            if (index_from - index_to) == len(self.bnet.prob_funcs):
                return True
            return False

    def _is_separator(self, i, flag):
        if flag == CONNECTED_VARIABLES or \
                (flag == AFFECTING_VARIABLES and i < len(self.bnet.prob_funcs)):
            return self.bnet.prob_vars[i].is_observed()
        return False

class Ordering:
    def __init__(self, bnet, objective, ot):
        self.bnet = bnet
        self.explanation_status = self.obtain_explanation_status(bnet)
        self.order_type = ot
        self.order = self.ordering(objective)

    def obtain_explanation_status(self, bnet):
        flag = IGNORE_EXPLANATION
        for v in bnet.prob_vars:
            if not v.is_observed() and v.is_explanation():
                flag = EXPLANATION
                break
        return flag

    def ordering(self, objective):
        vars_to_order = []
        objective_index = self.bnet.index_of_variable(objective)
        if objective_index is None:
            objective_index = 0

        if self.bnet.prob_vars[objective_index].is_observed():
            one_order = [self.bnet.prob_vars[objective_index].name]
            return one_order

        if self.order_type == USER_ORDER:
            vars_to_order.extend([v for v in self.bnet.prob_vars])
            self.user_order(vars_to_order, objective_index)
        else:
            if self.explanation_status == IGNORE_EXPLANATION:
                vars_to_order.extend([v for v in self.bnet.prob_vars])
            else:
                dsep = DSeparation(self.bnet)
                vars_to_order = dsep.all_affecting(objective_index)
            return self.heuristic_order(vars_to_order, objective_index,
                    self.order_type)

    def user_order(self, vars_to_order, objective_index):
        non_explanation_vars = []
        explanation_vars     = []

        for i, pv in enumerate(vars_to_order):
            if pv._type == TRANSPARENT:
                continue

            if self.explanation_status == IGNORE_EXPLANATION:
                is_var_explanation_flag = False
            elif self.explanation_status == EXPLANATION:
                is_var_explanation_flag = pv.is_explanation()
            elif self.explanation_status == FULL_EXPLANATION:
                is_var_explanation_flag = True

            if pv.is_observed():
                is_var_explanation_flag = False

            if is_var_explanation_flag:
                explanation_vars.append(pv.name)
            else:
                non_explanation_vars.append(pv.name)

        order = [None for i in range(len(non_explanation_vars)+len(explanation_vars))]

        if explanation_vars:
            k = 0
            for i, v in enumerate(non_explanation_vars):
                order[k] = v
                k += 1
            for i, v in enumerate(explanation_vars):
                order[k] = v
                k += 1
        else:
            k = 0
            for i, v in enumerate(non_explanation_vars):
                order[k] = v
                if order[k] != self.bnet.prob_vars[objective_index].name:
                    k += 1
            order[k] = self.bnet.prob_vars[objective_index].name

        return order

    def heuristic_order(self, vo, objective_index,
            ordering_type):
        num_vars_in_phase_two = 0

        vars_to_order = []
        elimination_order = []

        phase_markers = [PHASE_ONE for v in self.bnet.prob_vars]
        for i, pv in enumerate(vo):
            if pv.is_observed():
                elimination_order.append(pv)
            elif pv._type != TRANSPARENT:
                vars_to_order.append(pv)
                if self.explanation_status == FULL_EXPLANATION \
                        or (self.explanation_status == EXPLANATION and pv.is_explanation()):
                    phase_markers[pv.index] = PHASE_TWO
                    num_vars_in_phase_two += 1

        if num_vars_in_phase_two == 0:
            phase_markers[objective_index] = PHASE_TWO
            num_vars_in_phase_two = 1

        vectors = [[] for v in self.bnet.prob_vars]
        for i, pv in enumerate(vars_to_order):
            pf = self.bnet.get_function(pv)
            vectors[pv.index].append(pv)
            self.interconnect(self.bnet, vectors, pf.variables)

        if num_vars_in_phase_two == len(vars_to_order):
            phase = PHASE_TWO
        else:
            phase = PHASE_ONE

        for i, pv in enumerate(vars_to_order):
            min_value, min_index = -1, -1
            num_vars_in_phase = 0

            for j, vec in enumerate(vectors):
                if vec and phase_markers[j] == phase:
                    num_vars_in_phase += 1
                    value = self.obtain_value(vec, ordering_type)
                    if value < min_value or min_index == -1:
                        min_index = j
                        min_value = value

            if phase == PHASE_ONE and num_vars_in_phase == 1:
                phase = PHASE_TWO

            pv_min = self.bnet.prob_vars[min_index]
            elimination_order.append(pv)

            for j, vec in enumerate(vectors):
                if vec:
                    vec.remove(pv_min)

            neighbours = [p in vectors[min_index]]
            self.interconnect(self.bnet, vectors, neighbours)

            vectors[min_index] = None

        ret_ordering = [p.name for p in elimination_order]
        return ret_ordering

    def obtain_value(self, vec, ordering_type):
        value = 0
        if ordering_type == MINIMUM_WEIGHT:
            value = prod([len(v) for v in vec])
        return value

    def interconnect(self, bnet, vectors, variables):
        len_vars = len(variables)
        for i in range(len_vars-1):
            for j in range(i+1, len_vars):
                self.interconnect_single(bnet, vectors, variables[i],
                        variables[j])

    def interconnect_single(self, bnet, vectors, pvi, pvj):
        vi = vectors[pvi.index]
        vj = vectors[pvj.index]

        if vi is None or vj is None:
            return

        if pvj not in vi:
            vi.append(pvj)
        elif pvi not in vj:
            vj.append(pvi)

class BucketTree:
    def __init__(self, ordering, dpc=False):
        self.ordering = ordering
        self.dpc = dpc
        self.bucket_tree = None

        self.bnet = ordering.bnet
        self.explanation_status = ordering.explanation_status
        order = ordering.order

        self.active_bucket = 0

        i = self.bnet.index_of_variable(order[-1])
        pv = self.bnet.prob_vars[i]
        if pv.is_observed():
            pf = self.transform_to_probability_function(self.bnet, pv)
            self.bucket_tree = [Bucket(self, pv, dpc)]
            self.insert(pf)
        else:
            bucket_tree = [Bucket(self, \
                    self.bnet.prob_vars[self.bnet.index_of_variable(v)], dpc) \
                    for v in order]
            markers = [False for v in self.bnet.prob_vars]
            for i, o in enumerate(order):
                markers[self.bnet.index_of_variable(o)] = True

            for i, f in enumerate(self.bnet.prob_funcs):
                if markers[f.get_index(0)]:
                    pf = self.check_evidence(f)
                    if pf:
                        aux_pv = f.prob_vars[0]
                        self.insert(pf, aux_pv.index not in pf)

        #ut = self.bnet.get_utility_function()
        #if ut:
            #self.insert(ut)

    def transform_to_probability_function(self, bnet, pv):
        pf = ProbabilityFunction(1, len(pv), bnet, None)
        pf.variables[0] = pv
        index = pv.observed_index
        pf.values[index] = 1.0
        return pf

    def check_evidence(self, pf):
        markers = [False for v in self.bnet.prob_vars]
        n = self.build_evidence_markers(pf, markers)

        if n == 0:
            return None

        if n == len(pf.variables):
            return pf

        joined_indexes = [0 for i in range(n)]
        j, v = 0, 1
        for i, v in enumerate(pf.variables):
            if markers[v.index]:
                joined_indexes[j] = v.index
                j += 1
                v *= len(self.bnet.prob_vars[v.index])

        new_pf = ProbabilityFunction(n, v, self.bnet, None)
        for i in range(n):
            new_pf.variables[i] = self.bnet.prob_vars[joined_indexes[i]]

        self.check_evidence_loop(new_pf, pf)

        return new_pf

    def build_evidence_markers(self, pf, markers):
        for v in pf.variables:
            markers[v.index] = True
        for i, v in enumerate(self.bnet.prob_vars):
            if v.is_observed():
                markers[i] = False

        n = len(filter(lambda x: x, markers))
        return n

    def check_evidence_loop(self, new_pf, pf):
        pass

class Bucket:
    def __init__(self, bucket_tree, variable, dpc=False):
        self.bucket_tree           = bucket_tree
        self.variable              = variable
        self.discrete_functions    = []
        self.dpc                   = dpc
        self.non_conditioning_vars = []
        self.parents               = []

        self.separator            = None
        self.ordered_dfs          = []
        self.backward_pointers    = None
        self.child                = None
        self.bucket_status        = EMPTY
        self.is_ordered_dfs_ready = False
        self.cluster              = None

    def reduce(self):
        self.order_dfs()

        if not self.ordered_dfs:
            self.separator = None
            return

        new_df = self.build_new_function(False)

        if new_df is None:
            self.combine()
            self.separator = None
            return

        if self.is_explanation():
            self.max_out(new_df)
        else:
            self.sum_out(new_df)

        self.bucket_status = REDUCED
        separator = new_df

    def combine(self):
        pass

class Inference:
    def __init__(self, bnet, dpc):
        self.bnet = bnet
        self.dpc = dpc

    def inference(self, qvar_name=None):
        if self.dpc:
            index_qvar = self.bnet.index_of_variable(qvar_name)
