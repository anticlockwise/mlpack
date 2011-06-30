import numpy
from mlpack.bnet.network import *

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
                        if not self.below[i] and not self._is_separator(i, flag):
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
                            if not self._is_separator(i, flag) and not self.below[i]:
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
        if flag == CONNECTED_VARIABLES or \
                (flag == AFFECTING_VARIABLES and index_to < len(self.bnet.prob_funcs) and index_from < len(self.bnet.prob_funcs)):
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
                (flag == AFFECTING_VARIABLES and i < len(self.bnet.prob_funcs):
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

class BucketTree:
    def __init__(self, ordering, dpc=False):
        self.ordering = ordering
        self.dpc = dpc

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
