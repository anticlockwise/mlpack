import numpy
from itertools import prod

class DiscreteVariable:
    def __init__(self, name=None, index=-1, values=None):
        self.name = name
        self.index = index
        self.values = values

    def index_of_value(self, val):
        try:
            return self.values.index(val)
        except:
            return None

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

class DiscreteFunction:
    def __init__(self, num_var=0, num_val=0):
        self.variables = None
        self.values = None

        if num_var:
            self.variables = [None for i in range(num_var)]
        if num_val:
            self.values    = numpy.zeros((num_val,))

    def copy(self):
        func = DiscreteFunction()
        func.variables = self.variables
        func.values = values
        return func

    def __contains__(self, index):
        return (len(filter(lambda v: v.index == index, self.variables)) > 0)

    def evaluate(self, pvs, val_indexes):
        pos = self.get_pos_from_indexes(pvs, val_indexes)
        return self.values[pos]

    def get_pos_from_indexes(self, pvs, val_indexes):
        pos, jump = 0, 1
        for i, v in enumerate(self.variables):
            k = v.index
            pos += val_indexes[k] * jump
            jump *= len(pvs[k])
        return pos

    def sum_out(self, pvs, markers):
        indexes = numpy.zeros((len(pvs),))
        val_lengths = numpy.array([len(v) for v in pvs])

        vars_to_sum_out = numpy.array([v for v in self.variables if markers[v.index]])
        vars_to_stay = numpy.array([v for v in self.variables if not markers[v.index]])

        num_sum_out_var, num_stay_var = len(vars_to_sum_out), len(vars_to_stay)
        num_sum_out_val, num_new_df_val = prod([len(v) for v in vars_to_sum_out]), prod([len(v) for v in vars_to_stay])

        if num_stay_var == 0:
            return None

        if num_sum_out_var == 0:
            return self.copy()

        indexes_for_var_sum_out = numpy.array([v.index for v in vars_to_sum_out])

        new_df = DiscreteFunction(num_stay_var, num_new_df_val)

        for i, v for enumerate(vars_to_stay):
            new_df.variables[i] = v

        last_new_df = num_stay_var - 1
        last_index_for_sum_out_vars = num_sum_out_var - 1

        for i in range(num_new_df_val):
            v = 0.0
            indexes[indexes_for_var_sum_out] = 0
            for j in range(num_sum_out_val):
                v += self.evaluate(pvs, indexes)
                indexes[indexes_for_var_sum_out[-1]] += 1
                for k in range(last_index_for_sum_out_vars, 0, -1):
                    cur = indexes_for_var_sum_out[k]
                    if indexes[cur] >= val_lengths[cur]:
                        indexes[cur] = 0
                        indexes[indexes_for_var_sum_out[k-1]] += 1
                    else:
                        break

            new_df.values[i] = v

            indexes[new_df.index(last_new_df)] += 1
            for j in range(last_new_df, 0):
                cur = new_df.index(j)
                if indexes[cur] >= val_lengths[cur]:
                    indexes[cur] = 0
                    indexes[new_df.index(j - 1)] += 1
                else:
                    break

        return new_df

    def multiply(self, dvs, mult):
        len_dvs = len(dvs)
        var_markers = numpy.zeros((len_dvs,))
        indexes = numpy.zeros((len_dvs,))
        val_lengths = numpy.array([len(v) for v in dvs])

        n = 0
        for i, v in enumerate(self.variables):
            k = self.index(i)
            if not var_markers[k]:
                var_markers[k] = True
                n += 1

        joined_indexes = numpy.array([i for i in range(len_dvs) if var_markers[i]])
        v = prod([len(v) for i, v in enumerate(dvs) if var_markers[i]])

        new_df = DiscreteFunction(n, v)
        for i in range(n):
            new_df.variables[i] = dvs[joined_indexes[i]]

        last_new_df = n - 1

        for i in range(v):
            t = self.evaluate(dvs, indexes) * mult.evaluate(dvs, indexes)
            new_df.values[i] = t

            indexes[new_df.index(last_new_df)] += 1

            for j in range(last_new_df, 0, -1):
                cur = new_df.index(j)
                if indexes[cur] >= val_lengths[cur]:
                    indexes[cur] = 0
                    indexes[new_df.index(j - 1)] += 1
                else:
                    break

        return new_df

    def normalize(self):
        total = numpy.sum(self.values)
        if total > 0.0:
            self.values /= total

    def normalize_first(self):
        pass

class ProbabilityVariable(DiscreteVariable):
    def __init__(self, bnet=None, name=None, index=-1, values=None,
            props={}):
        DiscreteVariable.__init__(self, name, index, values)
        self.bnet = bnet
        self.props = props

class ProbabilityFunction(DiscreteFunction):
    def __init__(self, num_var, num_val, bnet=None, props={}):
        DiscreteFunction.__init__(self, num_var, num_val)
        self.bnet = bnet
        self.props = props

class BayesNet:
    def __init__(self, name=None, num_var=0, num_func=0, props={}):
        self.name = name
        self.prob_vars = [None for i in range(num_var)]
        self.prob_funcs = [None for i in range(num_func)]
        self.props = props
