#! /usr/bin/env python
import numpy as np
from scipy import optimize
from collections import Sequence
from itertools import chain, count
import numbers
import argparse

def NumberQ(x):
    return isinstance(x, numbers.Number)

def depth(seq):
    for level in count():
        if not seq:
            return level
        seq = list(chain.from_iterable(s for s in seq if isinstance(s, Sequence)))

class Info_Collapse:
    def __init__(self,
                 a,
                 b,
                 c,
                 d,
                 e,
                 f,
                 Pi_T,
                 Pi_C,
                 Pi_A,
                 Pi_G,
                 ratevector_file):

        self.frequency = [Pi_T, Pi_C, Pi_A, Pi_G]
        self.len_freq = len(self.frequency)
        mu = 1 / 2 / (a * Pi_T * Pi_C + b * Pi_T * Pi_A + c * Pi_T * Pi_G + d * Pi_C * Pi_A + e * Pi_C * Pi_G + f * Pi_A * Pi_G)

        with open(ratevector_file) as infile:
            ratevector = [float(x) for x in infile.read().split(",")]
        clean_rate = ratevector[ratevector == 0]

        Q = mu *  np.array([[-a * Pi_C - b * Pi_A - c * Pi_G, a * Pi_C, b * Pi_A, c * Pi_G],
                            [a * Pi_T, -a * Pi_T - d * Pi_A - e * Pi_G, d * Pi_A, e * Pi_G],
                            [b * Pi_T, d * Pi_C, -b * Pi_T - d * Pi_C - f * Pi_G, f * Pi_G],
                            [c * Pi_T, e * Pi_C, f * Pi_A, -c * Pi_T - e * Pi_C - f * Pi_A]])

        self.evals, py_v = np.linalg.eig(Q) # for some reason py_v seems to already be transposed
        self.tev = py_v
        self.itev = np.linalg.inv(self.tev)

    def P(self, lamda, T):
        output = np.dot(np.dot(self.tev, np.diag(np.exp(self.evals * lamda * T))), self.itev)
        return output

    ## RHS
    def _JointProb3(self, root, char1, char2, T1, T2, lamda):
        return self.frequency[root] * self.P(lamda, T1)[root, char1] * self.P(lamda, T2)[root, char2]

    def _CondProb3(self, root, char1, char2, T1, T2, lamda):
        output = self._JointProb3(root, char1, char2, T1, T2, lamda)
        output /= np.sum([self._JointProb3(i, char1, char2, T1, T2, lamda) for i in range(self.len_freq)])
        return output

    def _CondEntropy3(self, T1, T2, lamda):
        output = 0
        for root in range(self.len_freq):
            for char1 in range(self.len_freq):
                for char2 in range(self.len_freq):
                    output += self._JointProb3(root, char1, char2, T1, T2, lamda) * \
                              np.log(self._CondProb3(root, char1, char2, T1, T2, lamda))
        return -1 * output

    ## LHS
    def _JointProb2(self, rootprime, charprime, Tprime, lamda):
        return self.frequency[rootprime] * self.P(lamda, Tprime)[rootprime, charprime]

    def _CondProb2(self, rootprime, charprime, Tprime, lamda):
        output1 = self._JointProb2(rootprime, charprime, Tprime, lamda)
        output2 = np.sum([self._JointProb2(i, charprime, Tprime, lamda) for i in range(self.len_freq)])
        # print()
        return output1 / output2

    def _CondEntropy2(self, Tprime, lamda):
        output = 0
        for root in range(self.len_freq):
            for charprime in range(self.len_freq):
                cprob = self._CondProb2(root, charprime, Tprime, lamda)
                output += self._JointProb2(root, charprime, Tprime, lamda) * np.log(cprob)
        return -1 * output


    def _InfoEquivalent(self, T1, T2, lamda):
        cond_entropy3 = self._CondEntropy3(T1, T2, lamda)
        def to_find_root(Tprime):
            cond_entropy2 = self._CondEntropy2(Tprime[0], lamda)
            return cond_entropy2 - cond_entropy3
        output = optimize.root(to_find_root, np.min([T1, T2]), options={'maxfev': 500}).x[0]
        return output

    def BranchCollapse(self, branch, lamda):
        if not isinstance(branch, Sequence):
            return branch
        elif depth(branch[0]) == 1 and NumberQ(branch[-1]):
            return self._InfoEquivalent(branch[0][0],
                                  branch[0][-1],
                                  lamda) + branch[-1]
        elif not NumberQ(branch[0][0]):
            return self._InfoEquivalent(self.BranchCollapse(branch[0][0], lamda),
                                  self.BranchCollapse(branch[0][-1], lamda),
                                  lamda) + branch[-1]
        else:
            raise ValueError("Something went wrong")

    def TreeCollapse(self, Tree, lamda):
        output = [self.BranchCollapse(Tree[i], lamda) for i in range(len(Tree) - 1)]
        output.append(Tree[-1])
        return output

def main():
    parser = argparse.ArgumentParser(description='''The program depends on scipy, and numpy.\n
The positional arguments must be inputted in the correct order.\n
If the arguments are named they can go in any order.\n
All arguments also have default values and can be excluded\n
Example runs would be:\n
  ./InfoCollapse.py 1 1 1 1 1 1 .25 .25 .25 .25 -i 1 2 3 4 5 -r .5 .6 .7 .8
  ./InfoCollapse.py -a 1 -b 1 1 1 1 1 .25 .25 .25 .25 -i 1 2 3 4 5 -r .5 .6 .7 .8

''')

    parser.add_argument("a", type=float, nargs='?', default=5.26)
    parser.add_argument("b", type=float, nargs='?', default=8.15)
    parser.add_argument("c", type=float, nargs='?', default=1)
    parser.add_argument("d", type=float, nargs='?', default=2.25)
    parser.add_argument("e", type=float, nargs='?', default=3.16)
    parser.add_argument("f", type=float, nargs='?', default=5.44)
    parser.add_argument("Pi_T", type=float, nargs='?', default=0.34)
    parser.add_argument("Pi_C", type=float, nargs='?', default=0.16)
    parser.add_argument("Pi_A", type=float, nargs='?', default=0.32)
    parser.add_argument("Pi_G", type=float, nargs='?', default=0.18)
    parser.add_argument("-r", "--ratevector_file", type=str, default='dummy_rate_vector', help="The string of , comma seperated")
    parser.add_argument("-t", "--tree_file", type=float, default='dummy_tree', help="The tree file in the Newick format")

    args = parser.parse_args()

    a = args.a
    b = args.bn
    c = args.c
    d = args.d
    e = args.e
    f = args.f
    Pi_T = args.Pi_T
    Pi_C = args.Pi_C
    Pi_A = args.Pi_A
    Pi_G = args.Pi_G
    ratevector_file = args.ratevector_file

    info_collapse = Info_Collapse(a,
                                  b,
                                  c,
                                  d,
                                  e,
                                  f,
                                  Pi_T,
                                  Pi_C,
                                  Pi_A,
                                  Pi_G,
                                  ratevector_file)

    return info_collapse

def test():
    info_collapse = main()

    testbranch2 = [[[[20, 15], 40], [[30, 30], 30]], 5]
    testTree = [[[[[10, 10], 30], [[10, 10], 30]], 5],
                [[[[20, 20], 30], [[10, 10], 30]], 5],
                [[[[10, 10], 30], [[20, 20], 30]], 5],
                [[[[20, 20], 30], [[20, 20], 30]], 5],
                1]

    print("Branch Test: {}".format(info_collapse.BranchCollapse(testbranch2, .0008)))
    print()
    print("Tree Test: {}".format(info_collapse.TreeCollapse(testTree, .0008)))
    print()
    for i in np.arange(1 / 10000, 50 /  10000, 1 / 10000):
        print(info_collapse.TreeCollapse(testTree, i))

if __name__ == "__main__":
    info_collapse = main()
