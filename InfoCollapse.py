#! /usr/bin/env python
import numpy as np
from scipy import optimize
from collections import Sequence
from itertools import chain, count
import numbers
import argparse

parser = argparse.ArgumentParser(description='''. The program depends on scipy, and numpy. The positional arguments must be inputted in the correct order. An example run would be ./InfoCollapse.py 1 1 1 1 1 1 .25 .25 .25 .25 -i 1 2 3 4 5 -r .5 .6 .7 .8''')

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
# parser.add_argument("-i", "--internode", nargs='*', type=float, default=[62.4937, 62.4937, 62.4937, 62.4937, 8.9939], help="Must be 5 numbers")
parser.add_argument("-r", "--ratevector", nargs='*', type=float, default=None, help="The long array or rates")
parser.add_argument("-t", "--tree", nargs='*', type=float, default=[62.4937, 62.4937, 62.4937, 62.4937, 8.9939])

args = parser.parse_args()

a    = args.a
b    = args.b
c    = args.c
d    = args.d
e    = args.e
f    = args.f
Pi_T = args.Pi_T
Pi_C = args.Pi_C
Pi_A = args.Pi_A
Pi_G = args.Pi_G

# internode = args.internode
# if len(internode) != 5:
#     raise Exception("Internode distance list not correct")
# ratevector = args.ratevector

def depth(seq):
    for level in count():
        if not seq:
            return level
        seq = list(chain.from_iterable(s for s in seq if isinstance(s, Sequence)))


def NumberQ(x):
    return isinstance(x, numbers.Number)

tree = [62.4937, 62.4937, 62.4937, 62.4937, 8.9939]

frequency = [Pi_T, Pi_C, Pi_A, Pi_G]

len_freq = len(frequency)

mu = 1 / 2 / (a * Pi_T * Pi_C + b * Pi_T * Pi_A + c * Pi_T * Pi_G + d * Pi_C * Pi_A + e * Pi_C * Pi_G + f * Pi_A * Pi_G)
if args.ratevector is None:
    ratevector = np.array([0.000554, 0, 0.001154, 0, 0, 0.001099, 0.001022,
                           0.000406, 0.000459, 0, 0, 0.004695, 0.000406, 0, 0.00063, 0,
                           0.002502, 0.001516, 0.000458, 0, 0.002716, 0.0009, 0, 0.00038,
                           0.005657, 0.001682, 0.012845, 0.015094, 0, 0.002726, 0.00078, 0,
                           0.002689, 0.000416, 0, 0, 0.000601, 0.000652, 0.00285, 0.000448, 0,
                           0, 0.000567, 0, 0.004032, 0.000412, 0.000483, 0.002715, 0,
                           0.000404, 0.00086, 0.004166, 0.000417, 0.001849, 0.000858, 0,
                           0.000746, 0, 0.00396, 0, 0.001436, 0.002641, 0.002485, 0.000752, 0,
                           0.003083, 0.001752, 0, 0.004817, 0.000751, 0.002241, 0.000748,
                           0.002173, 0.000787, 0.00087, 0, 0.001726, 0.000854, 0.001603,
                           0.002878, 0.001246, 0.002058, 0.000757, 0.003021, 0.000364,
                           0.000785, 0.001801, 0.004569, 0.003829, 0.003055, 0.002941, 0,
                           0.004751, 0.007454, 0.001005, 0.004875, 0.000886, 0, 0.000415, 0,
                           0, 0.000366, 0, 0.000362, 0.002414, 0.00357, 0, 0.007102, 0.000807,
                           0.004049, 0.001368, 0, 0, 0, 0, 0, 0.004797, 0, 0, 0, 0.001828,
                           0.002194, 0.003915, 0.001749, 0, 0.002173, 0, 0, 0.001165,
                           0.000363, 0, 0.006264, 0.001164, 0, 0.002899, 0, 0, 0, 0, 0,
                           0.003037, 0.000742, 0.000435, 0.007132, 0, 0, 0.007494, 0, 0,
                           0.003916, 0.000569, 0.000987, 0.003939, 0.003945, 0, 0.002202,
                           0.003442, 0, 0.004277, 0.001325, 0, 0.000368, 0, 0, 0, 0.003026,
                           0.000871, 0.002185, 0, 0, 0.003836, 0, 0, 0.00364, 0, 0, 0.003302,
                           0, 0, 0.00373, 0.001229, 0.00172, 0.001323, 0, 0, 0.004371, 0, 0,
                           0.003349, 0.000416, 0, 0.000364, 0.000859, 0, 0.000364, 0, 0,
                           0.003822, 0.001814, 0, 0.000364, 0.000841, 0.001926, 0.00041, 0, 0,
                           0.000416, 0, 0, 0.000366, 0, 0, 0.004293, 0.001248, 0, 0.001188,
                           0.000874, 0, 0.004006, 0, 0, 0.003724, 0.001424, 0, 0.002922,
                           0.000364, 0, 0.001788, 0, 0, 0.001761, 0.001152, 0, 0.000839, 0, 0,
                           0, 0, 0.000892, 0.002088, 0.001652, 0, 0.004179, 0, 0, 0.003829,
                           0.000416, 0.000425, 0.00038, 0, 0, 0, 0, 0, 0, 0, 0, 0.000858, 0,
                           0, 0, 0, 0, 0, 0, 0, 0.001825, 0, 0, 0.002112, 0, 0, 0.003494, 0,
                           0, 0.004096, 0, 0, 0.001324, 0, 0, 0.001839, 0.001959, 0, 0.002154,
                           0.002005, 0.002457, 0.004605, 0.00299, 0.001185, 0.001797,
                           0.001695, 0.002645, 0.004414, 0, 0.000569, 0.003484, 0.000621,
                           0.000373, 0.00115, 0.002167, 0.002521, 0.005825, 0.002161, 0,
                           0.004777, 0, 0, 0.0016, 0.000411, 0.002274, 0.000431, 0, 0,
                           0.002265, 0, 0, 0.001596, 0, 0, 0, 0.00219, 0, 0.004848, 0.004166,
                           0, 0.00435, 0.00048, 0.00048, 0.001294, 0, 0, 0.004252, 0.000569,
                           0.000363, 0.005216, 0.001249, 0, 0.00143, 0, 0, 0, 0, 0, 0.000796,
                           0.000579, 0.000375, 0.003929, 0, 0, 0.005788, 0.000484, 0.000375,
                           0.000364, 0.000412, 0, 0.000362, 0.001373, 0, 0.001368, 0.000851,
                           0, 0.000366, 0.000416, 0, 0.000364, 0, 0, 0.000415, 0.000397,
                           0.000494, 0.001296, 0, 0, 0, 0.002246, 0, 0.003641, 0, 0, 0.004502,
                           0.001846, 0, 0.005645, 0, 0, 0.00485, 0.002881, 0.001161,
                           0.004192, 0, 0, 0.002715, 0, 0, 0.003676, 0.000576, 0, 0, 0, 0,
                           0.005386, 0, 0, 0.001847, 0.001777, 0, 0.003227, 0.000799, 0,
                           0.000752, 0, 0, 0.001818, 0.005208, 0, 0.004303, 0.001139, 0,
                           0.006145, 0.001229, 0.000478, 0.002993, 0.00161, 0.000481,
                           0.002067, 0.000428, 0.000413, 0.000411, 0, 0, 0.000413, 0, 0,
                           0.002849, 0.002002, 0, 0.002814, 0, 0, 0.000871, 0.001005,
                           0.002606, 0.002448, 0, 0.000462, 0.002338, 0.003053, 0.000532,
                           0.002852, 0.006569, 0, 0.006074, 0, 0, 0.002295, 0.00041, 0.000494,
                           0.003479, 0.000415, 0, 0.000363, 0, 0, 0.002285, 0.001586, 0,
                           0.001419, 0, 0, 0.000366, 0.003765, 0, 0.003537, 0, 0, 0, 0,
                           0.00048, 0.001783, 0.003482, 0.002098, 0.004487, 0, 0, 0.001372, 0,
                           0, 0.002263, 0.002365, 0, 0.001788, 0.000416, 0, 0.000364, 0, 0,
                           0.003434, 0, 0, 0, 0.00056, 0.000435, 0.006337, 0.000381, 0.000531,
                           0.002372, 0, 0, 0.001813, 0.000431, 0, 0.000364, 0.000362, 0,
                           0.007557, 0.000484, 0, 0.000366, 0.001241, 0, 0.001141, 0.000428,
                           0.000433, 0.002673, 0, 0, 0.007125, 0, 0, 0.004283, 0, 0.002435,
                           0.005828, 0, 0, 0.000413, 0, 0, 0.005709, 0.002173, 0.001672,
                           0.001641, 0, 0, 0, 0.00187, 0.003076, 0.003599, 0, 0, 0.004004,
                           0.003055, 0, 0.003212, 0, 0, 0.000432, 0, 0, 0.009151, 0.004399,
                           0.001402, 0.00192, 0.002034, 0.001899, 0.008806, 0, 0, 0.001948,
                           0.005619, 0.002725, 0.005532, 0.005133, 0.001477, 0.005291, 0, 0,
                           0.000595, 0, 0, 0.002502, 0, 0.001008, 0.001608, 0, 0, 0.000444])
else:
    ratevector = args.ratevector

clean_rate = ratevector[ratevector == 0]

Q = mu *  np.array([[-a * Pi_C - b * Pi_A - c * Pi_G, a * Pi_C, b * Pi_A, c * Pi_G],
                    [a * Pi_T, -a * Pi_T - d * Pi_A - e * Pi_G, d * Pi_A, e * Pi_G],
                    [b * Pi_T, d * Pi_C, -b * Pi_T - d * Pi_C - f * Pi_G, f * Pi_G],
                    [c * Pi_T, e * Pi_C, f * Pi_A, -c * Pi_T - e * Pi_C - f * Pi_A]])


evals, py_v = np.linalg.eig(Q) # for some reason py_v seems to already be transposed
tev = py_v
itev = np.linalg.inv(tev)



def P(lamda, T):
    output = np.dot(np.dot(tev, np.diag(np.exp(evals * lamda * T))), itev)
    return output


## RHS
def JointProb3(root, char1, char2, T1, T2, lamda):
    return frequency[root] * P(lamda, T1)[root, char1] * P(lamda, T2)[root, char2]

def CondProb3(root, char1, char2, T1, T2, lamda):
    output = JointProb3(root, char1, char2, T1, T2, lamda)
    output /= np.sum([JointProb3(i, char1, char2, T1, T2, lamda) for i in range(len_freq)])
    return output

def CondEntropy3(T1, T2, lamda):
    output = 0
    for root in range(len_freq):
        for char1 in range(len_freq):
            for char2 in range(len_freq):
                output += JointProb3(root, char1, char2, T1, T2, lamda) * np.log(CondProb3(root, char1, char2, T1, T2, lamda))
    return -1 * output



## LHS

def JointProb2(rootprime, charprime, Tprime, lamda):
    return frequency[rootprime] * P(lamda, Tprime)[rootprime, charprime]

def CondProb2(rootprime, charprime, Tprime, lamda):
    output1 = JointProb2(rootprime, charprime, Tprime, lamda)
    output2 = np.sum([JointProb2(i, charprime, Tprime, lamda) for i in range(len_freq)])
    # print()
    return output1 / output2

def CondEntropy2(Tprime, lamda):
    output = 0
    for root in range(len_freq):
        for charprime in range(len_freq):
            cprob = CondProb2(root, charprime, Tprime, lamda)
            output += JointProb2(root, charprime, Tprime, lamda) * np.log(cprob)
    return -1 * output




def InfoEquivalent(T1, T2, lamda):
    cond_entropy3 = CondEntropy3(T1, T2, lamda)
    def to_find_root(Tprime):
        cond_entropy2 = CondEntropy2(Tprime[0], lamda)
        return cond_entropy2 - cond_entropy3
    output = optimize.root(to_find_root, np.min([T1, T2]), options={'maxfev': 500}).x[0]
    return output

def BranchCollapse(branch, lamda):
    if not isinstance(branch, Sequence):
        return branch
    elif depth(branch[0]) == 1 and NumberQ(branch[-1]):
        return InfoEquivalent(branch[0][0],
                              branch[0][-1],
                              lamda) + branch[-1]
    elif not NumberQ(branch[0][0]):
        return InfoEquivalent(BranchCollapse(branch[0][0], lamda),
                              BranchCollapse(branch[0][-1], lamda),
                              lamda) + branch[-1]
    else:
        raise ValueError("Something went wrong")

def TreeCollapse(Tree, lamda):
    output = [BranchCollapse(Tree[i], lamda) for i in range(len(Tree) - 1)]
    output.append(Tree[-1])
    return output


testbranch2 = [[[[20, 15], 40], [[30, 30], 30]], 5]
testTree = [[[[[10, 10], 30], [[10, 10], 30]], 5],
            [[[[20, 20], 30], [[10, 10], 30]], 5],
            [[[[10, 10], 30], [[20, 20], 30]], 5],
            [[[[20, 20], 30], [[20, 20], 30]], 5],
            1]

print("Branch Test: {}".format(BranchCollapse(testbranch2, .0008)))
print()
print("Tree Test: {}".format(TreeCollapse(testTree, .0008)))
print()
for i in np.arange(1 / 10000, 50 /  10000, 1 / 10000):
    print(TreeCollapse(testTree, i))
