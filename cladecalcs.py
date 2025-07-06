import sys
import os
from collections import defaultdict
import numpy as np
import math
import gzip
import tqdm
import tskit
import tszip
from scipy.stats import norm
from scipy.optimize import brentq
import statsmodels.api as sm

class Clades:
    """
    Class storing list of clades in the given ts
    """
    def __init__(
            self,
            sequence_length,
            size
    ):
        self.name = ""
        self.num = 0  # number of clades
        self.sequence_length = sequence_length

        self.active_clades = {}

        # These are indexed by clade ID
        self.id = np.array([-1] * size, dtype=int)
        self.nodeid = np.array([-1] * size, dtype=int)
        self.binid = [-1] * size
        self.chunkindex = np.array([-1] * size, dtype=int)
        self.treeindex = np.array([-1] * size, dtype=int)
        self.cladesize = np.array([-1] * size, dtype=int)
        self.start = np.array([-1] * size, dtype=float)
        self.end = np.array([-1] * size, dtype=float)
        self.num_mutations = np.array([-1] * size, dtype=int)
        self.span = np.array([-1] * size, dtype=float)

        # This is a dict of mutations (keys = clade ID)
        self.mutations = defaultdict(set)  # positions of all mutations on the MRCA branches of this clade

    def close(self, end, closed_clades):
        # Set clade ends to sequence length and resize all the arrays
        for key, value in self.active_clades.items():
            self.set_span(value, end)
            closed_clades.append(value)
        self.active_clades = {}
        self.id = np.array(self.id[0 : self.num], dtype=int)
        self.nodeid = np.array(self.nodeid[0 : self.num], dtype=int)
        self.binid = self.binid[0 : self.num]
        self.chunkindex = np.array(self.chunkindex[0 : self.num], dtype=int)
        self.treeindex = np.array(self.treeindex[0 : self.num], dtype=int)
        self.cladesize = np.array(self.cladesize[0 : self.num], dtype=int)
        self.start = np.array(self.start[0 : self.num], dtype=float)
        self.end = np.array(self.end[0 : self.num], dtype=float)
        self.num_mutations = np.array(self.num_mutations[0 : self.num], dtype=int)
        self.span = np.array(self.span[0 : self.num], dtype=float)

    def write_to_file(self, filehandle, clade_indices=None):
        with gzip.open(filehandle + ".clades.gz", "at") as file:
            for i in clade_indices:
                if self.binid[i] is None:
                    sys.exit(str(i) + " binid is None")
                file.write(
                    str(i)
                    + ";"
                    + str(self.nodeid[i])
                    + ";"
                    + str(self.binid[i])
                    + ";"
                    + str(self.chunkindex[i])
                    + ";"
                    + str(self.treeindex[i])
                    + ";"
                    + str(self.cladesize[i])
                    + ";"
                    + str(self.start[i])
                    + ";"
                    + str(self.end[i])
                    + ";"
                    + str(self.num_mutations[i])
                    + ";"
                    + str(self.span[i])
                    + ";"
                    + " ".join(str(m) for m in self.mutations[i])
                    + ";\n"
                )


    def fix_numbering(self, filehandle):
        if not os.path.exists(filehandle + ".clades.gz"):
            sys.exit("Error: no .clades.gz file found.")
        else:
            with gzip.open(filehandle + "_.clades.gz", "wt") as newfile:
                with gzip.open(filehandle + ".clades.gz", "rt") as oldfile:
                    for l, line in enumerate(oldfile):
                        if l == 0:
                            newfile.write("NUM_CLADES " + str(self.num) + "\n")
                        elif l == 1:
                            newfile.write(
                                "SEQUENCE_LENGTH " + str(self.sequence_length) + "\n"
                            )
                        else:
                            newfile.write(line)
        os.system("mv " + filehandle + "_.clades.gz " + filehandle + ".clades.gz ")


    def add_mutations(self, i, mut):
        if len(mut) > 0:
            self.num_mutations[i] += len(mut)
            self.mutations[i].update(mut)

    def add_clade(
            self,
            binid,
            nodeid,
            chunkindex,
            treeindex,
            cladesize,
            start,
    ):
        self.id[self.num] = self.num
        self.nodeid[self.num] = nodeid
        self.binid[self.num] = binid
        self.chunkindex[self.num] = chunkindex
        self.treeindex[self.num] = treeindex
        self.active_clades[binid] = self.num  # new clade is active
        self.cladesize[self.num] = cladesize
        self.start[self.num] = start
        self.num_mutations[self.num] = 0
        self.num += 1

        return self.num - 1

    def set_span(self, i, end):
        self.end[i] = end
        self.span[i] = self.end[i] - self.start[i]

def clade_span(ts_list, num_trees, num_samples, write_to_file=None, write_to_file_freq=None):
    """
    Compute the span of clades in a given tree sequence.
    This is defined by the samples subtending a clade staying the same.
    :param ts_list: list of tskit tree sequence file handles with extensions (will be loaded as we go)
    :param num_trees: total number of trees in full ts (excluding the empty trees)
    :param num_samples: number of samples in each ts
    :param write_to_file: filename for outputting clades periodically
    :param write_to_file_freq: how often to output clades to file
    :return:
    """

    clades = Clades(None, size=int(num_trees * (num_samples - 2) / 2))
    closed_clades = []
    tree_counter = -1
    left = 0.0
    if write_to_file is not None:
        with gzip.open(write_to_file + ".clades.gz", "wt") as file:
            file.write("NUM_CLADES 0\n")
            file.write("SEQUENCE_LENGTH None\n")

    with tqdm.tqdm(total=num_trees, desc="Computing clade spans",
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
        for i, ts_handle in enumerate(ts_list):
            if os.path.exists(ts_handle + ".tsz"):
                ts = tszip.decompress(ts_handle + ".tsz")
            else:
                ts = tskit.load(ts_handle)
            if i == len(ts_list) - 1:
                clades.sequence_length = ts.sequence_length

            for t in ts.trees():
                if (
                    t.num_roots == 1
                ):  # Sometimes first tree in ts out of stdpopsim is empty so skip it
                    tree_counter += 1
                    left = max(left, t.interval[0])  # This is for having multiple Relate ts parts
                    bitset = {}  # record node IDs as an array
                    bitset_ = set()  # record node IDs as a string
                    prevclades = list(clades.active_clades.values())  # clades in previous tree
                    tree_muts = defaultdict(set)

                    # Dictionary of {node_tskit_id : set(mutation_positions)}
                    for mut in t.mutations():
                        tree_muts[mut.node].add(ts.site(mut.site).position)

                    for g in t.nodes(order='timeasc'):
                        if g != t.root:
                            if t.is_sample(g):
                                # Just record the bitset
                                g_id = 1 << g
                                bitset[g] = g_id
                            else:
                                m = tree_muts[g]  # Set of mutation positions above this node in this tree
                                g_id = 0
                                for ch in t.children(g):
                                    g_id = g_id | bitset[ch]
                                cladesize = g_id.bit_count()
                                bitset[g] = g_id
                                if t.num_samples(g) != cladesize:
                                    sys.exit("Error: clade size per genotype ID does not match tree.")

                                if g_id in clades.active_clades:
                                    # The clade was there in the previous tree, so record mutations and
                                    # remove from prevclades (so prevclades will become the list of
                                    # clades that have disappeared).
                                    p = clades.active_clades[g_id]
                                    clades.add_mutations(p, m)
                                    if p in prevclades:
                                        prevclades.remove(p)
                                    # Add number of mutations above g to clade mutation count
                                else:
                                    # This is a new clade
                                    # (this will be recorded as active when added)
                                    if g_id is None:
                                        sys.exit("Error: binid is None")
                                    p = clades.add_clade(
                                        binid=g_id,  # this is number or string
                                        nodeid=g,
                                        chunkindex=i,
                                        treeindex=t.index,
                                        cladesize=cladesize,
                                        start=left,
                                    )
                                    clades.add_mutations(p, m)
                    for p in prevclades:
                        # These clades have disappeared
                        clades.set_span(p, left)
                        closed_clades.append(p)

                    clades.active_clades = {
                        key: val
                        for key, val in clades.active_clades.items()
                        if val not in prevclades
                    }

                    if i == len(ts_list) - 1 and t.index == ts.num_trees - 1:
                        clades.close(t.interval[1], closed_clades)
                        if write_to_file is not None:
                            clades.write_to_file(write_to_file, closed_clades)
                            clades.fix_numbering(write_to_file)
                        pbar.update(1)
                        break

                    if write_to_file is not None:
                        if len(closed_clades) >= write_to_file_freq:
                            clades.write_to_file(write_to_file, closed_clades)
                            for d in closed_clades:
                                if clades.binid[d] is None:
                                    sys.exit("binid is already None")
                                del clades.mutations[d]
                                clades.binid[d] = None
                            closed_clades = []

                    pbar.update(1)

    print("Done, traversed", tree_counter + 1, "trees, out of", num_trees)
    if write_to_file is not None:
        clades = None

    return clades

def read_from_file(filehandle, p1=0, p2=math.inf):
    """
    Read in clades with positions in given range
    :param filehandle: handle of clades file (with no .clades.gz extension)
    :param p1: minimum position of clades to read in
    :param p2: maximum position of clades to read in
    :return:
    """
    maxp = 0
    filename = filehandle + ".clades.gz"
    if not os.path.exists(filename):
        sys.exit("Error: no .clades.gz file found.")
    else:
        info = [""] * 2
        with gzip.open(
            filename,
            "rt",
        ) as file:
            for l, line in enumerate(file):
                if l <= 1:
                    line = line.strip().split(" ")
                    if len(line) > 1:
                        info[l] = line[1]
                else:
                    if l == 2:
                        clades = Clades(float(info[1]), int(info[0]))
                        clades.num = int(info[0])
                    line = line.strip().split(";")
                    start = float(line[6])
                    end = float(line[7])
                    maxp = max(maxp, end)
                    if start < p2 and p1 < end:
                        i = int(line[0])
                        clades.id[i] = i
                        clades.nodeid[i] = int(line[1])
                        clades.binid[i] = line[2]
                        clades.chunkindex[i] = int(line[3])
                        clades.treeindex[i] = int(line[4])
                        clades.cladesize[i] = int(line[5])
                        clades.start[i] = start
                        clades.end[i] = end
                        clades.num_mutations[i] = int(line[8])
                        clades.span[i] = float(line[9])
                        if line[10] != '':
                            clades.mutations[i] = set({float(m) for m in line[10].strip().split(" ")})

                        if clades.start[i] == -1 or clades.end[i] == -1 or clades.span[i] <= 0:
                            sys.exit("Error: clade start, end or span is not recorded properly.")
                        if clades.start[i] > clades.end[i]:
                            sys.exit("Error: clade start cannot be greater than clade end.")

    print("Read in", clades.num, "clades")
    return clades, int(maxp)

def get_chunk(filepath, position):
    """
    Finding the tree sequence chunk index for each SNP
    :param filepath: filepath of treeinfo file
    :param position: position in GRCh37 coords
    :return:
    """
    chunkindex = None
    with open(filepath, "r") as file:
        for l, line in enumerate(file):
            if l > 0:
                line = line.strip().split(";")
                if int(line[3]) <= position < int(line[4]) and not (
                    int(line[3]) == 0 and int(line[4]) - int(line[3]) > 10000000
                ):
                    chunkindex = int(line[1])
    return chunkindex

def R2(set1, set2, NN):
    """
    Calculate r^2 between two sets
    :param set1: set 1
    :param set2: set 2
    :param NN: total sample size
    :return:
    """
    s = len(set1 & set2)
    s1 = len(set1)
    s2 = len(set2)
    return (s*NN - s1*s2)**2 / (s1*s2*(NN-s1)*(NN-s2))


def solve_for_b(P, C, beta_partial):
    """
    Numerically solves for the effect size b from equation (S9).
    :param P: probability P
    :param C: significance threshold from the original test (e.g., 1.96 for alpha=0.05)
    :param beta_partial: estimated partial regression coefficient (beta_{z,s|x})
    """

    def equation_to_solve(b_abs):
        # We assume b_abs is positive and use beta_partial's sign
        # to determine the direction of the effect.
        term = b_abs * beta_partial
        calculated_prob = (norm.cdf(-C + term) + (1 - norm.cdf(C + term)))
        return calculated_prob - P

    try:
        # Use a numerical solver (Brent's method) to find the root.
        # We search for a solution for b in the interval [0, 5].
        # The solver will find the positive root for b.
        solution_b = brentq(equation_to_solve, a=0, b=5)
        return solution_b
    except ValueError:
        return np.nan


def set_to_vector(A, n_total_samples):
    """
    Converts set of sample indices into binary genotype vectors.
    """
    A_vec = np.zeros(n_total_samples)
    A_vec[list(A)] = 1

    return A_vec


def sets_to_vectors(Z, S, X1, X2, n_total_samples):
    """
    Converts sets of sample indices into binary genotype vectors.
    """
    z_vec = set_to_vector(Z, n_total_samples)
    s_vec = set_to_vector(S, n_total_samples)
    x1_vec = set_to_vector(X1, n_total_samples)
    x2_vec = set_to_vector(X2, n_total_samples)

    return z_vec, s_vec, x1_vec, x2_vec


def standardize(variable):
    """Converts a variable to have a mean of 0 and a standard deviation of 1."""
    return (variable - np.mean(variable)) / np.std(variable, ddof=1)


def estimate_beta_partial(z, s, x1, x2):
    """
    Estimates the standardized partial regression coefficient of s on z,
    conditional on x1 and x2.
    """
    z_std, s_std, x1_std, x2_std = map(standardize, [z, s, x1, x2])
    X = np.column_stack((x1_std, x2_std, s_std))
    model = sm.OLS(z_std, X)
    results = model.fit()
    # The coefficient for 's' is the last one in the parameter list
    beta_hat = results.params[-1]
    return beta_hat


# def check_conditions(S, A, notA, M1, M2, N, P, C):
#     """
#     Check whether A provides evidence against existence of a clade Z such that
#     significance inequality is satisfied
#     :param S: samples in S
#     :param A: samples in A
#     :param M1: samples carrying SNP1
#     :param M2: samples carrying SNP2
#     :param N: total sample size
#     :return:
#     """
#     S_vec, A_vec, M1_vec, M2_vec = sets_to_vectors(S, A, M1, M2, N)
#
#     Z1 = A & S
#     Z2 = notA & S
#     Z3 = A | S
#     Z4 = A - S
#     Z5 = notA - S
#     Z6 = A & Z5
#
#     b_min = math.inf
#
#     for Z in [Z1, Z2, Z3, Z4, Z5, Z6]:
#         Z_vec = set_to_vector(Z, N)
#         beta = estimate_beta_partial(Z_vec, S_vec, M1_vec, M2_vec)
#         b = solve_for_b(P, C, beta)
#         if b < b_min:
#             b_min = b
#
#     return b_min


def check_conditions_fast(Y, notY, X, N, P, C, pseudo_inverse):
    """
    A much faster version of check_conditions that uses a pre-calculated
    pseudo-inverse of the design matrix to avoid repeated regressions.

    Args:
        Y (set): The set of samples for the clade.
        X (set): The set of samples for the target (e.g. S, K1, K2, O).
        N (int): Total sample size.
        P (float): Target probability for solve_for_b.
        C (float): Significance threshold for solve_for_b.
        pseudo_inverse (np.array): The pre-calculated pseudo-inverse of the standardized design matrix.

    Returns:
        float: The minimum 'b' value.
    """
    # Define the 6 Z sets based on the clade (Y) and the target (X)
    Z_sets = [
        Y & X,
        notY & X,
        Y | X,
        Y - X,
        notY - X,
        Y | (notY - X),
    ]

    b_min = np.inf

    # Convert all 6 Z sets to standardized vectors
    z_vectors_std = [standardize(set_to_vector(z_set, N)) for z_set in Z_sets]

    for z_std in z_vectors_std:
        beta_coeffs = pseudo_inverse @ z_std
        beta_partial = beta_coeffs[-1]
        b = solve_for_b(P, C, beta_partial)
        if b < b_min:
            b_min = b

    return b_min


# def calculate_all_bmins_for_clade(Y, M1, M2, S, K1, K2, O, N, P, C):
#     """
#     Calculates the minimum effect size 'b' for a given clade Y across all
#     four target sets (S, K1, K2, O).
#
#     This function consolidates the loops to be more efficient.
#
#     Returns:
#         list: A list of four 'b_min' values, one for each target set.
#     """
#
#     M1n = {s for s in range(N) if s not in M1}
#     M2n = {s for s in range(N) if s not in M2}
#     notY = {s for s in range(N) if (s not in Y)}
#
#     target_sets = [
#         (S, M1, M2),  # Target S uses main effects M1, M2
#         (K1, M1, M2n),  # Target K1 uses M1, not M2
#         (K2, M1n, M2),  # Target K2 uses not M1, M2
#         (O, M1n, M2n)  # Target O uses not M1, not M2
#     ]
#
#     bmins = [np.inf] * 4
#
#     for i, (X, MM1, MM2) in enumerate(target_sets):
#         bmin = check_conditions(X, Y, notY, MM1, MM2, N, P, C)
#         bmins[i] = bmin
#
#     return bmins


def get_searchrange(pos, rec_map, d=1.0):
    """
    Output genomic positions (left, right) such that rec_map(right) - rec_map(left) == d
    :param pos: position of SNP
    :param rec_map: recombination map
    :param d: distance in cM
    :return:
    """
    if pos > rec_map.sequence_length:
        print("Cannot set searchrange, outside rec_map limits.")
        return int(pos - 1000000), math.inf
    P = rec_map.get_cumulative_mass(pos)
    sl = rec_map.sequence_length
    S = [0, sl]
    A = np.arange(max(0, pos - 15000000), pos - 1, step=100)
    A = A[::-1]
    B = np.arange(pos, min(pos + 15000000, sl - 1), step=100)
    AA = rec_map.get_cumulative_mass(A)
    BB = rec_map.get_cumulative_mass(B)
    for i, (X, XX) in enumerate([(A, AA), (B, BB)]):
        if max(abs(XX - P)) >= d/100:
            for j, x in enumerate(X):
                if abs(XX[j] - P) >= d/100:
                    S[i] = x
                    break
    # print(S[0], pos - S[0], (P - rec_map.get_cumulative_mass(S[0]))*100)
    # print(S[1], pos - S[1], (P - rec_map.get_cumulative_mass(S[1]))*100)
    return int(S[0]), int(S[1])

