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

_sqrt2 = math.sqrt(2)

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

def clade_span(
    ts_list,
    num_trees,
    num_samples,
    start=0,
    end=math.inf,
    write_to_file=None,
    write_to_file_freq=None,
):
    """
    Compute the span of clades in a given tree sequence.
    This is defined by the samples subtending a clade staying the same.
    :param ts_list: list of tskit tree sequence file handles with extensions (will be loaded as we go)
    :param num_trees: total number of trees in full ts (excluding the empty trees)
    :param num_samples: number of samples in each ts
    :param start: start of region to compute
    :param end: end of region to compute
    :param write_to_file: filename for outputting clades periodically
    :param write_to_file_freq: how often to output clades to file
    :return:
    """
    clades = Clades(None, size=int(num_trees * (num_samples - 2) / 2))
    closed_clades = []
    tree_counter = -1

    if write_to_file is not None:
        with gzip.open(write_to_file + ".clades.gz", "wt") as file:
            file.write("NUM_CLADES 0\n")
            file.write("SEQUENCE_LENGTH None\n")

    with tqdm.tqdm(
        total=num_trees,
        desc="Computing clade spans",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ) as pbar:
        for i, ts_handle in enumerate(ts_list):
            if os.path.exists(ts_handle + ".tsz"):
                ts = tszip.decompress(ts_handle + ".tsz")
            elif ts_handle[-3:] == "tsz":
                ts = tszip.decompress(ts_handle)
            else:
                ts = tskit.load(ts_handle)
            if i == len(ts_list) - 1:
                clades.sequence_length = ts.sequence_length

            for t in ts.trees():
                if t.num_roots == 1:
                    tree_counter += 1
                    tree_interval_start = t.interval[0]

                    # Cap start and end points at the region boundaries
                    clade_start_pos = max(tree_interval_start, start)
                    clade_end_pos = min(tree_interval_start, end)

                    # Only do processing if the tree overlaps the region
                    if t.interval[0] < end and t.interval[1] > start:
                        bitset = {}
                        prevclades = list(clades.active_clades.values())
                        tree_muts = defaultdict(set)

                        for mut in t.mutations():
                            tree_muts[mut.node].add(ts.site(mut.site).position)

                        for g in t.nodes(order="timeasc"):
                            if g != t.root:
                                if t.is_sample(g):
                                    bitset[g] = 1 << g
                                else:
                                    m = tree_muts[g]
                                    g_id = 0
                                    for ch in t.children(g):
                                        g_id = g_id | bitset[ch]
                                    cladesize = g_id.bit_count()
                                    bitset[g] = g_id
                                    if t.num_samples(g) != cladesize:
                                        sys.exit("Error: clade size does not match tree.")

                                    if g_id in clades.active_clades:
                                        p = clades.active_clades[g_id]
                                        clades.add_mutations(p, m)
                                        if p in prevclades:
                                            prevclades.remove(p)
                                    else:
                                        p = clades.add_clade(
                                            binid=g_id, nodeid=g, chunkindex=i,
                                            treeindex=t.index, cladesize=cladesize,
                                            start=clade_start_pos,
                                        )
                                        clades.add_mutations(p, m)

                        for p in prevclades:
                            clades.set_span(p, clade_end_pos)
                            closed_clades.append(p)

                        clades.active_clades = {
                            key: val for key, val in clades.active_clades.items()
                            if val not in prevclades
                        }

                if write_to_file is not None and write_to_file_freq is not None:
                    if len(closed_clades) >= write_to_file_freq:
                        clades.write_to_file(write_to_file, closed_clades)
                        # Clear memory for clades that have been written
                        for d in closed_clades:
                            del clades.mutations[d]
                            clades.binid[d] = None
                        closed_clades = []

                pbar.update(1)

    final_end_pos = min(getattr(clades, 'sequence_length', end), end)
    remaining_clades = list(clades.active_clades.keys())
    for binid in remaining_clades:
        clade_index = clades.active_clades[binid]
        clades.set_span(clade_index, final_end_pos)
        closed_clades.append(clade_index)

    # Close remaining active clades and write any unwritten closed clades to file
    if write_to_file is not None:
        clades.write_to_file(write_to_file, closed_clades)
        clades.close(final_end_pos, []) # Pass empty list as they've just been written
        clades.fix_numbering(write_to_file)
    else:
        clades.close(final_end_pos, closed_clades)

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

    if s1 == 0 or s2 == 0 or s1 == NN or s2 == NN:
        return 0.0

    return (s*NN - s1*s2)**2 / (s1*s2*(NN-s1)*(NN-s2))


def check_clade_probability(P, C, beta_partial, beta_s_sd, maxeffectsize):
    # We set b = maxeffectsize (as a very high threshold) and check if resulting P is greater than the input P
    # If not, there is no sense testing this clade because b would have to be greater than maxeffectsize
    # (which is implausibly high)
    term = maxeffectsize * beta_partial / beta_s_sd
    calculated_prob = fast_norm_cdf(- C - term) + (1 - fast_norm_cdf(C - term))
    # term_ = 1 * beta_partial
    # calculated_prob_ = norm.cdf(- C - term_) + (1 - norm.cdf(C - term_))
    # print("check clade:", P, maxeffectsize, beta_partial, calculated_prob, calculated_prob_)
    return calculated_prob > P

def fast_norm_cdf(x):
    """A fast approximation of the normal CDF using math.erf."""
    return 0.5 * (1 + math.erf(x / _sqrt2))

def solve_for_b(P, C, beta_partial, beta_s_sd, maxeffectsize):
    """
    Numerically solves for the effect size b from equation (S9).
    If resulting b > 1 this will return np.inf
    :param P: probability P
    :param C: significance threshold from the original test (e.g., 1.96 for alpha=0.05)
    :param beta_partial: estimated partial regression coefficient (beta_{z,s|x})
    """

    def equation_to_solve(b_abs):
        # We assume b_abs is positive and use beta_partial's sign
        # to determine the direction of the effect.
        term = b_abs * beta_partial / beta_s_sd
        calculated_prob = fast_norm_cdf(- C - term) + (1 - fast_norm_cdf(C - term))
        return calculated_prob - P

    b = np.inf
    cont = True

    if check_clade_probability(P, C, beta_partial, beta_s_sd, maxeffectsize):
        try:
            # Use a numerical solver (Brent's method) to find the root.
            # We search for a solution for b in the interval [0, maxeffectsize + 10].
            # The solver will find the positive root for b.
            b = brentq(equation_to_solve, a=0, b=maxeffectsize + 10)
            # print("solve for b:", P, b, equation_to_solve(b))
        except ValueError:
            return np.inf, cont
    else:
        cont = False

    return b, cont


def solve_for_b_part2(teststat, D, beta_partial, beta_s_sd, maxeffectsize):
    return min(maxeffectsize, abs((abs(teststat) - D) * beta_s_sd / beta_partial))


def set_to_vector(A, n_total_samples):
    """
    Converts set of sample indices into binary genotype vectors.
    """
    A_vec = np.zeros(n_total_samples)
    A_vec[list(A)] = 1

    return A_vec


def standardize(variable):
    """Converts a variable to have a mean of 0 and a standard deviation of 1."""
    s = np.std(variable, ddof=1)
    if s == 0:
        return np.zeros_like(variable)
    else:
        return (variable - np.mean(variable)) / s


def standardize_tensor(tensor, axis=None):
    """
    Standardizes a NumPy tensor along a specified axis.

    Args:
        tensor (np.ndarray): The input NumPy array to be standardized.
        axis (int, optional): The axis along which to perform the standardization.
                              If None, the tensor is flattened before standardization.
                              Defaults to None.

    Returns:
        np.ndarray: The standardized tensor with the same shape as the input.
    """
    mean = np.mean(tensor, axis=axis, keepdims=True)
    std = np.std(tensor, axis=axis, keepdims=True)
    std[std == 0] = 1

    return (tensor - mean) / std


def run_all_tests(P_range, C, D, teststat, maxeffectsize, beta_s_sd, beta_partial_t1=None, beta_partials_t2=None):
    """
    Runs Test 1 and Test 2 on a clade in one go.
    """
    # --- Test 1 ---
    bmin_test1 = [np.inf] * len(P_range)
    # print("test 1")
    for p, P in enumerate(P_range):
        b, cont = solve_for_b(P, C, beta_partial_t1, beta_s_sd, maxeffectsize)
        bmin_test1[p] = b
        if not cont:
            break

    bmin_test1_part2 = np.inf
    if teststat is not None:
        bmin_test1_part2 = solve_for_b_part2(teststat, D, beta_partial_t1, beta_s_sd, maxeffectsize)

    # --- Test 2 ---
    if beta_partials_t2 is not None:
        bmin_test2 = [bmin for bmin in bmin_test1]
        for beta in beta_partials_t2:
            if beta != 0:
                for p, P in enumerate(P_range):
                    b, cont = solve_for_b(P, C, beta, beta_s_sd, maxeffectsize)
                    bmin_test2[p] = min(bmin_test2[p], b)
                    if not cont:
                        break
    else:
        bmin_test2 = [0] * len(P_range)

    return bmin_test1, bmin_test1_part2, bmin_test2


def get_searchrange(pos, rec_map, d=1.0):
    """
    More efficient version using a numerical solver.
    """
    sl = rec_map.sequence_length
    target_cm_dist = d / 100.0
    current_pos_cm = rec_map.get_cumulative_mass(pos)

    # Define functions whose roots are the boundaries we seek
    def left_boundary_func(x):
        return current_pos_cm - rec_map.get_cumulative_mass(x) - target_cm_dist

    def right_boundary_func(x):
        return rec_map.get_cumulative_mass(x) - current_pos_cm - target_cm_dist

    # Set reasonable search brackets (e.g., 20 Mbp)
    search_bracket_bp = 20000000

    left_bracket = max(0, pos - search_bracket_bp)
    right_bracket = min(sl, pos + search_bracket_bp)

    try:
        # Solve for the left and right boundaries
        left_bound = brentq(left_boundary_func, left_bracket, pos)
    except ValueError:
        # If no root is found, fall back to the bracket edge
        left_bound = left_bracket

    try:
        right_bound = brentq(right_boundary_func, pos, right_bracket)
    except ValueError:
        right_bound = right_bracket

    return int(left_bound), int(right_bound)

