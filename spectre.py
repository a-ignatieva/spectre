#!/usr/bin/env python
# coding: utf-8

import tszip
import tskit
import stdpopsim
import numpy as np
import scipy
import tqdm
from collections import defaultdict
import sys
import os
import getopt
import gc
from matplotlib import pyplot as plt

import cladecalcs

col_green = "#228833"
col_red = "#EE6677"
col_purp = "#AA3377"
col_blue = "#66CCEE"
col_yellow = "#CCBB44"
col_indigo = "#4477AA"
col_grey = "#BBBBBB"

def main(argv):
    path = ""
    chrA = posA = chrB = posB = None
    chrC = posC = -1
    h2_est = b_est = alpha = n_reg = RHS = None
    plot_on = True
    try:
        opts, args = getopt.getopt(argv,
                                   "hf:A:a:B:b:CcHETNp",
                                   [
                                       "filepath=",
                                       "chrSNP1=",
                                       "posSNP1=",
                                       "chrSNP2=",
                                       "posSNP2=",
                                       "chrSNP3=",
                                       "posSNP3=",
                                       "h2=",
                                       "b=",
                                       "alpha=",
                                       "N=",
                                       "plot=",
                                   ]
                                   )
    except getopt.GetoptError:
        print ("For usage: python -m spectre -h")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("")
            print ("python -m spectre -f <filepath> -A <chrSNP1> -a <posSNP1> -B <chrSNP2> -b <posSNP2>")
            print("-"*100)
            print("Example:")
            print("python -m spectre -f example -A 20 -a 15489683 -B 20 -b 15620278")
            print("-"*100)
            print("Options:")
            print("-"*100)
            print("-f", "--filepath", ": Directory where ARGs are stored (in ts format). Output folder will be created here.")
            print("-A", "--chrSNP1", ": Chromosome of SNP1 (int)")
            print("-a", "--posSNP1", ": Position of SNP1 (int)")
            print("-B", "--chrSNP2", ": Chromosome of SNP2 (int)")
            print("-b", "--posSNP2", ": Position of SNP2 (int)")
            print("-C", "--chrSNP3", ": Chromosome of SNP3 (int), optional")
            print("-c", "--posSNP3", ": Position of SNP3 (int), optional")
            print("-H", "--h2", ": Estimate of trait heritability, optional")
            print("-E", "--b", ": Estimate of true effect size, optional")
            print("-T", "--alpha", ": Significance threshold for interaction test, optional")
            print("-N", "--N", ": Total sample size for interaction test, optional")
            print("-p", "--plot", ": Whether to create plot of results, optional")
            print("-"*100)
            sys.exit()
        elif opt in ("-f", "--filepath"):
            path = arg
        elif opt in ("-A", "--chrSNP1"):
            chrA = int(arg)
        elif opt in ("-a", "--posSNP1"):
            posA = int(arg)
        elif opt in ("-B", "--chrSNP2"):
            chrB = int(arg)
        elif opt in ("-b", "--posSNP2"):
            posB = int(arg)
        elif opt in ("-C", "--chrSNP3"):
            chrC = int(arg)
        elif opt in ("-c", "--posSNP3"):
            posC = int(arg)
        elif opt in ("-H", "--h2"):
            h2_est = int(arg)
        elif opt in ("-E", "--b"):
            b_est = int(arg)
        elif opt in ("-T", "--alpha"):
            alpha = int(arg)
        elif opt in ("-N", "--N"):
            n_reg = int(arg)
        elif opt in ("-p", "--plot"):
            plot_on = bool(arg)

    if chrA is None or posA is None or chrB is None or posB is None:
        print ("For usage: python -m run -h")
        sys.exit(2)

    if h2_est is not None and alpha is not None and b_est is not None and n_reg is not None:
        sigdsq = 1 - h2_est
        C = scipy.stats.norm.ppf(1 - alpha / 2)
        RHS = C / b_est * np.sqrt(sigdsq / n_reg)

    d = 1.0  # cM to search around each SNP
    labels = {0: "S", 1: "K1", 2: "K2", 3: "O"}
    output_dir = path + "/output"
    trees_dir = path
    if not os.path.exists(output_dir):
        print("Making results directory: " + output_dir)
        os.mkdir(output_dir)

    print("=" * 100)
    print("Inputs:")
    print("SNP1:", chrA, posA)
    print("SNP2:", chrB, posB)
    print("SNP3:", chrC, posC)
    print("h^2 estimate:", h2_est)
    print("b estimate:", b_est)
    print("n:", n_reg)
    print("alpha:", alpha)
    print("=" * 100)

    # ===========================================================================================
    # Splitting up tree sequence and calculating clade spans
    # ===========================================================================================

    chr_list = [chrA]
    if chrB not in chr_list:
        chr_list.append(chrB)
    if chrC != -1 and chrC not in chr_list:
        chr_list.append(chrC)
    for chromosome in chr_list:
        if chromosome != -1:
            print("Splitting up trees for chr" + str(chromosome) + "...")
            filename_trees = trees_dir + "/trees_chr" + str(chromosome) + ".trees"
            if os.path.exists(filename_trees + ".tsz"):
                ts = tszip.decompress(filename_trees + ".tsz")
            elif os.path.exists(filename_trees):
                ts = tskit.load(filename_trees)
            else:
                sys.exit("No tree sequence file found in " + trees_dir)

            num_samples = ts.num_samples
            num_trees = ts.num_trees
            bps = ts.breakpoints(as_array=True)
            num_chunks = int(len(bps) / 1000) + 1

            for i, j in enumerate(range(num_chunks)):
                filename_chunk = (
                        output_dir
                        + "/trees_chr"
                        + str(chromosome)
                        + "_chunk"
                        + str(j)
                        + ".trees"
                )
                if not os.path.exists(filename_chunk) and not os.path.exists(filename_chunk + ".tsz"):
                    print("...chunk", j)
                    left = bps[i*1000]
                    if (i+1)*1000 >= len(bps):
                        right = bps[-1]
                    else:
                        right = bps[(i+1)*1000]
                    ts_sub = ts.keep_intervals([[left, right]])
                    tszip.compress(ts_sub, filename_chunk + ".tsz")
                    del ts_sub
                    gc.collect()
                else:
                    print("...chunk", j, "already done")
            del ts
            gc.collect()
            print("Done")

            if not os.path.exists(output_dir + "/treeinfo_chr" + str(chromosome) + ".txt"):
                print("Getting tree info...")
                with open(output_dir + "/treeinfo_chr" + str(chromosome) + ".txt", "w") as file:
                    file.write("chr;chunk;chunk_tree_index;tree_start;tree_end\n")
                    for i in range(num_chunks):
                        print("Loading chunk " + str(i) + "...")
                        filename = output_dir + "/trees_chr" + str(chromosome) + "_chunk" + str(i) + ".trees"
                        if os.path.exists(filename + ".tsz"):
                            ts = tszip.decompress(filename + ".tsz")
                        else:
                            ts = tskit.load(filename)
                        for t in ts.trees():
                            if t.num_roots == 1:
                                file.write(
                                    "chr"
                                    + str(chromosome)
                                    + ";"
                                    + str(i)
                                    + ";"
                                    + str(t.index)
                                    + ";"
                                    + str(int(t.interval[0]))
                                    + ";"
                                    + str(int(t.interval[1]))
                                    + "\n"
                                )

            filename = output_dir + "/clades_chr" + str(chromosome)
            if not os.path.exists(filename + ".clades.gz"):
                ts_handles = [output_dir + "/trees_chr" + str(chromosome) + "_chunk" + str(i) + ".trees" for i in range(num_chunks)]
                _ = cladecalcs.clade_span(ts_handles, num_trees, num_samples, write_to_file=filename, write_to_file_freq=50000)
                del ts_handles
                gc.collect()
            else:
                print("Calculating clade spans already done")

    # ===========================================================================================
    # Getting mutation info from tree sequences
    # ===========================================================================================

    chunkA = cladecalcs.get_chunk(output_dir + "/treeinfo_chr" + str(chrA) + ".txt", posA)
    chunkB = cladecalcs.get_chunk(output_dir + "/treeinfo_chr" + str(chrB) + ".txt", posB)
    if chrC != -1:
        chunkC = cladecalcs.get_chunk(output_dir + "/treeinfo_chr" + str(chrC) + ".txt", posC)

    print("="*100)

    print("Loading tree sequences...")
    # Load ts chunks at each SNP
    ts1 = tszip.decompress(
        output_dir
        + "/trees_chr"
        + str(chrA)
        + "_chunk"
        + str(chunkA)
        + ".trees.tsz"
    )
    ts2 = tszip.decompress(
        output_dir
        + "/trees_chr"
        + str(chrB)
        + "_chunk"
        + str(chunkB)
        + ".trees.tsz"
    )
    ts3 = None
    if chrC != -1:
        ts3 = tszip.decompress(
            output_dir
            + "/trees_chr"
            + str(chrC)
            + "_chunk"
            + str(chunkC)
            + ".trees.tsz"
        )
    N = ts1.num_samples
    print("N:", N)

    print("Getting SNPs and target sets...")
    # Calculating target sets
    M1 = set()
    M2 = set()
    M3 = set()
    for ts, M, pos, label in [
        (ts1, M1, posA, "SNP1"),
        (ts2, M2, posB, "SNP2"),
        (ts3, M3, posC, "SNP3"),
    ]:
        if ts is not None:
            for m in ts.mutations():
                if int(ts.site(m.site).position) == pos:
                    t = ts.at(ts.site(m.site).position)
                    M.update({s for s in t.samples(m.node)})
        print("Frequency of " + label + ":", round(len(M) / N, 2))

    # Target sets
    S = {s for s in M1 if s in M2}
    K1 = {s for s in M1 if s not in M2}
    K2 = {s for s in M2 if s not in M1}
    O = {s for s in range(N) if (s not in M1 and s not in M2)}
    M1n = {s for s in range(N) if s not in M1}
    M2n = {s for s in range(N) if s not in M2}

    # Frequencies of SNP1, SNP2, X
    fS = [len(M1) / N, len(M2) / N, len(S) / N]
    fK1 = [len(M1) / N, 1 - len(M2) / N, len(K1) / N]
    fK2 = [1 - len(M1) / N, len(M2) / N, len(K2) / N]
    fO = [1 - len(M1) / N, 1 - len(M2) / N, len(O) / N]

    for j, f in enumerate([fS, fK1, fK2, fO]):
        print("Frequency of " + labels[j] + ":", round(f[2], 2))

    LHSs = [None] * 4
    for i, f in enumerate([fS, fK1, fK2, fO]):
        sig_11 = f[0] * (1 - f[0])
        sig_22 = f[1] * (1 - f[1])
        sig_ss = f[2] * (1 - f[2])
        sig_1s = f[2] * (1 - f[0])
        sig_2s = f[2] * (1 - f[1])
        sig_12 = f[2] - f[0] * f[1]
        LHS1 = sig_12 * sig_2s - sig_22 * sig_1s  # coefficient of E(x1z)
        LHS2 = sig_12 * sig_1s - sig_11 * sig_2s  # coefficient of E(x2z)
        LHS3 = sig_11 * sig_22 - sig_12**2  # coefficient of E(sz)
        det = (
                sig_11 * sig_22 * sig_ss
                + 2 * sig_12 * sig_2s * sig_1s
                - sig_1s**2 * sig_22
                - sig_12**2 * sig_ss
                - sig_11 * sig_2s**2
        )
        if i==0:
            print("det(T):", round(det, 6))
        factor = np.sqrt(det * (sig_11 * sig_22 - sig_12**2))
        LHSs[i] = (LHS1/factor, LHS2/factor, LHS3/factor)
        print("V_1, V_2, V_3 for target set " + labels[i] + ":", [round(v, 2) for v in LHSs[i]])

    if len(M1) + len(M2) - len(S) + len(O) != N:
        sys.exit("Wrong clade sizes.")

    if len(M3) > 0:
        print("r^2 between target sets and M3:")
        r2_M3_target_sets = [
            cladecalcs.R2(S, M3, N),
            cladecalcs.R2(K1, M3, N),
            cladecalcs.R2(K2, M3, N),
            cladecalcs.R2(O, M3, N),
        ]
        print(r2_M3_target_sets)

    print("-"*100)
    maxFM3 = None
    if RHS is not None:
        print("RHS from input parameters:", round(RHS, 2))
    if len(M3) > 0:
        FM31 = cladecalcs.check_significance(fS, M1, M2, S, M3, N, LHSs[0])
        FM32 = cladecalcs.check_significance(fK1, M1, M2n, K1, M3, N, LHSs[1])
        FM33 = cladecalcs.check_significance(fK2, M1n, M2, K2, M3, N, LHSs[2])
        FM34 = cladecalcs.check_significance(fO, M1n, M2n, O, M3, N, LHSs[3])
        maxFM3 = max(FM31, FM32, FM33, FM34)
        print("RHS from known SNP3:",  maxFM3)
    print("RHS from SNP1 (should be ~0):", round(cladecalcs.check_significance(fS, M1, M2, S, M1, N, LHSs[0]), 2))
    print("RHS from SNP2 (should be ~0):", round(cladecalcs.check_significance(fS, M1, M2, S, M2, N, LHSs[0]), 2))
    FS = cladecalcs.check_significance(fS, M1, M2, S, S, N, LHSs[0])
    FK1 = cladecalcs.check_significance(fK1, M1, M2n, K1, K1, N, LHSs[1])
    FK2 = cladecalcs.check_significance(fK2, M1n, M2, K2, K2, N, LHSs[2])
    FO = cladecalcs.check_significance(fO, M1n, M2n, O, O, N, LHSs[3])
    print("RHS from interaction:", round(FS, 2))
    RHS = FS * 0.8
    print("Updating RHS to", round(RHS, 2))
    print("-"*100)

    # ===========================================================================================
    # Clade calculations
    # ===========================================================================================

    species = stdpopsim.get_species("HomSap")
    if chrA == chrB:
        contigA = species.get_contig(
            chromosome="chr" + str(chrA), genetic_map="HapMapII_GRCh37"
        )
        rec_mapA = contigA.recombination_map
        s1, _ = cladecalcs.get_searchrange(posA, rec_mapA, d)
        _, s2 = cladecalcs.get_searchrange(posB, rec_mapA, d)
        searchrangeA = (s1, s2)
    else:
        contigA = species.get_contig(
            chromosome="chr" + str(chrA), genetic_map="HapMapII_GRCh37"
        )
        contigB = species.get_contig(
            chromosome="chr" + str(chrB), genetic_map="HapMapII_GRCh37"
        )
        rec_mapA = contigA.recombination_map
        rec_mapB = contigB.recombination_map
        searchrangeA = cladecalcs.get_searchrange(posA, rec_mapA, d)
        searchrangeB = cladecalcs.get_searchrange(posB, rec_mapB, d)

    print("Loading clades...")
    clades1, sl1 = cladecalcs.read_from_file(output_dir + "/clades_chr" + str(chrA), searchrangeA[0], searchrangeA[1])
    if chrA == chrB:
        clades2 = clades1
        searchrangeA = (searchrangeA[0], min(searchrangeA[1], sl1))
        sl = [sl1]
    else:
        clades2, sl2 = cladecalcs.read_from_file(output_dir + "/clades_chr" + str(chrB), searchrangeB[0], searchrangeB[1],)
        searchrangeA = (searchrangeA[0], min(searchrangeA[1], sl1))
        searchrangeB = (searchrangeB[0], min(searchrangeB[1], sl2))
        sl = [sl1, sl2]

    print("Setting up...")
    if chrA == chrB:
        resultsA = np.zeros((4, clades1.num), dtype=int)
        xxA = np.zeros((4, searchrangeA[1] - searchrangeA[0] + 1), dtype=int)
        L = [(ts1, clades1, resultsA, xxA, chrA, posA, searchrangeA)]
        dist_bp = posB - posA
        if posA > rec_mapA.sequence_length or posB > rec_mapA.sequence_length:
            dist_cM = "NA"
        else:
            dist_cM = abs(
                (rec_mapA.get_cumulative_mass(posB) - rec_mapA.get_cumulative_mass(posA)) * 100
            )
    else:
        resultsA = np.zeros((4, clades1.num), dtype=int)
        resultsB = np.zeros((4, clades2.num), dtype=int)
        xxA = np.zeros((4, searchrangeA[1] - searchrangeA[0] + 1), dtype=int)
        xxB = np.zeros((4, searchrangeB[1] - searchrangeB[0] + 1), dtype=int)
        L = [
            (ts1, clades1, resultsA, xxA, chrA, posA, searchrangeA),
            (ts2, clades2, resultsB, xxB, chrB, posB, searchrangeB),
        ]
        dist_bp = "NA"
        dist_cM = "NA"

    print("="*100)
    mutsR = defaultdict(set)  # Significant SNPs
    cladesR = defaultdict(set)  # Significant clades
    R_clade_max = [0] * 4  # Highest observed LHS for each clade
    r2_clade_max = [0] * 4  # corresponding r^2
    R_mut_max = [0] * 4  # Highest observed LHS for each SNP
    R_mut_max_pos = ["NA"] * 4
    R_mut_max_chr = ["NA"] * 4
    r2_mut_max = [0] * 4  # corresponding r^2
    for ts, clades, results, xx, ch, pos, searchrange in L:
        print("Search range:", searchrange, searchrange[1] - searchrange[0] + 1)
        with tqdm.tqdm(total=clades.num, desc="Finding significant clades",
                       bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
            for i in range(clades.num):
                if clades.start[i] < searchrange[1] and clades.end[i] > searchrange[0]:
                    b = bin(int(clades.binid[i]))
                    Y = {j for j, digit in enumerate(reversed(b)) if digit == "1"}
                    r = [0] * 4
                    R = [False] * 4
                    for j, (MM1, MM2, X, f) in enumerate(
                            [
                                (M1, M2, S, fS),
                                (M1, M2n, K1, fK1),
                                (M1n, M2, K2, fK2),
                                (M1n, M2n, O, fO),
                            ]
                    ):
                        LHS = LHSs[j]
                        r[j] = cladecalcs.R2(X, Y, N)
                        R[j] = cladecalcs.check_significance(f, MM1, MM2, X, Y, N, LHS)
                    for j in range(4):
                        if R[j] >= RHS and r[j] == np.max(r):
                            # Record to the target set that this clade is most highly correlated with
                            cladesR[j].add(
                                (
                                    clades.start[i],
                                    clades.end[i],
                                    clades.nodeid[i],
                                    clades.num_mutations[i],
                                )
                            )
                            # Same for the SNPs
                            mutsR[j].update(clades.mutations[i])
                        if r[j] == np.max(r):
                            if R[j] > R_clade_max[j]:
                                # Just update the highest observed LHS
                                R_clade_max[j] = R[j]
                                r2_clade_max[j] = r[j]
                            if R[j] > R_mut_max[j] and clades.num_mutations[i] > 0:
                                R_mut_max[j] = R[j]
                                R_mut_max_pos[j] = ",".join(
                                    str(int(m)) for m in clades.mutations[i]
                                )
                                R_mut_max_chr[j] = str(ch)
                                r2_mut_max[j] = r[j]
                pbar.update(1)

    print("-" * 100)
    print("Highest clade r^2 with a target set:", round(np.max(r2_clade_max), 2))
    print("Number of clades and mutations found above threshold:")
    for j in range(4):
        print(labels[j], ": clades ", len(cladesR[j]), ", mutations ", len(mutsR[j]))
    print("Highest clade LHS with a target set:", round(np.max(R_clade_max), 2))
    print("Highest SNP r^2 with a target set:", round(np.max(r2_mut_max), 2))
    print("Highest SNP LHS with a target set:", round(np.max(R_mut_max), 2))

    for ts, clades, results, xx, ch, pos, searchrange in L:
        with tqdm.tqdm(total=clades.num, desc="Finding uncovered regions",
                       bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
            for i in range(clades.num):
                if clades.start[i] < searchrange[1] and clades.end[i] > searchrange[0]:
                    b = bin(int(clades.binid[i]))
                    Y = {j for j, digit in enumerate(reversed(b)) if digit == "1"}
                    for j, (MM1, MM2, X, f) in enumerate(
                            [
                                (M1, M2, S, fS),
                                (M1, M2n, K1, fK1),
                                (M1n, M2, K2, fK2),
                                (M1n, M2n, O, fO),
                            ]
                    ):
                        LHS = LHSs[j]
                        if cladecalcs.check_conditions(f, X, Y, MM1, MM2, LHS, RHS, N):
                            results[j][i] = 1
                            for k in range(
                                    int(max(searchrange[0], clades.start[i]) - searchrange[0]),
                                    int(min(searchrange[1], clades.end[i]) - searchrange[0])
                                    + 1,
                            ):
                                xx[j][k] += 1
                pbar.update(1)

    print("="*100)
    if plot_on:
        print("Plotting...")

    # ===========================================================================================
    # Plot
    # ===========================================================================================

    outname = (
            output_dir
            + "/"
            + str(chrA)
            + "_"
            + str(chrB)
            + "_"
            + str(posA)
            + "_"
            + str(posB)
            + "_results"
    )

    # Get max y limit
    # Get total bp where number of clades is 0
    ymax_ = 10
    ff_list = {}
    for j, (_, _, _, xx, _, _, _) in enumerate(L):
        for i in range(4):
            ymax_ = max(ymax_, max(xx[i]))
            ff = len([k for k in range(len(xx[i])) if xx[i][k] == 0])
            ff_list[(i, j)] = ff
    ystep = ymax_ / 4

    if plot_on:
        if chrA == chrB:
            fig, axs = plt.subplots(4, figsize=(4.5, 8), squeeze=False)
        else:
            fig, axs = plt.subplots(4, 2, figsize=(9, 8), squeeze=False)

    if plot_on:
        for j, (ts, clades, results, xx, ch, pos, searchrange) in enumerate(L):
            yy = [0, 0, 0, 0]
            xplot = np.array([m for m in range(searchrange[0], searchrange[1] + 1)])

            # plot everything
            for i in range(4):
                axs[i][j].plot(xplot, xx[i], color="black", lw=1)
                axs[i][j].scatter(
                    [xplot[k] for k in range(len(xplot)) if xx[i][k] == 0],
                    [-2 * ystep / 5 for k in range(len(xplot)) if xx[i][k] == 0],
                    color=col_purp,
                    marker="|",
                    s=12,
                )
                if i in cladesR:
                    for x1, x2, n, nm in cladesR[i]:
                        axs[i][j].scatter(
                            x=[x1, x2, (x2 + x1) / 2],
                            y=[-3 * ystep / 5] * 3,
                            color=col_blue,
                            marker="|",
                            s=12,
                        )
                if i in mutsR:
                    axs[i][j].scatter(
                        [j for j in mutsR[i]],
                        [-4 * ystep / 5 for j in mutsR[i]],
                        color=col_green,
                        marker="x",
                        s=12,
                    )
                if posC != -1:
                    axs[i][j].scatter(
                        [posC],
                        [-ystep],
                        color=col_red,
                        marker="x",
                        s=12,
                    )
                axs[i][j].set_xlim([searchrange[0], searchrange[1]])
                axs[i][j].hlines(
                    y=0, xmin=searchrange[0], xmax=searchrange[1], lw=0.5, color="black"
                )
                # props = dict(facecolor="white", alpha=1.0, edgecolor="none", pad=0)
                axs[i][j].text(
                    searchrange[0] + 10000,
                    ymax_,
                    "uncovered: " + str(int(ff_list[(i, j)])) + "bp",
                    ha="left",
                    va="top",
                    color=col_purp,
                    # transform=axs[i][j].transAxes,
                )
                if len(M3) > 0:
                    if chrC == ch:
                        axs[i][j].text(
                            searchrange[1] - 10000,
                            ymax_,
                            "r2(SNP3, "
                            + labels[i]
                            + ") = "
                            + str(round(r2_M3_target_sets[i], 2)),
                            ha="right",
                            va="top",
                            color=col_red,
                            # transform=axs[i][j].transAxes,
                        )
                axs[i][j].scatter(
                    x=pos,
                    y=-ystep / 5,
                    color="black",
                    s=12,
                )

                axs[i][j].set_ylim([-ystep * 1.15, ymax_ * 1.05])
                ticks = [tick for tick in axs[i][j].get_yticks() if tick >= 0]
                labs = [str(tick) for tick in axs[i][j].get_yticks() if tick >= 0]
                axs[i][j].set_yticks(
                    ticks=ticks,
                    labels=labs,
                )
                axs[i][j].set_ylim([-ystep * 1.15, ymax_ * 1.05])

            # x axes
            for i in range(3):
                axs[i][j].axes.get_xaxis().set_visible(False)
            length = sl[j]
            ticks = [0.2 * i * 1e6 for i in range(1, 6 * int(length / 1e6))]
            labs = [str(round(0.2 * i, 1)) for i in range(1, 6 * int(length / 1e6))]
            axs[3][j].set_xticks(
                ticks=ticks,
                labels=labs,
            )
            axs[3][j].set_xlim([searchrange[0], searchrange[1]])

            axs[0][j].set_title("S (freq = " + str(round(len(S) / N, 2)) + ")")
            axs[1][j].set_title("K1 (freq = " + str(round(len(K1) / N, 2)) + ")")
            axs[2][j].set_title("K2 (freq = " + str(round(len(K2) / N, 2)) + ")")
            axs[3][j].set_title("O (freq = " + str(round(len(O) / N, 2)) + ")")
            axs[3][j].set_xlabel("Genome position (chr" + str(ch) + ", GRCh37), Mbp")

        if chrA == chrB:
            for i in range(4):
                axs[i][0].scatter(
                    x=[posB],
                    y=[-ystep / 5],
                    color="black",
                    s=12,
                )

    if plot_on:
        plt.tight_layout()
        plt.savefig(
            outname + ".png",
            bbox_inches="tight",
            dpi=300,
            )

    # ===========================================================================================
    # Output info into .csv file
    # ===========================================================================================

    print("Writing output to file...")

    if chrC == -1:
        chrC = "NA"
        posC = "NA"
    coverage = ["NA", "NA"]
    coverage_p = ["NA", "NA"]
    for j, (ts, clades, results, xx, ch, pos, searchrange) in enumerate(L):
        cc = 0
        for k in range(len(xx[0])):
            if min(xx[i][k] for i in range(4)) == 0:
                cc += 1
        coverage[j] = cc
        coverage_p[j] = cc / (searchrange[1] - searchrange[0] + 1)

    with open(outname + ".txt", "w") as file:
        # for j in range(4):
        #     file.write("r2_threshold_" + labels[j] + " " + str(r_thresh) + "\n")
        file.write("b " + str(b_est) + "\n")
        file.write("h2 " + str(h2_est) + "\n")
        file.write("alpha " + str(alpha) + "\n")
        file.write("n " + str(n_reg) + "\n")
        file.write("F " + str(RHS) + "\n")
        file.write("F_SNP3 " + str(maxFM3) + "\n")
        file.write("F_interaction " + str(FS) + "\n")
        file.write("SNP1_chr " + str(chrA) + "\n")
        file.write("SNP1_pos " + str(posA) + "\n")
        file.write("SNP1_freq " + str(len(M1) / N) + "\n")
        file.write("SNP2_chr " + str(chrB) + "\n")
        file.write("SNP2_pos " + str(posB) + "\n")
        file.write("SNP2_freq " + str(len(M2) / N) + "\n")
        file.write("SNP1_SNP2_dist_bp " + str(dist_bp) + "\n")
        file.write("SNP1_SNP2_dist_cM " + str(dist_cM) + "\n")
        file.write("SNP3_chr " + str(chrC) + "\n")
        file.write("SNP3_pos " + str(posC) + "\n")
        if len(M3) != 0:
            file.write("SNP3_freq " + str(len(M3) / N) + "\n")
        else:
            file.write("SNP3_freq NA\n")
        file.write("S_freq " + str(len(S) / N) + "\n")
        file.write("K1_freq " + str(len(K1) / N) + "\n")
        file.write("K2_freq " + str(len(K2) / N) + "\n")
        file.write("O_freq " + str(len(O) / N) + "\n")
        file.write("r2_M1_M2 " + str(cladecalcs.R2(M1, M2, N)) + "\n")
        file.write("r2_S_M1 " + str(cladecalcs.R2(S, M1, N)) + "\n")
        file.write("r2_K1_M1 " + str(cladecalcs.R2(K1, M1, N)) + "\n")
        file.write("r2_K2_M1 " + str(cladecalcs.R2(K2, M1, N)) + "\n")
        file.write("r2_O_M1 " + str(cladecalcs.R2(O, M1, N)) + "\n")
        file.write("r2_S_M2 " + str(cladecalcs.R2(S, M2, N)) + "\n")
        file.write("r2_K1_M2 " + str(cladecalcs.R2(K1, M2, N)) + "\n")
        file.write("r2_K2_M2 " + str(cladecalcs.R2(K2, M2, N)) + "\n")
        file.write("r2_O_M2 " + str(cladecalcs.R2(O, M2, N)) + "\n")
        if len(M3) == 0:
            file.write("r2_S_M3 NA\n")
            file.write("r2_K1_M3 NA\n")
            file.write("r2_K2_M3 NA\n")
            file.write("r2_O_M3 NA\n")
        else:
            file.write("r2_S_M3 " + str(cladecalcs.R2(S, M3, N)) + "\n")
            file.write("r2_K1_M3 " + str(cladecalcs.R2(K1, M3, N)) + "\n")
            file.write("r2_K2_M3 " + str(cladecalcs.R2(K2, M3, N)) + "\n")
            file.write("r2_O_M3 " + str(cladecalcs.R2(O, M3, N)) + "\n")
        for j, (ts, clades, results, xx, ch, pos, searchrange) in enumerate(L):
            for i in range(4):
                file.write(
                    "purple_bp_ch"
                    + str(j)
                    + "_"
                    + labels[i]
                    + " "
                    + str(ff_list[(i, j)])
                    + "\n"
                )
                file.write(
                    "purple_prop_ch"
                    + str(j)
                    + "_"
                    + labels[i]
                    + " "
                    + str(ff_list[(i, j)] / (searchrange[1] - searchrange[0] + 1))
                    + "\n"
                )
            file.write("purple_bp_ch" + str(j) + " " + str(coverage[j]) + "\n")
            file.write("purple_prop_ch" + str(j) + " " + str(coverage_p[j]) + "\n")
            file.write(
                "searchrange_bp_ch"
                + str(j)
                + " "
                + str(searchrange[1] - searchrange[0] + 1)
                + "\n"
            )
        if chrA == chrB:
            for i in range(4):
                file.write("purple_bp_ch1_" + labels[i] + " NA\n")
                file.write("purple_prop_ch1_" + labels[i] + " NA\n")
            file.write("purple_bp_ch1  NA\n")
            file.write("purple_prop_ch1 NA\n")
            file.write("searchrange_bp_ch1 NA\n")
        for i in range(4):
            file.write("blue_n_" + labels[i] + " " + str(len(cladesR[i])) + "\n")
            supported = 0
            for start, end, n, nm in cladesR[i]:
                if nm > 0:
                    supported += 1
            file.write("blue_n_" + labels[i] + "_supported " + str(supported) + "\n")
            file.write("SNPs_n_" + labels[i] + " " + str(len(mutsR[i])) + "\n")
        for i in range(4):
            file.write("max_clade_LHS_" + labels[i] + " " + str(R_clade_max[i]) + "\n")
            file.write("max_clade_r2_" + labels[i] + " " + str(r2_clade_max[i]) + "\n")
            file.write("max_SNP_LHS_" + labels[i] + " " + str(R_mut_max[i]) + "\n")
            file.write("max_SNP_r2_" + labels[i] + " " + str(r2_mut_max[i]) + "\n")
            file.write("max_SNP_chr_" + labels[i] + " " + R_mut_max_chr[i] + "\n")
            file.write("max_SNP_pos_" + labels[i] + " " + R_mut_max_pos[i] + "\n")

    print("="*100)

if __name__ == "__main__":
    main(sys.argv[1:])
