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
import argparse
import gc
import csv
from matplotlib import pyplot as plt

import cladecalcs
import utils

col_green = "#228833"
col_red = "#EE6677"
col_purp = "#AA3377"
col_blue = "#66CCEE"
col_yellow = "#CCBB44"
col_indigo = "#4477AA"
col_grey = "#BBBBBB"


def main(argv):
    parser = argparse.ArgumentParser(
        description="SPECTRE: A tool for identifying causal variants for pairwise epistatic interactions.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example usage:\n"
               "python -m spectre -f /path/to/chr20.trees -C 20,20 -P 15489683,15620278 -O /path/to/output\n"
               "python -m spectre -f /path/to/chr20.trees,/path/to/chr22.trees -C 20,22 -P 15489683,10056 -O /path/to/output"
    )

    parser.add_argument("-O", "--output_dir", required=True, type=str,
                        help="Path to the directory where output files will be saved.")
    parser.add_argument("-C", "--chromosomes", required=True, type=str,
                        help="Comma-separated list of chromosome numbers for each SNP (e.g., '20,20,22').")
    parser.add_argument("-P", "--positions", required=True, type=str,
                        help="Comma-separated list of base-pair positions for each SNP (e.g., '15489683,15620278').")
    parser.add_argument("-f", "--treefiles", type=str,
                        help="Optional: Comma-separated list of paths to tree-sequence files (in .ts or .tsz format).\n"
                             "The number of files must match the number of unique chromosomes.")
    parser.add_argument("-c", "--chunknames", type=str,
                        help="Optional: Comma-separated list of paths to files containing chunk names.\n"
                             "Used to skip the tree splitting step. The number of files must match\n"
                             "the number of unique chromosomes. One of --treefiles and --chunknames must be provided.")
    parser.add_argument("-t", "--treeinfo", type=str,
                        help="Optional: Comma-separated list of paths to treeinfo files.\n"
                             "The number of files must match the number of unique chromosomes.\n"
                             "Must be provided if --chunknames is used.")
    parser.add_argument("-A", "--alpha", type=float, default=0.05,
                        help="Optional: Significance threshold for the interaction test (default: 0.05).")
    parser.add_argument("-S", "--teststatistic", type=float,
                        help="Optional: Test statistic for the interaction test. If provided, p-value is ignored.")
    parser.add_argument("-p", "--pvalue", type=float,
                        help="Optional: p-value for the interaction test. Converted to a test statistic if --teststatistic is not set.")
    parser.add_argument("-e", "--maxeffectsize", type=float, default=1.0,
                        help="Optional: Upper bound on the plausible effect size for the trait (default: 1.0).")
    parser.add_argument("-E", "--effectsize", type=float,
                        help="Optional: Estimated effect size for the interaction.")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Optional: Generate a plot of the results (default: True).")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Optional: Overwrite existing results if they exist (default: False).")

    args = parser.parse_args(argv)

    # Process and validate args
    chromosomes = [int(c.strip()) for c in args.chromosomes.split(',')]
    positions = [int(p.strip()) for p in args.positions.split(',')]
    tree_files = [f.strip() for f in args.treefiles.split(',')] if args.treefiles else []
    chunknames_files = [c.strip() for c in args.chunknames.split(',')] if args.chunknames else []
    treeinfo_files = [c.strip() for c in args.treeinfo.split(',')] if args.treeinfo else []

    if len(chromosomes) < 2 or len(chromosomes) > 3:
        sys.exit("Error: Please provide 2 or 3 SNPs using the -C and -P arguments.")
    if len(chromosomes) != len(positions):
        sys.exit("Error: The number of chromosomes and positions must be the same.")

    unique_chroms = []
    for chrom in chromosomes:
        if chrom not in unique_chroms:
            unique_chroms.append(chrom)
    if len(chunknames_files) != len(unique_chroms) and len(tree_files) != len(unique_chroms):
        sys.exit(
            f"Error: Expected {len(unique_chroms)} tree file(s) or {len(chunknames_files)} chunkname file(s), but got {len(tree_files)}.")
    if chunknames_files and len(treeinfo_files) != len(unique_chroms):
        sys.exit(
            f"Error: Expected {len(unique_chroms)} treeinfo file(s) for the specified unique chromosomes, but got {len(treeinfo_files)}.")
    if chunknames_files and len(chunknames_files) != len(unique_chroms):
        sys.exit(
            f"Error: Expected {len(unique_chroms)} chunknames file(s) for the specified unique chromosomes, but got {len(chunknames_files)}.")

    chrA, posA = chromosomes[0], positions[0]
    chrB, posB = chromosomes[1], positions[1]
    chrC, posC = (chromosomes[2], positions[2]) if len(chromosomes) > 2 else (-1, -1)

    alpha = args.alpha
    teststat = args.teststatistic
    effectsize = args.effectsize
    if teststat is None and args.pvalue is not None:
        # Convert p-value to a two-tailed Z-score (test statistic)
        teststat = scipy.stats.norm.isf(args.pvalue / 2)

    maxeffectsize = args.maxeffectsize
    plot_on = args.plot
    overwrite = args.overwrite
    output_dir = args.output_dir + "/" + str(chrA) + "_" + str(posA) + "_" + str(chrB) + "_" + str(posB)

    # Setup
    P_range = [0.5, 0.8, 0.9, 0.95]
    C = scipy.stats.norm.ppf(1 - alpha / 2)
    D = scipy.stats.norm.ppf(1 - alpha / 2)
    d = 1.0  # cM to search around each SNP

    if not os.path.exists(output_dir):
        print(f"Creating results directory: {output_dir}")
        os.makedirs(output_dir)

    tree_file_map = {chrom: file for chrom, file in zip(unique_chroms, tree_files)} if tree_files else {}
    chunknames_map = {chrom: file for chrom, file in zip(unique_chroms, chunknames_files)} if chunknames_files else {}
    treeinfo_map = {chrom: file for chrom, file in zip(unique_chroms, treeinfo_files)} if treeinfo_files else {chrom: output_dir + "/treeinfo_chr" + str(chrom) + ".txt" for chrom in chromosomes}

    print("=" * 100)
    print("Inputs:")
    print(f"SNP1: chr{chrA}:{posA}")
    print(f"SNP2: chr{chrB}:{posB}")
    if chrC != -1:
        print(f"SNP3: chr{chrC}:{posC}")
    print(f"Alpha from interaction test: {alpha}")
    if effectsize is not None:
        print(f"Effect size from interaction test: {effectsize}")
    print(f"Test statistic from interaction test: {teststat}")
    print(f"Max effect size: {maxeffectsize}")
    print(f"Output directory: {output_dir}")
    print("-" * 100)

    # ===========================================================================================
    # Getting search range
    # ===========================================================================================

    species = stdpopsim.get_species("HomSap")
    if chrA == chrB:
        contigA = species.get_contig(
            chromosome=f"chr{chrA}", genetic_map="HapMapII_GRCh37"
        )
        rec_mapA = contigA.recombination_map
        s1, _ = cladecalcs.get_searchrange(posA, rec_mapA, d)
        _, s2 = cladecalcs.get_searchrange(posB, rec_mapA, d)
        searchrangeA = (s1, s2)
        searchrangeB = searchrangeA  # Placeholder
    else:
        contigA = species.get_contig(
            chromosome=f"chr{chrA}", genetic_map="HapMapII_GRCh37"
        )
        contigB = species.get_contig(
            chromosome=f"chr{chrB}", genetic_map="HapMapII_GRCh37"
        )
        rec_mapA = contigA.recombination_map
        rec_mapB = contigB.recombination_map
        searchrangeA = cladecalcs.get_searchrange(posA, rec_mapA, d)
        searchrangeB = cladecalcs.get_searchrange(posB, rec_mapB, d)

    # ===========================================================================================
    # Splitting up tree sequence and calculating clade spans
    # ===========================================================================================

    all_snps_info = [(chrA, posA, searchrangeA)]
    if chrA != chrB:
        all_snps_info.append((chrB, posB, searchrangeB))
    if chrC != -1:
        contigC = species.get_contig(chromosome=f"chr{chrC}", genetic_map="HapMapII_GRCh37")
        searchrangeC = cladecalcs.get_searchrange(posC, contigC.recombination_map, d)
        all_snps_info.append((chrC, posC, searchrangeC))

    processed_chroms = set()
    for chromosome, pos, searchrange in all_snps_info:
        if chromosome not in processed_chroms and chromosome != -1:
            treeinfo_dir = treeinfo_map[chromosome]
            if not chunknames_map:
                print(f"Splitting up trees for chr{chromosome}...")
                filename_trees = tree_file_map[chromosome]
                ts = tszip.decompress(filename_trees) if filename_trees.endswith(".tsz") else tskit.load(filename_trees)

                num_samples = ts.num_samples
                num_trees = ts.num_trees
                bps = ts.breakpoints(as_array=True)
                num_chunks = int(len(bps) / 1000) + 1

                # Define and open the new chunknames file for writing
                chunknames_output_path = os.path.join(output_dir, f"chr{chromosome}_chunknames.txt")
                print(f"Generating chunk list file: {chunknames_output_path}")
                with open(chunknames_output_path, "w") as f_chunknames:
                    for i in range(num_chunks):
                        filename_chunk_base = os.path.join(output_dir, f"trees_chr{chromosome}_chunk{i}.trees")
                        filename_chunk_tsz = filename_chunk_base + ".tsz"

                        if overwrite or not os.path.exists(filename_chunk_tsz):
                            print(f"...creating chunk {i}")
                            left = bps[i * 1000]
                            right = bps[-1] if (i + 1) * 1000 >= len(bps) else bps[(i + 1) * 1000]
                            ts_sub = ts.keep_intervals([[left, right]])
                            tszip.compress(ts_sub, filename_chunk_tsz)
                            del ts_sub
                            gc.collect()
                        else:
                            print(f"...chunk {i} already exists")

                        # Write the full path of the chunk to the list file
                        f_chunknames.write(filename_chunk_tsz + "\n")
                del ts
                gc.collect()

                if overwrite or not os.path.exists(treeinfo_dir):
                    print("Getting tree info...")
                    with open(treeinfo_dir, "w") as file:
                        file.write("chr;chunk;chunk_tree_index;tree_start;tree_end\n")
                        for i in range(num_chunks):
                            filename = os.path.join(output_dir, f"trees_chr{chromosome}_chunk{i}.trees.tsz")
                            ts_chunk = tszip.decompress(filename)
                            for t in ts_chunk.trees():
                                if t.num_roots == 1:
                                    file.write(
                                        f"chr{chromosome};{i};{t.index};{int(t.interval[0])};{int(t.interval[1])}\n")
            else:
                print("Chunk names given, reading info from treeinfo file.")
                num_trees, num_chunks = utils.get_tree_and_chunk_info(treeinfo_dir)

                with open(chunknames_map[chromosome], "r") as f:
                    first_chunk_path = f.readline().strip()
                ts_chunk1 = tszip.decompress(first_chunk_path)
                num_samples = ts_chunk1.num_samples
                print(f"Tree info for chr{chromosome}: {num_trees} trees, {num_chunks} chunks, {num_samples} samples")

            filename = os.path.join(output_dir, f"clades_chr{chromosome}_{searchrange[0]}_{searchrange[1]}")
            if not os.path.exists(filename + ".clades.gz"):
                if not chunknames_map:
                    num_trees, num_chunks = utils.get_tree_and_chunk_info(treeinfo_dir)
                    ts_handles = [os.path.join(output_dir, f"trees_chr{chromosome}_chunk{i}.trees") for i in
                                  range(num_chunks)]
                else:
                    with open(chunknames_map[chromosome], "r") as f:
                        ts_handles = [line.strip() for line in f]
                _ = cladecalcs.clade_span(ts_handles, num_trees, num_samples, start=searchrange[0], end=searchrange[1],
                                          write_to_file=filename, write_to_file_freq=50000)
            processed_chroms.add(chromosome)

    # ===========================================================================================
    # Getting mutation info from tree sequences
    # ===========================================================================================
    print("-" * 100)
    print("Loading tree sequences for specified SNPs...")

    chunkA = cladecalcs.get_chunk(treeinfo_map[chrA], posA)
    chunkB = cladecalcs.get_chunk(treeinfo_map[chrB], posB)
    chunkC_val = cladecalcs.get_chunk(treeinfo_map[chrC], posC) if chrC != -1 else -1

    def load_ts_chunk(chrom, chunk_idx):
        if chunk_idx == -1: return None
        if not chunknames_map:
            path = os.path.join(output_dir, f"trees_chr{chrom}_chunk{chunk_idx}.trees.tsz")
            return tszip.decompress(path)
        else:
            with open(chunknames_map[chrom], 'r') as f:
                path = f.readlines()[chunk_idx].strip()
                print(path)
                return tszip.decompress(path)

    ts1 = load_ts_chunk(chrA, chunkA)
    ts2 = load_ts_chunk(chrB, chunkB)
    ts3 = load_ts_chunk(chrC, chunkC_val) if chrC != -1 else None

    N = ts1.num_samples
    print("Sample size (N):", N)
    print("Getting SNPs and target sets...")

    M1, M2, M3 = set(), set(), set()
    for ts, M, pos, label in [(ts1, M1, posA, "SNP1"), (ts2, M2, posB, "SNP2"), (ts3, M3, posC, "SNP3")]:
        if ts:
            for m in ts.mutations():
                if int(ts.site(m.site).position) == pos:
                    t = ts.at(ts.site(m.site).position)
                    M.update(s for s in t.samples(m.node))
        print(f"Frequency of {label}: {round(len(M) / N, 2)}")
    if len(M1) == 0 or len(M2) == 0:
        sys.exit("Could not find SNPs in tree sequence.")

    S, K1, K2 = {s for s in M1 if s in M2}, {s for s in M1 if s not in M2}, {s for s in M2 if s not in M1}
    O = {s for s in range(N) if (s not in M1 and s not in M2)}
    M1n, M2n = {s for s in range(N) if s not in M1}, {s for s in range(N) if s not in M2}

    print("-" * 100)

    # ===========================================================================================
    # Clade calculations
    # ===========================================================================================

    print("Loading clades...")
    clades1_path = os.path.join(output_dir, f"clades_chr{chrA}_{searchrangeA[0]}_{searchrangeA[1]}")
    clades1, sl1 = cladecalcs.read_from_file(clades1_path, searchrangeA[0], searchrangeA[1])

    if chrA == chrB:
        clades2, sl2 = clades1, sl1
        searchrangeA = (searchrangeA[0], min(searchrangeA[1], sl1))
        sl = [sl1]
    else:
        clades2_path = os.path.join(output_dir, f"clades_chr{chrB}_{searchrangeB[0]}_{searchrangeB[1]}")
        clades2, sl2 = cladecalcs.read_from_file(clades2_path, searchrangeB[0], searchrangeB[1])
        searchrangeA = (searchrangeA[0], min(searchrangeA[1], sl1))
        searchrangeB = (searchrangeB[0], min(searchrangeB[1], sl2))
        sl = [sl1, sl2]

    L = [[ts1, clades1, None, None, None, chrA, posA, searchrangeA]]
    if chrA != chrB:
        L.append([ts2, clades2, None, None, None, chrB, posB, searchrangeB])

    target_configs = [(S, M1, M2), (K1, M1, M2n), (K2, M1n, M2), (O, M1n, M2n)]
    target_vectors = [cladecalcs.set_to_vector(tc[0], N) for tc in target_configs]
    not_target_vectors = [1 - v for v in target_vectors]

    pseudo_inverses, beta_s_sd_all = [], []
    for i, (X_set, MM1_set, MM2_set) in enumerate(target_configs):
        s_std = cladecalcs.standardize(target_vectors[i])
        x1_std = cladecalcs.standardize(cladecalcs.set_to_vector(MM1_set, N))
        x2_std = cladecalcs.standardize(cladecalcs.set_to_vector(MM2_set, N))
        design_matrix = np.column_stack((x1_std, x2_std, s_std))
        pinv_design = np.linalg.pinv(design_matrix)
        pseudo_inverses.append(pinv_design)
        beta_s_sd_all.append(np.sqrt(np.linalg.pinv(design_matrix.T @ design_matrix)[2, 2]))

    all_psi_stacked = np.vstack(pseudo_inverses)
    X_matrix, notX_matrix = np.vstack(target_vectors), np.vstack(not_target_vectors)
    psi_tensor = np.stack(pseudo_inverses)
    print("-" * 100)

    b_clade_min, r2_clade_min = [np.inf] * len(P_range), [0] * len(P_range)
    b_test1_part2_min = np.inf
    b_mut_min, r2_mut_min = [np.inf] * len(P_range), [0] * len(P_range)
    b_mut_min_pos, b_mut_min_chr = ["NA"] * len(P_range), ["NA"] * len(P_range)

    for l, (ts, clades, _, _, _, ch, pos, searchrange) in enumerate(L):
        test1_dir = os.path.join(output_dir, f"{ch}_{pos}_test1.txt")
        test1_part2_dir = os.path.join(output_dir, f"{ch}_{pos}_test1_part2.txt")
        test2_dir = os.path.join(output_dir, f"{ch}_{pos}_test2.txt")

        if not overwrite and os.path.exists(test1_dir) and os.path.exists(test2_dir):
            L[l][2:5] = np.loadtxt(test1_dir), np.loadtxt(test2_dir), np.loadtxt(test1_part2_dir)
            for p, P in enumerate(P_range):
                with open(output_dir + "/" + str(ch) + "_" + str(pos) + "_" + str(P) + "_testinfo.txt", "r") as file:
                    for line in file:
                        line = line.strip().split(" ")
                        b_clade_min[p] = float(line[0])
                        r2_clade_min[p] = float(line[1])
                        b_mut_min[p] = float(line[2])
                        b_mut_min_pos[p] = line[3]
                        if b_mut_min_pos[p] != "NA":
                            b_mut_min_pos[p] = b_mut_min_pos[p].strip().split(",")
                            b_mut_min_pos[p] = [float(a) for a in b_mut_min_pos[p]]
                        b_mut_min_chr[p] = line[4]
                        if b_mut_min_chr[p] != "NA":
                            b_mut_min_chr[p] = float(b_mut_min_chr[p])
                        r2_mut_min[p] = float(line[5])
                        b_test1_part2_min = float(line[6])
        else:
            xx_test1 = np.full((len(P_range), searchrange[1] - searchrange[0] + 1), np.inf)
            xx_test1_part2 = np.full(searchrange[1] - searchrange[0] + 1, np.inf)
            xx_test2 = np.zeros((len(P_range), searchrange[1] - searchrange[0] + 1))

            tqdm_total = sum(1 for i in range(clades.num) if
                             clades.start[i] <= searchrange[1] and clades.end[i] >= searchrange[0] and 1 <
                             clades.cladesize[i] < N - 1)
            with tqdm.tqdm(total=tqdm_total, desc=f"Applying tests to Chr {ch}",
                           bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
                for i in range(clades.num):
                    if not (clades.start[i] <= searchrange[1] and clades.end[i] >= searchrange[0] and 1 <
                            clades.cladesize[i] < N - 1): continue

                    Y = {j for j, digit in enumerate(reversed(bin(int(clades.binid[i])))) if digit == "1"}
                    Y_vec, notY_vec = cladecalcs.set_to_vector(Y, N), 1 - cladecalcs.set_to_vector(Y, N)
                    Y_std_vec, Y_vec_b, notY_vec_b = cladecalcs.standardize(Y_vec), Y_vec[np.newaxis, :], notY_vec[
                                                                                                          np.newaxis, :]

                    beta_partials_t1 = (all_psi_stacked @ Y_std_vec)[[2, 5, 8, 11]]

                    Z_tensor = np.stack([Y_vec_b * X_matrix, notY_vec_b * X_matrix, np.minimum(Y_vec_b + X_matrix, 1),
                                         Y_vec_b * notX_matrix, notY_vec_b * notX_matrix,
                                         Y_vec_b + (notY_vec_b * notX_matrix)], axis=2)
                    Z_tensor_std = cladecalcs.standardize_tensor(Z_tensor, axis=1)
                    beta_partials_t2 = (psi_tensor @ Z_tensor_std)[:, -1, :]

                    for j, target_set in enumerate([S, K1, K2, O]):
                        if len(target_set) != 0:
                            bmin1, bmin1_p2, bmin2 = cladecalcs.run_all_tests(P_range, C, D, teststat, maxeffectsize,
                                                                              beta_s_sd_all[j],
                                                                              beta_partial_t1=beta_partials_t1[j],
                                                                              beta_partials_t2=beta_partials_t2[j])
                            r_clade = cladecalcs.R2(target_set, Y, N)
                            start_idx, end_idx = int(max(searchrange[0], clades.start[i]) - searchrange[0]), int(
                                min(searchrange[1], clades.end[i]) - searchrange[0]) + 1
                            b_test1_part2_min = min(b_test1_part2_min, bmin1_p2)

                            for p, P in enumerate(P_range):
                                if bmin1[p] < b_clade_min[p] or (bmin1[p] == b_clade_min[p] and r_clade > r2_clade_min[p]):
                                    b_clade_min[p], r2_clade_min[p] = bmin1[p], r_clade
                                if clades.num_mutations[i] > 0 and (
                                        bmin1[p] < b_mut_min[p] or (bmin1[p] == b_mut_min[p] and r_clade > r2_mut_min[p])):
                                    b_mut_min[p], b_mut_min_pos[p], b_mut_min_chr[p], r2_mut_min[p] = bmin1[p], [int(m) for m in clades.mutations[i]], str(ch), r_clade
                                if bmin1[p] != np.inf:
                                    xx_test1[p, start_idx:end_idx] = np.minimum(xx_test1[p, start_idx:end_idx], bmin1[p])
                                if bmin2[p] != 0:
                                    xx_test2[p, start_idx:end_idx] = np.maximum(xx_test2[p, start_idx:end_idx], bmin2[p])
                            if bmin1_p2 != np.inf:
                                xx_test1_part2[start_idx:end_idx] = np.minimum(xx_test1_part2[start_idx:end_idx], bmin1_p2)
                    pbar.update(1)

            np.savetxt(test1_dir, xx_test1)
            np.savetxt(test2_dir, xx_test2)
            np.savetxt(test1_part2_dir, xx_test1_part2)
            L[l][2:5] = xx_test1, xx_test2, xx_test1_part2

            for p, P in enumerate(P_range):
                with open(output_dir + "/" + str(ch) + "_" + str(pos) + "_" + str(P) + "_testinfo.txt", "w") as file:
                    if b_mut_min_pos[p] == "NA":
                        b_mut_min_pos_string = "NA"
                    elif len(b_mut_min_pos[p]) == 1:
                        b_mut_min_pos_string = str(b_mut_min_pos[p][0])
                    else:
                        b_mut_min_pos_string = ",".join([str(a) for a in b_mut_min_pos[p]])
                    file.write(
                        str(b_clade_min[p])
                        + " "
                        + str(r2_clade_min[p])
                        + " "
                        + str(b_mut_min[p])
                        + " "
                        + b_mut_min_pos_string
                        + " "
                        + str(b_mut_min_chr[p])
                        + " "
                        + str(r2_mut_min[p])
                        + " "
                        + str(b_test1_part2_min)
                        + "\n"
                    )

    # ===========================================================================================
    # Plot and outputs
    # ===========================================================================================

    outname = os.path.join(output_dir, f"{chrA}_{chrB}_{posA}_{posB}_results")

    print("-" * 100)
    print("===== Detailed Output Summary =====")
    print("Writing detailed results to .csv file...")

    # Define the path for the output CSV file
    csv_output_path = os.path.join(output_dir, outname + ".csv")

    # Use a 'with' statement to handle the file safely
    with open(csv_output_path, 'w', newline='') as f:
        # Create a csv writer object
        csv_writer = csv.writer(f)

        # Write the header row
        csv_writer.writerow(['Variable', 'Value'])

        def write_output(description, variable_name, value):
            """Helper function to print to console and write to CSV."""
            # Format value for printing and writing
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = value

            print(f"  - {description}: {formatted_value}")
            csv_writer.writerow([variable_name, formatted_value])

        # --- Input Parameters & Setup ---
        print("\n## Input Parameters & Setup")

        write_output("Interacting SNP 1", "snp1", f"chr{chrA}:{posA}")
        write_output("Interacting SNP 2", "snp2", f"chr{chrB}:{posB}")
        if chrC != -1:
            write_output("Third SNP", "snp3", f"chr{chrC}:{posC}")

        write_output("Sample size (N)", "sample_size_N", N)
        write_output("Significance level (alpha)", "alpha", alpha)
        if teststat is not None:
            write_output("Interaction test statistic", "interaction_test_statistic", teststat)
        if effectsize is not None:
            write_output("Interaction effect size", "interaction_effect_size", effectsize)
        write_output("Maximum plausible effect size (b_max)", "max_effect_size", maxeffectsize)
        write_output("Search distance (cM)", "search_distance_cM", d)
        write_output(f"Genomic search range start (chr{chrA})", f"searchrange_chr{chrA}_start", searchrangeA[0])
        write_output(f"Genomic search range end (chr{chrA})", f"searchrange_chr{chrA}_end", searchrangeA[1])
        if chrA != chrB:
            write_output(f"Genomic search range start (chr{chrB})", f"searchrange_chr{chrB}_start", searchrangeB[0])
            write_output(f"Genomic search range end (chr{chrB})", f"searchrange_chr{chrB}_end", searchrangeB[1])

        # --- Target Set Frequencies ---
        print("\n## Target Set Frequencies")

        write_output("Freq(SNP1)", "freq_snp1", len(M1) / N)
        write_output("Freq(SNP2)", "freq_snp2", len(M2) / N)
        write_output("Freq(S = SNP1 & SNP2)", "freq_S", len(S) / N)
        write_output("Freq(K1 = SNP1 & not SNP2)", "freq_K1", len(K1) / N)
        write_output("Freq(K2 = not SNP1 & SNP2)", "freq_K2", len(K2) / N)
        write_output("Freq(O = not SNP1 & not SNP2)", "freq_O", len(O) / N)

        # --- Test 1A Results ---
        print("\n## Test 1A results: Minimum 'b' required to explain the interaction")
        print("  For a given probability P, this is the minimum effect size a hidden variant would need to have to cause\n  a false-positive interaction.")

        for p, P_val in enumerate(P_range):
            print(f"\n  For P = {P_val}:")

            # Write clade-only results
            desc_clade = f"Smallest 'b' from any clade (P={P_val})"
            var_name_clade = f"test1a_b_clade_min_p{P_val}"
            write_output(desc_clade, var_name_clade, b_clade_min[p])
            write_output(f"  r² for clade (P={P_val})", f"test1a_r2_clade_min_p{P_val}", r2_clade_min[p])

            # Write SNP-supported clade results
            desc_mut = f"Smallest 'b' from a SNP-supported clade (P={P_val})"
            var_name_mut = f"test1a_b_mut_min_p{P_val}"
            write_output(desc_mut, var_name_mut, b_mut_min[p])
            write_output(f"  r² for SNP-supported clade (P={P_val})", f"test1a_r2_mut_min_p{P_val}", r2_mut_min[p])
            write_output(f"  Supporting SNP Chromosome (P={P_val})", f"test1a_mut_chr_p{P_val}", int(b_mut_min_chr[p]))
            write_output(f"  Supporting SNP Position (P={P_val})", f"test1a_mut_pos_p{P_val}", b_mut_min_pos[p])

        # --- Test 1B Results ---
        print("\n## Test 1B results: Robustness to a hidden variant")
        print("  This computes the minimum effect size needed for the given test statistic to become not significant.\n")
        if teststat is not None:
            write_output("Minimum 'b' to explain test statistic", "test1b_b_min", b_test1_part2_min)
        else:
            print("  - Test 1B not run (requires a test statistic).")
            csv_writer.writerow(['test1b_b_min', 'NA'])

        # --- Test 2 Results ---
        print("\n## Test 2 results: Quantifying evidence against phantom epistasis")
        print("  For a given probability P, this identifies regions where phantom epistasis is implausible.")

        if effectsize is not None:
            print(
                f"\n   Mode: Calculating genomic span where Test 2 result 'b' is less than the provided effect size ({abs(effectsize)}).")
            total_problematic_span = 0
            for l, (ts, clades, xx_test1, xx_test2, xx_test1_part2, ch, pos_j, searchrange) in enumerate(L):
                min_b_at_each_position = np.min(xx_test2, axis=0)
                problematic_indices = np.where(min_b_at_each_position < abs(effectsize))[0]
                problematic_span_bp = len(problematic_indices)
                total_problematic_span += problematic_span_bp
                write_output(f"Span with 'b' < {abs(effectsize)} (chr{ch})", f"test2_span_lt_effectsize_chr{ch}",
                             problematic_span_bp)

            print(f"  - Total span where 'b' < |effectsize|: {total_problematic_span:,} bp")
            csv_writer.writerow(['test2_total_span_lt_effectsize', total_problematic_span])

        print("\n   Mode: Finding the overall minimum 'b' from Test 2.")
        overall_min_b = np.inf
        for l, (ts, clades, xx_test1, xx_test2, xx_test1_part2, ch, pos_j, searchrange) in enumerate(L):
            min_b_in_region = np.min(xx_test2)
            if min_b_in_region < overall_min_b:
                overall_min_b = min_b_in_region

        write_output("Overall minimum 'b' from Test 2", "test2_overall_min_b", overall_min_b)

    print(f"\nDetailed summary saved to: {csv_output_path}")

    if plot_on:
        print("Plotting...")
        from matplotlib.ticker import MultipleLocator, FuncFormatter

        plot_ymax = maxeffectsize
        ystep = plot_ymax / 5
        num_plot_cols = 2 if chrA != chrB else 1
        fig, axs = plt.subplots(3, num_plot_cols, figsize=(9, 5) if num_plot_cols == 2 else (4.5, 5),
                                squeeze=False, sharex='col', gridspec_kw={'hspace': 0.3})
        plt.subplots_adjust(wspace=0.05)
        colorpal = [col_purp, col_red, col_green, col_blue]

        for j, (ts, clades, xx_test1, xx_test2, xx_test1_part2, ch, pos_j, searchrange) in enumerate(L):
            ax0, ax1, ax2 = axs[0, j], axs[1, j], axs[2, j]
            xplot = np.arange(searchrange[0], searchrange[1] + 1)

            # --- Plot data for each panel ---
            ax0.set_title("Test 1A")
            ax1.set_title("Test 1B")
            ax2.set_title("Test 2")
            for p, P in enumerate(P_range):
                ax0.plot(xplot, np.clip(xx_test1[p], None, plot_ymax), color=colorpal[p], lw=1, label=f"P={P:.2f}")
            if teststat is not None:
                ax1.plot(xplot, np.clip(xx_test1_part2, None, plot_ymax), color="black", lw=1)

            if effectsize is not None:
                ax0.hlines(y=abs(effectsize), xmin=searchrange[0], xmax=searchrange[1], lw=0.5, color="black", linestyle=":")
                ax1.hlines(y=abs(effectsize), xmin=searchrange[0], xmax=searchrange[1], lw=0.5, color="black", linestyle=":")
                ax2.hlines(y=abs(effectsize), xmin=searchrange[0], xmax=searchrange[1], lw=0.5, color="black", linestyle=":")

            if b_mut_min_pos[p] != "NA":
                plot_mut = False
                if effectsize is None:
                    plot_mut = True
                elif max(b_mut_min_pos[p]) < abs(effectsize):
                    plot_mut = True
                if plot_mut:
                    ax0.scatter(
                        b_mut_min_pos[p], [-ystep / 6] * len(b_mut_min_pos[p]),
                        color="black",
                        marker="x",
                        s=12,
                    )

            # Plot Test 2 and its highlights
            test2_overall_min, test2_overall_max = np.min(xx_test2), np.max(xx_test2)
            for p, P in enumerate(P_range):
                ax2.plot(xplot, xx_test2[p], color=colorpal[p], lw=1)

            test2_range = test2_overall_max - test2_overall_min if test2_overall_max > test2_overall_min else 1
            ax2.hlines(y=test2_overall_min - test2_range / 40, xmin=searchrange[0], xmax=searchrange[1], lw=0.5,
                       color="black")
            for p, P in enumerate(P_range):
                y2 = xx_test2[p]
                line_min = np.min(y2)
                inds = np.where(y2 <= line_min * 1.05)[0]
                ax2.scatter(xplot[inds], [test2_overall_min - (p + 1.5) * test2_range / 20] * len(inds),
                            marker="s", s=1, color=colorpal[p])

            # Format axes
            ax0.set_xlim(searchrange)
            ax2.set_xlabel(f"Genome position (chr {ch}), Mbp")
            ax2.xaxis.set_major_locator(MultipleLocator(2e5))
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val / 1e6:.1f}'))

            for i, ax in enumerate([ax0, ax1, ax2]):
                ticks = []
                labels = []
                for k in range(6):
                    if i != 2 or test2_overall_min <= k * ystep <= test2_overall_max:
                        t = k * ystep
                        ticks.append(t)
                        if i != 2 and k == 5:
                            labels.append(r"$\geq$" + f"{int(plot_ymax)}")
                        else:
                            labels.append(f'{t:.1f}')
                if j == 0:
                    ax.set_yticks(ticks=ticks, labels=labels)
                else:
                    ax.set_yticks(ticks=ticks, labels=[""] * len(ticks))

            # Y-axis for top and middle plots
            for ax in [ax0, ax1]:
                ax.set_ylim([-ystep * 1.15, plot_ymax * 1.05])
                ax.hlines(y=0, xmin=searchrange[0], xmax=searchrange[1], lw=0.5, color="black")
                ax.scatter(pos_j, -5.5 * ystep / 6, color="black", s=10)
                if chrA == chrB: ax.scatter(posB if j == 0 else posA, -5.5 * ystep / 6, color="black", s=10)
                if posC != -1 and ch == chrC: ax.scatter(posC, -3.2 * ystep / 6, color="black", marker="^", s=10)

            if test2_overall_min != np.inf and test2_overall_max != np.inf:
                # Y-axis for bottom plot
                ax2.set_ylim(bottom=test2_overall_min - test2_range / 3.5, top=test2_overall_max + test2_range / 10)

            if j == 0:
                ax0.set_ylabel("Effect size (b)")
                ax1.set_ylabel("Effect size (b)")
                ax2.set_ylabel("Effect size (b)")
                leg  = ax0.legend(loc='best', fontsize="small", framealpha=0.8, fancybox=False,
                           labelspacing=0.1, borderaxespad=0.13)
                leg.get_frame().set_linewidth(0.0)

        # plt.tight_layout(pad=1.0, w_pad=1.5)
        plt.savefig(f"{outname}.png", bbox_inches="tight", dpi=300)
        print(f"Plot saved to {outname}.png")

    print("-" * 100)
    print("Spectre run finished.")
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1:])
