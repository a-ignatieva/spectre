### Spectre: Searching for Phantom EpistatiC interactions using TREes
Citation: Ignatieva, A., and Ferreira, L.A.F. Phantom epistasis through the lens of genealogies. bioRxiv, doi.org/

To run the script, first install the necessary packages

    pip install -r requirements.txt
    
You can run the example code (using the provided ARG for 10Mb of human chr20 simulated using stdpopsim and reconstructed using Relate):

    python -m spectre -f example -A 20 -a 15489683 -B 20 -b 15620278
    
Or run

    python -m spectre -h
    
to see the full list of options.

The script produces a plot of the results and a output file with the following entries:

    b: (Input parameter) Estimate of true effect size of the unobserved causal variant
    h2: (Input parameter) Estimate of trait heritability
    alpha: (Input parameter) Significance threshold used in interaction testing
    N: (Input parameter) Number of samples in interaction testing
    F: Threshold used (= RHS of equation (2.3))
    F_SNP3: LHS of equation (2.3) for genotype vector corresponding to SNP3
    F_interaction: LHS of equation (2.3) for genotype vector corresponding to interaction term
    SNP#_chr: Chromosome of SNP#
    SNP#_pos: Position of SNP#
    SNP#_freq: Frequency of SNP# (computed using the ARG)
    SNP1_SNP2_dist_bp: Distance between SNP1 and SNP2 in bp
    SNP1_SNP2_dist_cM: Genetic distance between SNP1 and SNP2 (note: the script uses the HapMapII GRCh37 recombination map)
    #_freq: Frequency of target set #
    r2_#_@: r^2 between target sets # and @
    purple_bp_ch#_@: Total span of uncovered regions for target set @ on chromosome #
    purple_prop_ch#_@: Uncovered regions for target set @ on chromosome #, as proportion of searched genomic span
    purple_bp_ch#: Total span of uncovered regions on chromosome #
    searchrange_bp_ch#: Genomic span searched, in bp
    blue_n_#: Number of significant clades for target set #
    blue_n_#_supported: Number of significant clades for target set # which are supported by at least one SNP
    SNPs_n_#: Number of SNPs that support significant clades for target set #
    max_clade_LHS_#: Maximum observed value of LHS of equation (2.3)
    max_clade_r2_#: Maximum observed r^2 between a clade and set #
    max_SNP_LHS_#: Maximum observed value of LHS of equation (2.3) (considering only clades supported by at least one SNP)
    max_SNP_r2_#: Maximum observed r^2 between a clade and set # (considering only clades supported by at least one SNP)
    max_SNP_chr_#: Chromosome of the SNP with the maximum observed value of LHS of equation (2.3)
    max_SNP_pos_#: Position of the SNP with the maximum observed value of LHS of equation (2.3)
