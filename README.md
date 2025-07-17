### Spectre: Searching for Phantom EpistatiC interactions using TREes
Citation: Ignatieva, A., and Ferreira, L.A.F. Phantom epistasis through the lens of genealogies. bioRxiv, doi.org/

To run the script, first install the necessary packages

    pip install -r requirements.txt
    
You can run the example code (using the provided ARG for 10Mb of human chr20 simulated using stdpopsim and reconstructed using Relate):

    python -m spectre \
    -f ~/spectre/example/trees_chr20.trees.tsz \
    -C 20,20 -P 15489683,15620278 --alpha 0.05 --teststatistic 2.5 --effectsize 0.1 \
    -O example
    
Or run

    python -m spectre -h
    
to see the full list of options.

The script produces a plot of the results and a output file in .csv format.
