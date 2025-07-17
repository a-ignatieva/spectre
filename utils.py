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


def get_tree_and_chunk_info(filepath):
    """
    Reads a tree info file to get the total number of trees and chunks.

    Args:
        filepath (str): The path to the input file.

    Returns:
        tuple: A tuple containing (num_trees, num_chunks), or (None, None) if an error occurs.
    """
    num_trees = 0
    last_line = ""
    try:
        with open(filepath, 'r') as f:
            next(f)
            for line in f:
                num_trees += 1
                last_line = line

        if last_line:
            num_chunks = int(last_line.strip().split(';')[1])
            return num_trees, num_chunks + 1
        else:
            sys.exit("Could not get info from treeinfo file")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit("treeinfo file not found")

