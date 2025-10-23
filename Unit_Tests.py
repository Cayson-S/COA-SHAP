import COA
import pytest as pt
import numpy as np
import networkx as nx
import timeit as t
np.set_printoptions(threshold=10000)

# Unit tests
class TestCOA:
    # Tests the is_prime function
    def test_is_prime(self):
        integers = {-1: False, 1: False, 3:True, 7883:True, 7887:False}
        
        for i, v in integers.items():
            assert COA.is_prime(i) == v

    def test_onecoa_prime(self):
        with pt.raises(ValueError):
            COA.onecoa_prime(15)
        
        # The function will return a memory error for large primes. Numpy attempts to assign 3 TiB of memory for the prime 7883.
        with pt.raises(MemoryError):
            COA.onecoa_prime(7883)
    
    #def test_est_shcoa_prime(self, val):

def adjacent():
    temp_adjacent = np.zeros((8, 8), dtype=int)
        # 2. Add directed edges (using 0-based indexing in Python)
    #temp_adjacent[0, [1, 2, 4]] = 1
    #temp_adjacent[1, 3] = 1
    #temp_adjacent[2, 4] = 1
    temp_adjacent[0, [5, 6, 7]] = 1
    temp_adjacent[1, [6]] = 1
    temp_adjacent[[3, 5, 6], [7]] = 1

    # 3. Make it symmetric (undirected)
    return temp_adjacent + temp_adjacent.T

def temp_val(sets, adjacent):
    sets = np.array(sets)
    adjacent = np.array(adjacent)
    nsets = len(sets)

    # Early return
    if nsets <= 1:
        return 0

    # Build graph (treat as undirected if symmetric)
    G = nx.from_numpy_array(adjacent)

    # Extract induced subgraph
    subG = G.subgraph(sets)

    # Check connectivity
    connected = nx.is_connected(subG)

    return int(connected)

# [[ 0.4   0.1  -0.1   0.05  0.55]]
# [[ 0.6  0.4  0.  -0.1  0.1]]
#print(COA.est_shcoa_prime(5, 20, temp_val, adjacent()))

print(COA.est_shcoa(8, 112, temp_val, 2, [1, 0, 1, 1], adjacent()))
# Elapsed time with changes:  0.004872542999219149
# Elapsed time without changes:  0.12018242500023917
#print("Elapsed time: ", t.timeit(stmt = "COA.est_shcoa_prime(5, 20, temp_val, adjacent())", setup = "from __main__ import COA, temp_val, adjacent", number = 100)/100)