import COA
import pytest as pt
import numpy as np
np.set_printoptions(threshold=10000)

def fail():
    raise SystemExit(1)

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
            
print(COA.est_shcoa_prime(3, 6, 112))