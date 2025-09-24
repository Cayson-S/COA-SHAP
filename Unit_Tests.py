import COA
import pytest as pt

def fail():
    raise SystemExit(1)

# Unit tests
class TestCOA:
    # Tests the is_prime function
    def test_is_prime(self):
        integers = {-1: False, 1: False, 3:True, 7883:True, 7887:False}
        
        for i, v in integers.items():
            assert COA.is_prime(i) == v


