import COA
import pytest as pt

def fail():
    raise SystemExit(1)

# Unit tests
class TestCOA:
    # Tests the is_prime function
    def test_is_prime(self):
        assert COA.is_prime(-1) == False
        assert COA.is_prime(1) == False
        assert COA.is_prime(3) == True
        assert COA.is_prime(7883) == True
        assert COA.is_prime(7887) == False

