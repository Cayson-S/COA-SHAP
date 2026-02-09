from typing import Callable, List, Sequence, Tuple
import numpy as np
import itertools
import galois

class COAExplainer():
    def __init__(self, d: int, n: int, val: Callable[[Sequence[int]], float], rng: np.random.Generator = None):
        """
        - d: prime number of players
        - n: sample size; must be multiple of d*(d-1)
        - val: function sets -> value; `sets` is a sequence of player indices (1..d) in the order used by COA
        - args: extra args passed to val(...)
        - rng: optional numpy Generator for reproducible sampling inside onecoa_prime
        This function initilizes the COAExplainer object for estimating Shapley values using the COA method for prime d.
        """
        self.d = d
        self.n = n
        self.val = val
        self.rng = rng or np.random.default_rng()

    # ---------------------------
    # finite field COA (prime-power d = p^r)
    # ---------------------------
    def _all_field_elements(self, p: int, r: int) -> List[Tuple[int]]:
        """Return list of all r-length tuples with entries 0..p-1 in lexicographic order from itertools.product.
        Each tuple represents polynomial coefficients highest->lowest.
        """
        # itertools.product yields tuples where the leftmost element changes slowest; rightmost changes fastest.
        return list(itertools.product(range(p), repeat=r))

    def _coeffs_to_index(self, coeffs: Sequence[int], p: int, r: int) -> int:
        """Map coefficient tuple (highest->lowest) to an index 0..p^r-1 using base-p representation."""
        val = 0
        for c in coeffs:
            val = val * p + int(c)
        return val

    def _index_to_coeffs(self, index: int, p: int, r: int) -> List[int]:
        """Convert 0-based index to coefficient vector of length r (highest->lowest)."""
        coeffs = [0] * r
        for i in range(r - 1, -1, -1):
            coeffs[i] = index % p
            index //= p
        return coeffs

    def onecoa_prime_gen(self):
        """
        Generates the COA one line at a time and yeilds to not save it in memory.
        Returns: a 1xd numpy array (row vector) of each combination of players.
        """
        sample_perm = self.rng.permutation(np.arange(1, self.d))
        firstline = np.concatenate(([0], sample_perm))  # length d
        row = np.empty(self.d, dtype=np.int16)
        
        for j in range(self.d):
            for i in range(1, self.d):
                row = (firstline * i + j) % self.d
                yield row
    
    def onecoa(self, p: int, f_d: Sequence[int], rng: np.random.Generator = None) -> np.ndarray:
        """
        Generate COA for d = p^r (prime-power).
        - p: prime base
        - f_d: coefficients of primitive polynomial on GF(d), highest->lowest, length r+1
            e.g. for x^2 + x + 2 -> [1,1,2]
        Returns array shape (d*(d-1), d) with entries in 1..d.
        """
        
        gf = galois.GF(p)
        # compute r such that p^r == d
        r = 0
        tmp = 1
        
        while tmp < self.d:
            tmp *= p
            r += 1
        if tmp != self.d:
            raise ValueError("d is not a power of p")

        if rng is None:
            rng = np.random.default_rng()

        # Build multiplication (M_d) and addition (A_d) tables on GF(d)
        # r > 1: represent elements as polynomials with coefficients 0..p-1, length r
        entries = self._all_field_elements(p, r)  # list of tuples length d
        # build maps from tuple->index quickly via base-p encoding
        #tuple_to_index = {self._coeffs_to_index(t, p, r): idx for idx, t in enumerate(entries)}
        # pre-generate list of coefficient lists for each element
        entries_list = [list(t) for t in entries]
        
        M_d = np.zeros((self.d, self.d), dtype = np.int16)
        A_d = np.zeros((self.d, self.d), dtype = np.int16)

        for i in range(self.d):
            for j in range(self.d):
                prod = galois.Poly((galois.Poly(gf(entries_list[i])) * galois.Poly(gf(entries_list[j]))).coeffs)

                if not np.any(prod):
                    raise ValueError("Error when attempting to create the COA: The polynomial divisor contains all zeroes")
                
                rem = prod % galois.Poly(gf(f_d))
                # find index of remainder among entries (map to 0..d-1)
                idx = self._coeffs_to_index(rem.coeffs, p, r)
                M_d[i, j] = idx  # store 0-based element value
                # addition: coefficient-wise mod p
                summ = (galois.Poly(gf(entries_list[i])) + galois.Poly(gf(entries_list[j]))).coeffs
                idx2 = self._coeffs_to_index(summ, p, r)
                A_d[i, j] = idx2

        return M_d, A_d

    def est_shcoa(self, *args, p: int = None, f_d: Sequence[int] = None) -> np.ndarray:
        """
        Estimate Shapley values using COA for prime-power d (= p^r)
        - p: prime base
        - f_d: primitive polynomial coefficients (highest->lowest) of degree r (length r+1)
        Returns: 1 x d numpy array (row vector) of estimated Shapley values
        """
        # If prime
        if p is None:
            p = self.d   
        elif f_d is None:
            raise ValueError("f_d must be provided for prime-power d")
        
        # validate inputs
        # coefficients: length should be r+1 where r = log_p(d)
        r = 0
        tmp = 1
        while tmp < self.d:
            tmp *= p
            r += 1

        # validate p prime
        if not galois.is_prime(p):
            raise ValueError("p should be prime")
        
        m = self.d * (self.d - 1)
        if self.n % m != 0:
            raise ValueError("n should be a multiple of d*(d-1)")

        sh = np.zeros(self.d, dtype=float)
        
        # If there is a prime number of players
        if r == 1:
            m = self.d * (self.d - 1)
            if self.n % m != 0:
                raise ValueError("n should be a multiple of d*(d-1)")
                    
            for perml in self.onecoa_prime_gen():
                preC = 0.0
                for i in range(1, self.d + 1):
                    # Convert to a list to save on memory space
                    delta = float(self.val(perml[:i].tolist(), *args)) - preC
                    # add to the Shapley accumulator for the player perml[i-1]
                    sh[int(perml[i-1]) - 1] += delta
                    preC += delta
        else: # Prime multiple number of players
            # check f_d length
            if len(f_d) != r + 1:
                raise ValueError("f_d must have length r+1 (primitive polynomial coefficients)")

            if max(f_d) >= p:
                raise ValueError("coefficients of f_d must be < p")
            
            k = self.n // m

            for _ in range(k):
                # Construction of COA
                M_d, A_d = self.onecoa(p, f_d, rng = self.rng)
                sample_perm = self.rng.permutation(np.arange(1, self.d))
                firstr = np.concatenate(([0], sample_perm))
                
                for i in range(0, self.d):
                    for j in range(1, self.d):
                        perml = np.zeros(self.d, dtype=np.int16)
                        for k in range(0, self.d):
                            preC = 0.0
                            e = M_d[j, int(firstr[k])]
                            perml[k] = A_d[i, e]
                            delta = float(self.val(perml[:k + 1], *args)) - preC
                            sh[int(perml[k])] += delta
                            preC += delta

        sh = sh / float(self.n)
        return sh.reshape(1, -1)


