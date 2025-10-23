from typing import Callable, List, Sequence, Tuple
import numpy as np
import itertools
import galois

def est_shcoa_prime(d: int, n: int, val: Callable[[Sequence[int]], float], *args, rng: np.random.Generator = None) -> np.ndarray:
    """
    Estimate Shapley values using COA for prime d.

    - d: prime number of players
    - n: sample size; must be multiple of d*(d-1)
    - val: function sets -> value; `sets` is a sequence of player indices (1..d) in the order used by COA
           The R code passes perml[1:i] (1-based indices). Here val receives a Python list of 1-based indices.
    - args: extra args passed to val(...)
    - rng: optional numpy Generator for reproducible sampling inside onecoa_prime

    Returns: 1 x d numpy array (row vector) of estimated Shapley values.
    """
    if not galois.is_prime(d):
        raise ValueError("d should be a prime")

    local_rng = rng or np.random.default_rng()
    
    m = d * (d - 1)
    if n % m != 0:
        raise ValueError("n should be a multiple of d*(d-1)")

    sh = np.zeros(d, dtype=float)  # zero-based for players 0..d-1
    sample_perm = local_rng.permutation(np.arange(1, d))
    firstline = np.concatenate(([0], sample_perm))  # length d
    cz = np.zeros((d - 1, d), dtype=np.int16)
    for i in range(1, d):  # i corresponds to 1:(d-1)
        cz[i - 1, :] = (firstline * i) % d
    
    print(cz)
    
    for j in range(d):  # j = 0:(d-1)
        block = (cz + j) % d
        for perml in block:  # values 1..d
            preC = 0.0
            for i in range(1, d + 1):
                delta = float(val(perml[:i], *args)) - preC
                # add to the Shapley accumulator for the player perml[i-1]
                sh[int(perml[i-1])] += delta
                preC += delta
    
    sh = sh / float(n)
    return  sh.reshape(1, -1)


# ---------------------------
# polynomial operations (coeff vectors highest->lowest)
# ---------------------------

def poly_div(dividend: Sequence[int], divisor: Sequence[int]) -> List[int]:
    """
    Polynomial long division: dividend / divisor, coefficients are given highest-degree first.
    Returns the remainder as a list of coefficients (highest-first).
    Example:
        dividend x^5 + 2 x^3 + 1 -> [1,0,2,0,0,1]
        divisor x^4 + 2 -> [1,0,0,0,2]
    This returns remainder coefficients.
    """
    f1 = list(dividend)
    f2 = list(divisor)
    d1 = len(f1) - 1
    d2 = len(f2) - 1
    if d1 < d2:
        # pad front of f1 so degrees match (R prepends zeros)
        f1 = [0] * (d2 - d1) + f1
        d1 = d2

    dd = d1 - d2
    divisend = f1.copy()
    divisor = f2.copy()
    b = [0] * (dd + 1)

    for i in range(dd + 1):
        # leading coefficient division: divisend[0] / divisor[0]
        # R uses plain arithmetic, not mod; keep float if necessary, but original likely integer arithmetic
        if divisor[0] == 0:
            raise ZeroDivisionError("Divisor leading coefficient is zero")
        b_i = divisend[0] / divisor[0]
        b[i] = b_i
        # subtract b_i * divisor from divisend aligned at front
        for j in range(d2 + 1):
            divisend[j] = divisend[j] - b_i * divisor[j]
        # drop the first element (shift)
        divisend = divisend[1:]

    # divisend is remainder (length d2)
    # If elements are nearly integers, convert to int
    # But we leave them as ints if they are exact ints
    rem = [int(x) if abs(x - round(x)) < 1e-12 else x for x in divisend]
    return rem


def gfpoly_div(f1: Sequence[int], f2: Sequence[int], s: int) -> List[int]:
    """
    Polynomial division over GF(s) (s prime): compute remainder of f1 / f2 with coefficients mod s.
    Input coefficients highest->lowest.
    """
    r = poly_div(f1, f2)
    # reduce modulo s and convert to integers in 0..s-1
    gfr = [int(x) % s for x in r]
    return gfr


def gfpoly_multi(f1: Sequence[int], f2: Sequence[int], s: int) -> List[int]:
    """
    Polynomial multiplication over GF(s). f1, f2 are coefficient lists highest->lowest.
    Returns coefficients (highest->lowest) of product reduced modulo s.
    """
    d1 = len(f1)
    d2 = len(f2)
    hd = d1 + d2 - 1
    # convolution (but highest->lowest). We can align degrees:
    # use indices such that coeff for x^{deg} corresponds to index 0
    res = [0] * hd
    for i in range(d1):
        for j in range(d2):
            res[i + j] += f1[i] * f2[j]
    gft = [int(v) % s for v in res]
    return gft


def gfpoly_add(f1: Sequence[int], f2: Sequence[int], s: int) -> List[int]:
    """
    Polynomial addition over GF(s). Coefficients highest->lowest.
    Pads shorter list on the front (higher degrees) with zeros to match lengths.
    """
    l1 = len(f1)
    l2 = len(f2)
    if l1 < l2:
        f1 = [0] * (l2 - l1) + list(f1)
    elif l2 < l1:
        f2 = [0] * (l1 - l2) + list(f2)
    f = [(int(a) + int(b)) % s for a, b in zip(f1, f2)]
    return f


# ---------------------------
# finite field COA (prime-power d = p^r)
# ---------------------------

def _all_field_elements(p: int, r: int) -> List[Tuple[int]]:
    """Return list of all r-length tuples with entries 0..p-1 in lexicographic order from itertools.product.
       Each tuple represents polynomial coefficients highest->lowest.
    """
    # itertools.product yields tuples where the leftmost element changes slowest; rightmost changes fastest.
    return list(itertools.product(range(p), repeat=r))


def _coeffs_to_index(coeffs: Sequence[int], p: int, r: int) -> int:
    """Map coefficient tuple (highest->lowest) to an index 0..p^r-1 using base-p representation."""
    val = 0
    for c in coeffs:
        val = val * p + int(c)
    return val  # 0-based index


def _index_to_coeffs(index: int, p: int, r: int) -> List[int]:
    """Convert 0-based index to coefficient vector of length r (highest->lowest)."""
    coeffs = [0] * r
    for i in range(r - 1, -1, -1):
        coeffs[i] = index % p
        index //= p
    return coeffs


def onecoa(d: int, p: int, f_d: Sequence[int], rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate COA for d = p^r (prime-power).
    - p: prime base
    - f_d: coefficients of primitive polynomial on GF(d), highest->lowest, length r+1
           e.g. for x^2 + x + 2 -> [1,1,2]
    Returns array shape (d*(d-1), d) with entries in 1..d.
    """
    # validate p prime
    if not galois.is_prime(p):
        raise ValueError("p should be prime")

    # compute r such that p^r == d
    r = 0
    tmp = 1
    while tmp < d:
        tmp *= p
        r += 1
    if tmp != d:
        raise ValueError("d is not a power of p")

    if rng is None:
        rng = np.random.default_rng()

    # Build multiplication (M_d) and addition (A_d) tables on GF(d)
    if r == 1:
        # for prime p, fields are simple integer modulo operations (0..p-1)
        M_d = np.zeros((p, p), dtype=int)
        A_d = np.zeros((p, p), dtype=int)
        for i in range(p):
            for j in range(i, p):
                M_val = ((i) * (j)) % p
                A_val = (i + j) % p
                M_d[i, j] = M_val
                M_d[j, i] = M_val
                A_d[i, j] = A_val
                A_d[j, i] = A_val
            M_d[i, i] = (i ** 2) % p
            A_d[i, i] = (2 * i) % p
    else:
        # r > 1: represent elements as polynomials with coefficients 0..p-1, length r
        entries = _all_field_elements(p, r)  # list of tuples length d
        # build maps from tuple->index quickly via base-p encoding
        tuple_to_index = {_coeffs_to_index(t, p, r): idx for idx, t in enumerate(entries)}
        # pre-generate list of coefficient lists for each element
        entries_list = [list(t) for t in entries]
        
        M_d = np.zeros((d, d*r), dtype=int)
        A_d = np.zeros((d, d*r), dtype=int)

        for i in range(d):
            for j in range(d):
                # multiply entry i and entry j as polynomials (coeff highest->lowest)
                prod = gfpoly_multi(entries_list[i], entries_list[j], p)  # length 2r-1
                # reduce modulo primitive polynomial f_d (length r+1) to get remainder length r
                rem = gfpoly_div(prod, f_d, p)  # remainder length r (highest->lowest)
                # find index of remainder among entries (map to 0..d-1)
                idx = _coeffs_to_index(rem, p, r)
                M_d[i, j] = idx  # store 0-based element value
                # addition: coefficient-wise mod p
                summ = gfpoly_add(entries_list[i], entries_list[j], p)
                idx2 = _coeffs_to_index(summ, p, r)
                A_d[i, j] = idx2

    # Construction of COA
    # firstr <- c(0, sample(d-1)) where sample(d-1) is permutation of 1..(d-1)
    sample_perm = rng.permutation(np.arange(1, d))
    firstr = np.concatenate(([0], sample_perm))  # length d, values in 0..d-1

    nr = d * (d - 1)
    coa = np.zeros((nr, d), dtype=int)
    for i in range(0, d):
        for j in range(1, d):  # j runs 1..d-1 (R), so here 1..d-1
            row_idx = i * (d - 1) + (j - 1)
            for k in range(0, d):
                e = M_d[j, int(firstr[k])]
                # Then A_d[i, e] (both 0-based indices) -> returns element value (0..d-1)
                val_elem = A_d[i, e]
                coa[row_idx, k] = val_elem

    # R returns coa + 1 to shift 0..d-1 -> 1..d
    return coa


def est_shcoa(d: int, n: int, val: Callable[[Sequence[int]], float], p: int, f_d: Sequence[int], *args, rng: np.random.Generator = None) -> np.ndarray:
    """
    Estimate Shapley values using COA for prime-power d (= p^r)
    - d: number of players (must equal p^r)
    - n: sample size; must be multiple of d*(d-1)
    - val: callable(sets, *args)
    - p: prime base
    - f_d: primitive polynomial coefficients (highest->lowest) of degree r (length r+1)
    - args: extra args passed to val(...)
    - rng: optional numpy Generator
    Returns: 1 x d numpy array (row vector) of estimated Shapley values
    """
    # validate inputs
    # check f_d length and coefficients: length should be r+1 where r = log_p(d)
    r = 0
    tmp = 1
    while tmp < d:
        tmp *= p
        r += 1
    if tmp != d:
        raise ValueError("d must be p^r for integer r")

    if len(f_d) != r + 1:
        raise ValueError("f_d must have length r+1 (primitive polynomial coefficients)")

    if max(f_d) >= p:
        raise ValueError("coefficients of f_d must be < p")

    m = d * (d - 1)
    if n % m != 0:
        raise ValueError("n should be a multiple of d*(d-1)")

    k = n // m
    sh = np.zeros(d, dtype=float)
    local_rng = rng or np.random.default_rng()

    for t in range(k):
        coat = onecoa(d, p, f_d, rng=local_rng)
        for l in range(m):
            perml = coat[l, :]  # values 1..d
            preC = 0.0
            for i in range(1, d + 1):
                subset = list(perml[:i])
                delta = float(val(subset, *args)) - preC
                #player_index = int(perml[i - 1]) - 1
                sh[int(perml[i - 1])] += delta
                preC += delta

    sh = sh / float(n)
    return sh.reshape(1, -1)


