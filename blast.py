import pathlib, numpy as np, strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from strands_robots.utils import finite_vector_error, coerce_size_vector, sequence_length

class LenRaises:
    def __len__(self): raise TypeError("nope")
    def __iter__(self): return iter([0.1, 0.2, 0.3])

CASES = [
    ("generator",        lambda: (x for x in (0.1, 0.2, 0.3))),
    ("iter(list)",       lambda: iter([0.1, 0.2, 0.3])),
    ("range(3)",         lambda: range(3)),
    ("map",              lambda: map(float, [1, 2, 3])),
    ("0-d np array",     lambda: np.array(0.3)),
    ("np.float64",       lambda: np.float64(0.3)),
    ("plain float",      lambda: 0.3),
    ("str",              lambda: "abc"),
    ("dict",             lambda: {"a": 1.0}),
    ("set",              lambda: {0.1, 0.2}),
    ("list ok",          lambda: [0.1, 0.2, 0.3]),
    ("tuple ok",         lambda: (0.1, 0.2, 0.3)),
    ("np 1-d ok",        lambda: np.array([0.1, 0.2, 0.3])),
    ("empty list",       lambda: []),
    ("__len__ raises",   lambda: LenRaises()),
    ("nan inside",       lambda: [0.1, float("nan")]),
]
print("\n%-16s %-8s %-46s %s" % ("value", "seq_len", "finite_vector_error", "coerce_size_vector err"))
print("-"*150)
for name, mk in CASES:
    try: sl = repr(sequence_length(mk()))
    except BaseException as e: sl = f"RAISE {type(e).__name__}"
    try: fve = repr(finite_vector_error("m", "size", mk()))
    except BaseException as e: fve = f"RAISED {type(e).__name__}: {e}"
    try:
        _v, csv = coerce_size_vector("m", "size", mk()); csv = repr(csv)
    except BaseException as e: csv = f"RAISED {type(e).__name__}: {e}"
    print("%-16s %-8s %-46s %s" % (name, sl, fve[:44], csv[:70]))
