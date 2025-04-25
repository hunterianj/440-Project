from causallearn.search.ConstraintBased import CCPG
from causallearn.utils.cit import CIT
import h5py
import os
from pytorch_lightning import seed_everything
from CCPG.data import synthetic_instance  # vendor from authors
from time import time

os.makedirs("simulate-data/nsample", exist_ok=True)

def simulated_data_test(nnodes: int, max_runs: int) -> bool:
    seed_everything(42)
    alpha = 1e-3

    outdir = "simulate-data/nsample"
    os.makedirs(outdir, exist_ok=True)

    # generate the random in-star DAG
    model = synthetic_instance(nnodes, 1.0, True)
    target_arcs = set(model.DAG.arcs)

    # pre-generate sample files
    for r in range(max_runs):
        for n in range(100, 2000, 100):
            samples = model.sample(n)
            fname = os.path.join(outdir, f"samples_{n}_{r}.h5")
            with h5py.File(fname, "w") as f:
                f.create_dataset("samples", data=samples)

    # try to recover the true DAG from the synthetic model in less than 5000 samples
    for r in range(max_runs):
        ns = 100
        succeeded = False
        while ns < 5000:
            samples = model.sample(ns)
            cit = CIT(samples.T, "fisherz")
            def ci_test(i: int, j: int, cond: set[int]) -> bool:
                return cit(i, j, list(cond)) > alpha

            ci = CCPG.MemoizedCIT(samples.T, "fisherz", alpha=1e-3)
            # comps, edges = CCPG.ccpg_alg(set(range(nnodes)), ci_test, verbose=False)
            comps, edges = CCPG.ccpg_alg(set(range(nnodes)), ci.is_ci, verbose=False)

            # check that every component is size 1
            if all(len(c) == 1 for c in comps):
                found = {
                    (list(comps[i])[0], list(comps[j])[0])
                    for (i, j) in edges
                }
                if found == target_arcs:
                    print(f"  ✓ nnodes={nnodes}, run={r} succeeded at nsamples={ns}")
                    succeeded = True
                    break
            ns += 100

        if not succeeded:
            print(f"✗ nnodes={nnodes}, run={r} FAILED to recover within 5000 samples")
            return False

    return True

start = time()
all_ok = True
for nn in [5, 10, 15]:
    print(f"Testing nnodes={nn} …")
    if not simulated_data_test(nnodes=nn, max_runs=3):
        all_ok = False

print(f"Elapsed time ={time() - start}")

if all_ok:
    print("All CCPG recovery tests passed")
else:
    print("Some CCPG tests failed")


# authors' implementation
from CCPG import ccpg as ccpg_original
from causaldag import (partial_correlation_suffstat,
                       partial_correlation_test,
                       MemoizedCI_Tester)

# causal-learn’s implementation
from causallearn.search.ConstraintBased import CCPG as ccpg_cl

def compare_algs(nnodes=10, nsamples=500, alpha=1e-3):
    # generate a random in-star DAG
    model = synthetic_instance(nnodes, 1.0, True)

    # draw a batch of samples
    X = model.sample(nsamples)

    # authors' CCPG
    suffstat = partial_correlation_suffstat(X.T)
    ci_authors = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=alpha)
    comps_authors, edges_authors = ccpg_original.ccpg(set(range(nnodes)), ci_authors, verbose=False)

    # our CCPG
    ci_ours = ccpg_cl.MemoizedCIT(X.T, "fisherz", alpha=alpha)
    comps_ours, edges_ours = ccpg_cl.ccpg_alg(set(range(nnodes)), ci_ours.is_ci, verbose=False)

    # freeze components
    set_authors = set(frozenset(c) for c in comps_authors)
    set_ours = set(frozenset(c) for c in comps_ours)

    # simple comparison
    print("Authors' comps:", set_authors)
    print("Ours comps:", set_ours)
    print("Match? ", set_authors == set_ours)
    print("Authors' edges:", set(edges_authors))
    print("Ours edges:", set(edges_ours))
    print("Match?", set(edges_authors) == set(edges_ours))
    return (set_authors == set_ours) and (set(edges_authors) == set(edges_ours))

seed_everything(42)
start = time()
for nn in [5, 10, 15]:
    for nsamples in [500, 1000, 2000, 4000]:
        print(f"Testing nnodes={nn}, nsamples={nsamples}...")
        compare_algs(nnodes=nn, nsamples=nsamples)

print(f"Elapsed time ={time() - start}")