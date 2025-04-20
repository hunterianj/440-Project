# 440-Project

### Data
Sachs dataset retrieved from https://github.com/cmu-phil/example-causal-datasets/blob/main/real/sachs/data/sachs.2005.continuous.txt \
    - shorter version from https://perso.univ-rennes1.fr/valerie.monbet/GM/Sachs.html \
IHDP dataset retrieved from https://www.fredjo.com \
    - Dataset with 1000 realizations is too large to be uploaded to github \

## To Do List

### Research

- Figure out which CI tests we'd expect to work best for which real-world data set
    - Sachs, observational (nonparameteric for continuous? what about discretized?)
    - Sachs, interventional
- Decide on a collection of constraint (PC, FCI, ...) and hybrid algorithms (GES, ...)

### Implementation work & testing

- Adapt CCPG/I-CCPG code ([from uhlerlab](https://github.com/uhlerlab/CCPG)) to `causal-learn` for comparison with other algorithms

#### Accuracy benchmark
- Implement graph distance metrics (s/c-metric from Wahl & Runge, or Markov distance from Faller & Janzing)
- Set up tests:
    - Load Sachs dataset
    - Find PAG for different CI tests/parameter choices
    - Report timing and distance from ground truth graph

#### Timing benchmark
