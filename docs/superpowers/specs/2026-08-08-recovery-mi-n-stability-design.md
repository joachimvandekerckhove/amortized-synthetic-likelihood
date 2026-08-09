# Recovery, MI Gate, and N-Stability Repair Design

## Goal

Correct recovery true-parameter sampling, make the joint parameter MI gate
invariant to summary units, and complete the N-stability study for all four
models.

## Recovery true-parameter draws

`recovery.true_param_bounds`, when configured, defines the distribution used
to simulate the true parameter values for recovery subjects. The resolved
bounds will be included in worker arguments and used for every retry.
Inference priors remain unchanged: JAGS priors and chain initialization
continue to use `model.prior_bounds`.

Recovery output will retain separate true-draw and inference-bound metadata,
so figures and downstream analyses can distinguish simulation support from
the JAGS prior.

## Joint parameter MI gate

Before estimating KSG mutual information, each summary column will be
transformed to deterministic normalized ranks. The transform maps the
smallest and largest finite values to 0 and 1, averages tied ranks, and is
applied identically to the observed and permutation-null calculations.
This preserves each summary's ordering and makes the Chebyshev neighborhoods
independent of original summary units.

The parameter coordinate is left unchanged because it is one-dimensional and
KSG neighborhoods in one coordinate are invariant to positive rescaling.

## N-stability coverage

All four pipeline Makefiles expose `evaluate-n-stability`. Existing committed
DDM artifacts are retained. The DW evaluation will be run at its fixed
canonical theta profile and agent-count sizes, then its generated summary,
table, and plot data will be committed alongside the DDM artifacts.

`agent/REPRODUCE.md` will describe the all-model target, explicitly note that
DW's size is number of agents, and state that its profile uses fixed canonical
parameter pairs.

## Tests and verification

Regression tests will:

1. prove configured recovery draw bounds are used by a recovery worker;
2. prove joint MI is stable when an independent summary is rescaled;
3. retain coverage of DW's N-stability profile and all-model support;
4. run the complete Python test suite without an inherited DW configuration.

The DW full N-stability command will run after code tests pass. Its outputs
will be checked for all configured agent counts and committed.
