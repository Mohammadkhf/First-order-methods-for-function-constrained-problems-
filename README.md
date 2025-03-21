# Optimal Primal-Dual Algorithm with Last iterate Convergence Guarantees for Stochastic Convex Optimization Problems
This repository contains the numerical experiments of the mentioned paper in the title. Please see the paper [here](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=HA-GlnkAAAAJ&citation_for_view=HA-GlnkAAAAJ:Y0pCki6q_DkC). We have the following files

 - Main_sparse_QCQP: Generates the problem settings with $\ell_1$-regularization.
     - To avoid overflow, we restart each algorithm with initialization from the first K/2 iterations.
- aug_conex: Corresponds to the Aug-ConEx algorithm in [our paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=HA-GlnkAAAAJ&citation_for_view=HA-GlnkAAAAJ:Y0pCki6q_DkC).
- proj: Corresponds to the Computation of the (x,s)-update in Aug-ConEx Method.
- ConEx: We compare our method in Aug-ConEx with ConEx presented in [this paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=brUg_BkAAAAJ&citation_for_view=brUg_BkAAAAJ:IjCSPb-OGe4C).
