Quantum-inspired algorithms for linear algebra
##############################################

Applying quantum-inspired algorithms to solve systems of linear equations
and to recommendation system

The repository contains all source code used to generate results
presented in `"Practical performance of quantum-inspired algorithms
for linear algebra"
[arXiv:to_be_specified] <https://arxiv.org/abs/to_be_specified>`_.

Contents
========

* ``quantum_inspired.py``: a Python module containing all functions composing the
  quantum-inspired algorithm. It contains three driver subroutines
  to use the implemented algorithms to address three practical applications:
  i)   to solve a general system of linear equations Ax = b,
  ii)  to optimize an investment portfolio across various assets, and
  iii) to recommend items to a given user based on an input preference matrix.

Usage and examples
==================

Below we describe usage of the module to tackle the above mentioned applications.
Notice that the first thing to do is to import the module. All input data required to
run these examples have been included in the repository.

1. Solving a system linear of equations Ax = b.

   .. code-block:: python
      :enphasize-lines: 12

      import quantum_inspired as qi
      # load a low-rank random matrix A with dimension 100 x 50
      A = np.load('A.npy')
      # load b vector (100 x 1) defining linear system Ax=b
      b = np.load('b.npy')
      # rank of matrix A
      rank = 5
      # Input parameters for the quantum inspired algorithm
      r = 80
      c = 50
      Nsamples = 10
      NcompX = 10
      sampled_comp, x = qi.linear_eqs(A, b, r, c, rank, Nsamples, NcompX)

where:

Args:

* ``A``: In general, a rectangular matrix
* ``b``: right-hand-side vector b
* ``r``: number of sampled rows from matrix A
* ``c``: number of sampled columns from matrix A
* ``rank``: rank of matrix A
* ``Nsamples``: number of stochastic samples performed to estimate coefficients ``lambda_l``
* ``NcompX``: number of entries to be sampled from the solution vector ``x_tilde``

Returns:

Tuple containing arrays with the ``NcompX`` sampled entries and corresponding components of
the solution vector ``x_tilde``.


2. Portfolio optimization.

   .. code-block:: python
      :enphasize-lines: 43

      import quantum_inspired as qi
      # Reading the correlation matrix
      corr_mat = np.load('SnP500_correlation.npy')
      # Reading vector of historical returns
      hist_returns = np.load('SnP500_returns.npy')

      mu = np.mean(hist_returns[:])

      n_assets = len(hist_returns[:])
      m_rows = n_assets + 1
      n_cols = n_assets + 1

      A = np.zeros((m_rows, n_cols))
      # Building the matrix A
      # In this case, matrix A is a block squared matrix of dimension 475 x 475
      # composed by the vector of historical returns r and correlation matrix \Sigma
      # of 474 assets comprised in the S&P 500 stock index.
      A[0, 0] = 0
      A[0, 1:n_cols] = hist_returns[:]
      A[1:m_rows, 0] = hist_returns[:]

      A[1:m_rows, 1:n_cols] = corr_mat[:, :]

      # b is a vector [\mu, \vec 0] with \mu being the expected return
      b = np.zeros(m_rows)
      b[0] = mu

      # This defines a portfolio optimization problem Ax = b
      # where x = [\nu, \vec{\omega}] with \vec{\omega} being the
      # portfolio allocation vector

      # low-rank approximation of matrix A
      rank = 5
      # Input parameters for the quantum inspired algorithm
      r = 340
      c = 340
      Nsamples = 10
      NcompX = 10

      # Notice that this function receive "mu" instead of the whole vector "b"
      # as the general coefficient <v_l|A^+|b> reduces to the inner product <mu*A_0., v_l>.
      # The latter allow us to reduce significantly the number of stochastic samples performed
      # to estimate "lambdas[0:rank]".
      sampled_comp, x = qi.linear_eqs_portopt(A, mu, r, c, rank, Nsamples, NcompX)

where:

    Args:

    * ``A``: In general, a rectangular matrix
    * ``b``: right-hand-side vector b
    * ``r``: number of sampled rows from matrix A
    * ``c``: number of sampled columns from matrix A
    * ``rank``: rank of matrix A
    * ``Nsamples``: number of stochastic samples performed to estimate coefficients ``lambda_l``
    * ``NcompX``: number of entries to be sampled from the solution vector ``x_tilde``

Returns:

    Tuple containing arrays with the ``NcompX`` sampled entries and corresponding components of
    the solution vector ``x_tilde``.

3. Recommendation system.

   .. code-block:: python
      :enphasize-lines: 20

      # load a preference matrix A of dimension m x n encoding the rates
      # provided by m = 611 users for n = 9724 movies
      A = np.load('A_movies_small.npy')

      # In this example we wan to reconstruct the full row of matrix A corresponding
      # to a specific user (416 in this case) and use highest components of the
      # reconstructed row vector to recommend new movies
      user = 416

      # low-rank approximation
      rank = 10
      # Input parameters for the quantum inspired algorithm
      r = 450
      c = 4500
      Nsamples = 10
      NcompX = 10
      sampled_comp, x = qi.recomm_syst(A, user, r, c, rank, Nsamples, NcompX)

where:

    Args:

    * ``A``: preference matrix
    * ``user``: row index of a specific user in the preference matrix A
    * ``r``: number of sampled rows from matrix A
    * ``c``: number of sampled columns from matrix A
    * ``rank``: rank of matrix A
    * ``Nsamples``: number of stochastic samples performed to estimate coefficients ``lambda_l``
    * ``NcompX``: number of entries to be sampled from the solution vector ``A[user, :]``

    Returns:

        Tuple containing arrays with the ``NcompX`` sampled entries and corresponding elements of
        the row vector ``A[user, :]``.

Requirements
============

Python

Authors
=======

Juan Miguel Arrazola, Alain Delgado, Bhaskar Roy Bardhan, Seth Lloyd

If you are doing any research using this source code, please cite the following paper:

> Juan Miguel Arrazola, Alain Delgado, Bhaskar Roy Bardhan, Seth Lloyd.
Practical performance of quantum-inspired algorithms for linear algebra. arXiv, 2019. [arXiv:to be defined](https://arxiv.org/abs/1809.04680)

License
=======

This source code is free and open source, released under the Apache License, Version 2.0.
