# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy import linalg as la
import time
import os


def ls_probs(m, n, A):

    r"""Function generating the length-squared (LS) probability distributions for sampling matrix A.

    Args:
        m (int): number of rows of matrix A
        n (int): row n of columns of matrix A
        A (array[complex]): most general case is a rectangular complex matrix

    Returns:
        tuple: Tuple containing the row-norms, LS probability distributions for rows and columns,
        and Frobenius norm.
    """

    # populates array with the row-norms squared of matrix A
    row_norms = np.zeros(m)
    for i in range(m):
        row_norms[i] = np.abs(la.norm(A[i, :]))**2

    # Frobenius norm of A
    A_Frobenius = np.sqrt(np.sum(row_norms))

    LS_prob_rows = np.zeros(m)

    # normalized length-square row probability distribution
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius**2

    LS_prob_columns = np.zeros((m, n))

    # populates array with length-square column probability distributions
    # LS_prob_columns[i]: LS probability distribution for selecting columns from row A[i]
    for i in range(m):
        LS_prob_columns[i, :] = [np.abs(k)**2 / row_norms[i] for k in A[i, :]]

    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius


def sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):

    r"""Function used to generate matrix C by performing LS sampling of rows and columns of matrix A.

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        m (int): number of rows of matrix A
        n (int): number of columns of matrix A
        r (int): number of sampled rows
        c (int): number of sampled columns
        row_norms (array[float]): norm of the rows of matrix A
        LS_prob_rows (array[float]): row LS probability distribution of matrix A
        LS_prob_columns (array[float]): column LS probability distribution of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        tuple: Tuple containing the singular values (sigma), left- (w) and right-singular vectors (vh) of matrix C,
        the sampled rows (rows), the column LS prob. distribution (LS_prob_columns_R) of matrix R and split running
        times for the FKV algorithm.
    """

    tic = time.time()
    # sample row indices from row length_square distribution
    rows = np.random.choice(m, r, replace=True, p=LS_prob_rows)

    columns = np.zeros(c, dtype=int)
    # sample column indices
    for j in range(c):
        # sample row index uniformly at random
        i = np.random.choice(rows, replace=True)
        # sample column from length-square distribution of row A[i]
        columns[j] = np.random.choice(n, 1, p=LS_prob_columns[i])

    toc = time.time()
    rt_sampling_C = toc - tic

    # building the lenght-squared distribution to sample columns from matrix R
    R_row = np.zeros(n)
    LS_prob_columns_R = np.zeros((r, n))

    for s in range(r):
        R_row[:] = A[rows[s], :] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))
        R_row_norm = np.abs(la.norm(R_row[:]))**2
        LS_prob_columns_R[s, :] = [np.abs(k)**2 / R_row_norm for k in R_row[:]]

    tic = time.time()
    # creates empty array for R and C matrices. We treat R as r x c here, since we only need columns later
    R_C = np.zeros((r, c))
    C = np.zeros((r, c))

    # populates array for matrix R with the submatrix of A defined by sampled rows/columns
    for s in range(r):
        for t in range(c):
            R_C[s, t] = A[rows[s], columns[t]]

        # renormalize each row of R
        R_C[s,:] = R_C[s,:] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))

    # creates empty array of column norms
    column_norms = np.zeros(c)

    # computes column Euclidean norms
    for t in range(c):
        for s in range(r):
            column_norms[t] += np.abs(R_C[s, t])**2

    # renormalize columns of C
    for t in range(c):
        C[:, t] = R_C[:, t] * (A_Frobenius / np.sqrt(column_norms[t])) / np.sqrt(c)

    toc = time.time()
    rt_building_C = toc - tic

    tic = time.time()
    # Computing the SVD of sampled C matrix
    w, sigma, vh = la.svd(C, full_matrices=False)

    toc = time.time()
    rt_svd_C = toc - tic

    return w, rows, sigma, vh, LS_prob_columns_R, rt_sampling_C, rt_building_C, rt_svd_C


def sample_me_lsyst(A, b, m, n, samples, rank, r, w, rows, sigma, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):

    r""" Function to estimate the coefficients :math: '\lambda_l = \langle v^l \vert A^\dagger \vert b \rangle'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        m (int): number of rows of matrix A
        n (int): number of columns of matrix A
        samples (int): number of stochastic samples performed to estimate :math: '\lambda_l'
        rank (int): rank of matrix A
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        sigma (array[float]): singular values of matrix C
        row_norms (array[float]): row norms of matrix A
        LS_prob_rows (array[float]): LS row probability distribution of matrix A
        LS_prob_columns (array[float]): LS column probability distribution of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        array[float]: Array containing the coefficients :math: '\lambda_l = \langle v^l \vert A^\dagger \vert b \rangle'
    """

    # Number of independent estimates. We take the median of these as the final estimate
    reps = 10

    # creates empty array of matrix elements <v^l|A^dagger|b> for l=1,2,.., k and many repetitions of the estimates
    matrix_elements = np.zeros((reps, rank))

    for i in range(reps):

        # calculate matrix element for l=1,2,.., k
        for l in range(rank):

            # create empty array of sampled matrix elements
            X = np.zeros(samples)

            # sample matrix elements
            for k in range(samples):

                # sample row index from length-square distribution
                sample_i = np.random.choice(m, 1, replace=True, p=LS_prob_rows)[0]
                # sample column index from length-square distribution from previously sampled row
                sample_j = np.random.choice(n, 1, p=LS_prob_columns[sample_i])[0]

                # j-th entry of right singular vector of matrix R
                v_j = 0

                # calculates v_j
                for s in range(r):
                    v_j += A[rows[s], sample_j] * w[s, l] / (np.sqrt(row_norms[rows[s]]))
                    # print(v_j)
                v_j = v_j * A_Frobenius / (np.sqrt(r) * sigma[l])

                # computes sampled matrix element
                X[k] = ((A_Frobenius ** 2 * b[sample_i]) / (A[sample_i, sample_j])) * v_j

            # assigns estimates for each l and repetition
            matrix_elements[i, l] = np.mean(X)

    # creates empty array of matrix elements <v_l|A|b>
    lambdas = np.zeros(rank)

    # take median of all repeated estimates
    for l in range(rank):
        lambdas[l] = np.median(matrix_elements[:, l])

    return lambdas


def sample_me_rsys(A, user, n, samples, rank, r, w, rows, sigma, row_norms, LS_prob_columns, A_Frobenius):

    r""" Function to estimate the coefficients :math: '\lambda_l = \langle A_\mathrm{user}^\mathrm{T}, v^l \rangle'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        user (int): labels the row index of a specific user in the preference matrix A
        n (int): number of columns of matrix A
        samples (int): number of stochastic samples performed to estimate :math: '\lambda_l'
        rank (int): rank of matrix A
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        sigma (array[float]): singular values of matrix C
        row_norms (array[float]): row norms of matrix A
        LS_prob_columns (array[float]): LS column probability distribution of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        array[float]: Array containing the coefficients :math: '\lambda_l = \langle A_\mathrm{user}^\mathrm{T}, v^l \rangle'
    """

    # Number of independent estimates. We take the median of these as the final estimate
    reps = 10

    # creates empty array of the coefficients lambda for l=1,2,.., k and many repetitions of the estimates
    coefficients = np.zeros((reps, rank))

    for i in range(reps):

        # calculate matrix element for l=1,2,..,k
        for l in range(rank):

            # create empty array of sampled matrix elements
            X = np.zeros(samples)

            # sample matrix elements
            for k in range(samples):

                # sample column index from length-square distribution from previously sampled row
                sample_j = np.random.choice(n, 1, p=LS_prob_columns[user])[0]
                # j-th entry of right singular vector of matrix R
                v_j = 0

                # calculates v_j
                for s in range(r):
                    v_j += A[rows[s], sample_j] * w[s, l] / (np.sqrt(row_norms[rows[s]]))
                    # print(v_j)
                v_j = v_j * A_Frobenius / (np.sqrt(r) * sigma[l])

                # computes sampled matrix element
                X[k] = (row_norms[user]*v_j) / (A[user, sample_j])

            # assigns estimates for each l and repetition
            coefficients[i, l] = np.mean(X)

    # creates empty array of coefficients
    lambdas = np.zeros(rank)

    # take median of all repeated estimates
    for l in range(rank):
        lambdas[l] = np.median(coefficients[:, l])

    return lambdas


def sample_from_x(A, r, n, rows, row_norms, LS_prob_columns_R, A_Frobenius, w_vector, w_norm):

    r""" Function to perform LS sampling of the solution vector :math: '\bm{x}'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        n (int): number of columns of matrix A
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        LS_prob_columns_R (array[float]): LS column prob. distribution of matrix R
        A_Frobenius (float): Frobenius norm of matrix A
        w_vector (array[float]): See paper for different definitions
        w_norm (float): norm of vector :math: '\omega'

    Returns:
        tuple: Tuple with the index of the sampled component and the number of rejected samples
    """

    keep_going = True
    out_j = 0
    counter = 0
    while keep_going:

        counter += 1
        # sample row index uniformly at random
        i_sample = np.random.choice(r)

        # sample column index from length-square distribution of corresponding row
        j_sample = np.random.choice(n, 1, p=LS_prob_columns_R[i_sample])[0]

        # column j_sample of matrix R
        R_j = np.zeros(r)

        # compute entries of R_j
        for s in range(r):
            R_j[s] = A[rows[s], j_sample] / np.sqrt(row_norms[rows[s]])
        R_j = (A_Frobenius/np.sqrt(r)) * R_j

        # norm of column vector R_j
        R_j_norm = la.norm(R_j)
        # inner product of R_j and w
        Rw_dot = np.dot(R_j, w_vector)

        # probability to select j_sample as output
        prob = (Rw_dot / (w_norm * R_j_norm))**2

        # determine if we output j_sample given above probability
        coin = np.random.binomial(1, prob)
        if coin == 1:
            out_j = j_sample
            # if we get heads from coin, then stop while loop
            keep_going = False

    return int(out_j), counter


def vl_vector(l, A, r, w, rows, sigma, row_norms, A_Frobenius):

    r""" Function to reconstruct right-singular vector of matrix A

    Args:
        l (int): singular vector index
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        array[float]: reconstructed right-singular vector
    """

    n = len(A[1, :])
    v_approx = np.zeros(n)
    # building approximated v^l vector
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l] )
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    return v_approx


def uvl_vector(l, A, r, w, rows, sigma, row_norms, A_Frobenius):

    r""" Function to reconstruct right-singular vector of matrix A

    Args:
        l (int): singular vector index
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        tuple: Tuple with arrays containing approximated singular vectors :math: '\bm{u}^l, \bm{v}^l'
    """

    m, n = A.shape
    u_approx = np.zeros(m)
    v_approx = np.zeros(n)
    # building approximated v^l vector
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l] )
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    u_approx = (A @ v_approx) / sigma[l]

    return u_approx, v_approx


# SUBPROGRAM TO COMPUTE APPROXIMATED SOLUTIONS \TILDE X
def approx_solution(A, rank, r, w, rows, sigma, row_norms, A_Frobenius, lambdas, comp):

    r""" Function to compute the approximated value for a specific entry of the solution vector
    :math: '\widetilde{x}_\mathrm{comp}' for the system of linear equations :math: 'A \bm{x} = b'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        rank (int): rank of matrix A
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        sigma (array[float]): singular values of matrix C
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A
        lambdas (array[float]): coefficients :math: '\lambda_l = \langle v^l \vert A^\dagger \vert b \rangle'
        comp (int): entry of the solution vector to be evaluated

    Returns:
        float: component of the solution vector :math: '\widetilde{x}_\mathrm{comp}'
    """

    approx_value = 0
    for l in range(rank):

        # building the component "comp" of vector v^l
        v_comp = 0
        for s in range(r):
            v_comp += A[rows[s], comp] * w[s, l] / np.sqrt( row_norms[ rows[s] ] )
        v_comp = v_comp * A_Frobenius / (np.sqrt(r) * sigma[l])

        # computing the approximated value for x (\tilde x)
        approx_value += v_comp * lambdas[l] / sigma[l]**2

    return approx_value


def approx_solution_rsys(A, rank, r, w, rows, sigma, row_norms, A_Frobenius, lambdas, comp):

    r""" Function to compute the matrix element :math: 'A_{\mathrm{user}, \mathrm{comp}}' of the preference matrix A

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        rank (int): rank of matrix A
        r (int): number of sampled rows from matrix A
        sigma (array[float]): singular values of matrix C
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A
        lambdas (array[float]): coefficients :math: '\lambda_l = \langle A_\mathrm{user}^\mathrm{T}, v^l \rangle'
        comp (int): entry of the solution vector to be evaluated

    Returns:
        float: the element :math: 'A_{\mathrm{user}, \mathrm{comp}}'
    """

    approx_value = 0
    for l in range(rank):

        # building the component "comp" of vector v^l
        v_comp = 0
        for s in range(r):
            v_comp += A[rows[s], comp] * w[s, l] / np.sqrt( row_norms[ rows[s] ] )
        v_comp = v_comp * A_Frobenius / (np.sqrt(r) * sigma[l])

        # computing the approximated value for x (\tilde x)
        approx_value += v_comp * lambdas[l]

    return approx_value


def print_output(r, c, rank, sigma, ul_approx, vl_approx, Nsamples, lambdas, NcompX, sampled_comp, x_tilde,
                 rt_ls_prob, rt_sampling_C, rt_building_C, rt_svd_C, rt_sampling_me, rt_sampling_sol):

    r""" Function printing out numerical results and running times

    Args:
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A
        sigma (array[float]): singular values of matrix C
        vl_approx (array[float]): reconstructed right-singular vectors of matrix A
        Nsamples (int): number of stochastic samples performed to estimate :math: '\lambda_l'
        lambdas (array[float]): coefficients :math: '\lambda_l'
        NcompX (int): number of entries to be sampled from the solution vector
        sampled_comp (array[int]): indices with sampled entries
        x_tilde (array[float]): stores NcompX values components of the vector solution
        rt_ls_prob (float): running time to compute LS prob. distributions
        rt_sampling_C (float): running time to sample 'r' rows and 'c' columns of matrix A
        rt_building_C (float): running time to build submatrix C
        rt_svd_C (float): running time to perform SVD of submatrix C
        rt_sampling_me (float): running time to sample all coeffients :math:'lambda_l'
        rt_sampling_sol (float): running time to sample entries from the vector solution
    """

    filename = "timing_C_{}_x_{}_Nsamples_{}_rank_{}_NcompX_{}.out".format(r, c, Nsamples, rank, NcompX)

    with open(filename, 'w') as f:
        f.write("#  r\t   c\trt_ls_prob\trt_sampling_C\trt_building_C\trt_svd_C\trt_sampling_me\trt_sampling_sol \n")
        f.write(" {:4d} \t {:4d} \t {:6.4f} \t {:6.4f} \t {:6.4f} \t {:6.4f} \t {:6.4f} \t {:6.4f} \n".format(
                r, c, rt_ls_prob, rt_sampling_C, rt_building_C, rt_svd_C, rt_sampling_me, rt_sampling_sol))

    # approximated singular values and right-vectors and coefficients lambda_l
    filename = "sigma_l_C_{}_x_{}_rank_{}.out".format(r, c, rank)

    with open(filename, 'w') as f:
        f.write("#  l\t            sigma_l \n")
        for l in range(rank):
            f.write("{:4d} \t {:20.10f} \n".format(l + 1, sigma[l]))
            np.save("v_l_" + str(l), vl_approx[:, l])
            np.save("u_l_" + str(l), ul_approx[:, l])

    # approximated coefficients lambda_l
    filename = "lambda_l_C_{}_x_{}_rank_{}_Nsamples_{}.out".format(r, c, rank, Nsamples)

    with open(filename, 'w') as f:
        f.write("#  l\t          lambda_l \n")
        for l in range(rank):
            f.write("{:4d} \t {:20.10f} \n".format(l + 1, lambdas[l]))

    # sampled components of the approximate vector solution

    filename = "x_vector_C_{}_x_{}_rank_{}_Nsamples_{}.out".format(r, c, rank, Nsamples)

    with open(filename, 'w') as f:
        f.write("#  i \t comp[i] \t x[comp[i]] \n")
        for t in range(NcompX):
            f.write("{:4d} \t {:4d} \t       {:12.8f} \n".format(t, sampled_comp[t] + 1, x_tilde[t]))

    return


def linear_eqs(A, b, r, c, rank, Nsamples, NcompX):

    r""" Function to solve the the linear system of equations :math:'A \bm{x} = b' using the quantum-inspired
    algorithm

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        b (array[float]): right-hand-side vector b
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A
        Nsamples (int): number of stochastic samples performed to estimate :math: '\lambda_l'
        NcompX (int): number of entries to be sampled from the solution vector :math:'\bm{x}'

    Returns:
        tuple: Tuple containing arrays with the sampled entries and corresponding components of
        the solution vector :math: '\bm{x}'
    """

    m_rows, n_cols = np.shape(A)

    # 1- Generating LS probability distributions to sample from matrix A
    tic = time.time()

    LS = ls_probs(m_rows, n_cols, A)

    toc = time.time()

    rt_ls_prob = toc - tic

    # 2- Building matrix C by sampling "r" rows and "c" columns from matrix A and computing SVD of matrix C
    svd_C = sample_C(A, m_rows, n_cols, r, c, *LS[0:4])
    w = svd_C[0]
    sigma = svd_C[2]

    # Reconstruction of the right-singular vectors of matrix A
    ul_approx = np.zeros((m_rows, rank))
    vl_approx = np.zeros((n_cols, rank))
    for l in range(rank):
        ul_approx[:, l], vl_approx[:, l] = uvl_vector(l, A, r, w, svd_C[1], sigma, LS[0], LS[3])

    # 3- Sampling of the matrix elements lambdas[0:rank] = <v^l|A^dagger|b>
    tic = time.time()
    lambdas = sample_me_lsyst(A, b, m_rows, n_cols, Nsamples, rank, r, *svd_C[0:3], *LS[0:4])
    toc = time.time()
    rt_sampling_me = toc - tic

    # 4- Sampling the vector solution
    tic = time.time()

    # computes vector w = sum_l lambda_l/sigma_l^3 * w_l
    w_vector = np.zeros(r)
    for l in range(rank):
        w_vector[:] += (lambdas[l] / sigma[l] ** 3) * w[:, l]

    w_norm = la.norm(w_vector)

    # create array to stored the sampled components
    sampled_comp = np.zeros(NcompX, dtype=np.uint32)
    n_of_rejected_samples = np.zeros(NcompX, dtype=np.uint32)
    x_tilde = np.zeros(NcompX)

    for t in range(NcompX):
        sampled_comp[t], n_of_rejected_samples[t] = \
            sample_from_x(A, r, n_cols, svd_C[1], LS[0], svd_C[4], LS[3], w_vector, w_norm)

    toc = time.time()
    rt_sampling_sol = toc - tic

    for t in range(NcompX):
        x_tilde[t] = approx_solution(A, rank, r, w, svd_C[1], svd_C[2],
                                     LS[0], LS[3], lambdas, sampled_comp[t])

    RT = [rt_ls_prob, *svd_C[5:8], rt_sampling_me, rt_sampling_sol]

    # 5- Printing output of the algorithm

    FKV = [r, c, rank, sigma, ul_approx, vl_approx]
    MC  = [Nsamples, lambdas]
    RS  = [NcompX, sampled_comp, x_tilde]
    RT  = [rt_ls_prob, *svd_C[5:8], rt_sampling_me, rt_sampling_sol]

    print_output(*FKV, *MC, *RS, *RT)

    return sampled_comp, x_tilde


def recomm_syst(A, user, r, c, rank, Nsamples, NcompX):

    r""" Function to compute missing entries of preference matrix row :math: 'A_{\mathrm{user}.}' for the user "user"

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        user (int): labels the row index of a specific user in the preference matrix A
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A
        Nsamples (int): number of stochastic samples performed to estimate :math: '\lambda_l'
        NcompX (int): number of entries to be sampled from the preference matrix row :math:'A_{\mathrm{user}.}'

    Returns:
        tuple: Tuple containing arrays with the sampled entries and corresponding elements of
        the preference matrix row :math:'A_{\mathrm{user}.}'
    """

    m_rows, n_cols = np.shape(A)

    # 1- Generating LS probability distributions to sample from matrix A
    tic = time.time()

    LS = ls_probs(m_rows, n_cols, A)

    toc = time.time()

    rt_ls_prob = toc - tic

    # 2- Building matrix C by sampling "r" rows and "c" columns from matrix A and computing SVD of matrix C
    svd_C = sample_C(A, m_rows, n_cols, r, c, *LS[0:4])
    w = svd_C[0]
    sigma = svd_C[2]

    # Reconstruction of the right-singular vectors of matrix A
    ul_approx = np.zeros((m_rows, rank))
    vl_approx = np.zeros((n_cols, rank))
    for l in range(rank):
        ul_approx[:, l], vl_approx[:, l] = uvl_vector(l, A, r, w, svd_C[1], sigma, LS[0], LS[3])

    # 3- Sampling of the matrix elements lambdas[0:rank] = <v^l, A[user, :]>
    tic = time.time()

    lambdas = sample_me_rsys(A, user, n_cols, Nsamples, rank, r, *svd_C[0:3], LS[0], *LS[2:4] )

    toc = time.time()
    rt_sampling_me = toc - tic

    # 4- Sampling the vector solution
    tic = time.time()

    # computes vector w = sum_l lambda_l/sigma_l * w_l
    w_vector = np.zeros(r)
    for l in range(rank):
        w_vector[:] += (lambdas[l] / sigma[l]) * w[:, l]

    w_norm = la.norm(w_vector)

    # create array to stored the sampled components
    sampled_comp = np.zeros(NcompX, dtype=np.uint32)
    n_of_rejected_samples = np.zeros(NcompX, dtype=np.uint32)
    x_tilde = np.zeros(NcompX)

    for t in range(NcompX):
        sampled_comp[t], n_of_rejected_samples[t] = \
            sample_from_x(A, r, n_cols, svd_C[1], LS[0], svd_C[4], LS[3], w_vector, w_norm)

    toc = time.time()
    rt_sampling_sol = toc - tic

    for t in range(NcompX):
        x_tilde[t] = approx_solution_rsys(A, rank, r, w, svd_C[1], svd_C[2],
                                          LS[0], LS[3], lambdas, sampled_comp[t])

    # 5- Printing out extensive information

    FKV = [r, c, rank, sigma, ul_approx, vl_approx]
    MC  = [Nsamples, lambdas]
    RS  = [NcompX, sampled_comp, x_tilde]
    RT  = [rt_ls_prob, *svd_C[5:8], rt_sampling_me, rt_sampling_sol]

    print_output(*FKV, *MC, *RS, *RT)

    return sampled_comp, x_tilde


def linear_eqs_portopt(A, mu, r, c, rank, Nsamples, NcompX):

    r""" Function to optimize the portfolio allocation vector for different assets for a given
    expected return

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        mu (float): expected return
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): low-rank approximation of matrix A
        Nsamples (int): number of stochastic samples performed to estimate :math: '\lambda_l'
        NcompX (int): number of entries to be sampled from the portfolio allocation vector

    Returns:
        tuple: Tuple containing arrays with the sampled entries and corresponding components of
        the portfolio allocation vector
    """

    m_rows, n_cols = np.shape(A)

    # 1- Generating LS probability distributions to sample from matrix A
    tic = time.time()

    LS = ls_probs(m_rows, n_cols, A)

    toc = time.time()

    rt_ls_prob = toc - tic

    # 2- Building matrix C by sampling "r" rows and "c" columns from matrix A and computing SVD of matrix C
    svd_C = sample_C(A, m_rows, n_cols, r, c, *LS[0:4])
    w = svd_C[0]
    sigma = svd_C[2]

    # Reconstruction of the right-singular vectors of matrix A
    ul_approx = np.zeros((m_rows, rank))
    vl_approx = np.zeros((n_cols, rank))
    for l in range(rank):
        ul_approx[:, l], vl_approx[:, l] = uvl_vector(l, A, r, w, svd_C[1], sigma, LS[0], LS[3])

    # 3- Sampling of the matrix elements lambdas[0:rank] = <v^l|A^dagger|b>
    tic = time.time()
    lambdas = sample_me_rsys(A, 0, n_cols, Nsamples, rank, r, *svd_C[0:3], LS[0], *LS[2:4])
    lambdas = mu*lambdas
    toc = time.time()
    rt_sampling_me = toc - tic

    # 4- Sampling the vector solution
    tic = time.time()

    # computes vector w = sum_l lambda_l/sigma_l^3 * w_l
    w_vector = np.zeros(r)
    for l in range(rank):
        w_vector[:] += (lambdas[l] / sigma[l] ** 3) * w[:, l]

    w_norm = la.norm(w_vector)

    # create array to stored the sampled components
    sampled_comp = np.zeros(NcompX, dtype=np.uint32)
    n_of_rejected_samples = np.zeros(NcompX, dtype=np.uint32)
    x_tilde = np.zeros(NcompX)

    for t in range(NcompX):
        sampled_comp[t], n_of_rejected_samples[t] = \
            sample_from_x(A, r, n_cols, svd_C[1], LS[0], svd_C[4], LS[3], w_vector, w_norm)

    toc = time.time()
    rt_sampling_sol = toc - tic

    for t in range(NcompX):
        x_tilde[t] = approx_solution(A, rank, r, w, svd_C[1], svd_C[2],
                                     LS[0], LS[3], lambdas, sampled_comp[t])

    RT = [rt_ls_prob, *svd_C[5:8], rt_sampling_me, rt_sampling_sol]

    # 5- Printing out extensive information

    FKV = [r, c, rank, sigma, ul_approx, vl_approx]
    MC  = [Nsamples, lambdas]
    RS  = [NcompX, sampled_comp, x_tilde]
    RT  = [rt_ls_prob, *svd_C[5:8], rt_sampling_me, rt_sampling_sol]

    print_output(*FKV, *MC, *RS, *RT)

    return sampled_comp, x_tilde


def linear_eqs_fkv(A, b, r, c, rank):

    r""" Function to solve the the linear system of equations :math:'A \bm{x} = b' using FKV algorithm
    and a direct calculation of the coefficients :math: '\lambda_l' and solution vector :math: '\bm{x}'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        b (array[float]): right-hand-side vector b
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A

    Returns:
        array[float]: array containing the components of the solution vector :math: '\bm{x}'
    """

    m_rows, n_cols = np.shape(A)

    # 1- Generating LS probability distributions used to sample rows and columns indices of matrix A
    tic = time.time()

    LS = ls_probs(m_rows, n_cols, A)

    toc = time.time()

    rt_ls_prob = toc - tic

    # 2- Building matrix C by sampling "r" rows and "c" columns from matrix A and computing SVD of matrix C
    svd_C = sample_C(A, m_rows, n_cols, r, c, *LS[0:4])
    w = svd_C[0]
    sigma = svd_C[2]
    rt_sampling_C = svd_C[5]
    rt_building_C = svd_C[6]
    rt_svd_C = svd_C[7]

    # Reconstruction of the right-singular vectors of matrix A
    ul_approx = np.zeros((m_rows, rank))
    vl_approx = np.zeros((n_cols, rank))
    for l in range(rank):
        ul_approx[:, l], vl_approx[:, l] = uvl_vector(l, A, r, w, svd_C[1], sigma, LS[0], LS[3])

    # 3- Direct calculation of matrix elements lambdas[rank] = <v^l|A^dagger|b>
    tic = time.time()
    lambdas = np.zeros(rank)
    for l in range(rank):
        lambdas[l] = np.transpose(vl_approx[:, l]) @ np.transpose(A) @ b
    toc = time.time()
    rt_dcalc_lambdas = toc - tic

    # 4- Direct calculation of the approximate vector solution \tilde X
    tic = time.time()
    x_tilde = np.zeros(n_cols)
    for l in range(rank):
        x_tilde[:] += (lambdas[l]/sigma[l]**2) * vl_approx[:, l]
    toc = time.time()
    rt_dcalc_x = toc - tic

    # 5- Printing out numerical results

    # timing information
    filename = "timing_C_{}_x_{}_rank_{}.out".format(r, c, rank)
    with open(filename,'w') as f:
        f.write("#  r\t   c \t rt_ls_prob \t rt_sampling_C \t rt_building_C \t rt_svd_C \t rt_dcalc_lambdas"
                "\t rt_dcalc_x \n")
        f.write(" {:4d} \t {:4d} \t {:6.4f} \t {:6.4f} \t {:6.4f} \t {:6.4f} \t {:6.4f} \t {:6.4f} \n"
                .format(r, c, rt_ls_prob, rt_sampling_C, rt_building_C, rt_svd_C, rt_dcalc_lambdas, rt_dcalc_x))

    # Approximate singular values and right-singular vectors
    filename = "sigma_l_C_{}_x_{}_rank_{}.out".format(r, c, rank)
    with open(filename,'w') as f:
        f.write("#  l \t        sigma_l \n")
        for l in range(rank):
            f.write("{:4d} \t {:20.10f} \n" .format(l + 1, sigma[l]))
            np.save("v_l_" + str(l), vl_approx[:, l])
            np.save("u_l_" + str(l), ul_approx[:, l])

    # Coefficients lambda_l = <v_l|A^+|b>
    filename = "lambda_l_C_{}_x_{}_rank_{}.out".format(r, c, rank)
    with open(filename, 'w') as f:
        f.write("#  l \t lambda_l \n")
        for l in range(rank):
            f.write("{:4d} \t {:20.10f} \n" .format(l + 1, lambdas[l]))

    # Approximate vector solution
    filename = "x_vector_C_{}_x_{}_rank_{}.out".format(r, c, rank)
    with open(filename, 'w') as f:
        f.write("#  i \t X[i] \n")
        for ii in range(n_cols):
            f.write("   {:i} \t  {:20.10f} \n" .format(ii, x_tilde[ii]))

    return x_tilde
