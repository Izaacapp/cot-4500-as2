import numpy as np


def neville(x_data, y_data, x):
    """
    Neville's Method: Computes the interpolating polynomial at a given x.
    This method constructs a divided difference table iteratively.

    Parameters:
    - x_data (array): The known x values
    - y_data (array): The corresponding function values f(x)
    - x (float): The target value to interpolate

    Returns:
    - float: The interpolated value at x
    """
    n = len(x_data)
    table = np.zeros((n, n))

    # Initialize first column with y_data values
    for i in range(n):
        table[i, 0] = y_data[i]

    # Compute the table using Neville’s recursive formula
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (
                (x - x_data[i + j]) * table[i, j - 1]
                - (x - x_data[i]) * table[i + 1, j - 1]
            ) / (x_data[i] - x_data[i + j])

    return table[0, n - 1]


# Data for Neville's Method
x_neville = np.array([3.6, 3.8, 3.9])
y_neville = np.array([1.675, 1.436, 1.318])
x_target = 3.7

print(neville(x_neville, y_neville, x_target))
print()


def divided_differences(x, y):
    """
    Constructs the Newton divided difference table for interpolation.

    Parameters:
    - x (array): The known x values
    - y (array): The corresponding function values f(x)

    Returns:
    - 2D numpy array: The divided difference table
    """
    n = len(x)
    table = np.zeros((n, n), dtype=float)

    # First column is just the function values
    table[:, 0] = y

    # Compute divided differences
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])

    return table


def evaluate_newton_divided(x, table, t, degree):
    """
    Evaluates Newton’s interpolating polynomial at a given point.

    Parameters:
    - x (array): The x values used for interpolation
    - table (2D array): The divided difference table
    - t (float): The target value to evaluate
    - degree (int): The degree of the polynomial to use

    Returns:
    - float: The interpolated value at t
    """
    result = table[0, 0]
    product = 1.0

    for i in range(1, degree + 1):
        product *= t - x[i - 1]
        result += table[0, i] * product

    return result


# Data for Newton's Divided Differences
x_newton = np.array([7.2, 7.4, 7.5, 7.6])
y_newton = np.array([23.5492, 25.3913, 26.8224, 27.4589])
t_newton = 7.3

# Generate divided difference table
dd_table = divided_differences(x_newton, y_newton)

# Extract polynomial coefficients
a1 = dd_table[0, 1]
a2 = dd_table[0, 2]
a3 = dd_table[0, 3]

print(a1)
print(a2)
print(a3)
print()

# Evaluate the polynomial at t_newton using degree 3
p3 = evaluate_newton_divided(x_newton, dd_table, t_newton, 3)
print(p3)
print()


def hermite():
    """
    Constructs and prints the Hermite interpolation divided difference table.
    Hermite interpolation considers both function values and derivatives.
    """
    x = [3.6, 3.6, 3.8, 3.8, 3.9, 3.9]
    fx = [1.675, 1.675, 1.436, 1.436, 1.318, 1.318]
    fpx = [-1.195, -1.195, -1.188, -1.188, -1.182, -1.182]

    n = len(x)
    table = [[0] * (n - 1) for _ in range(n)]

    # Fill first two columns
    for i in range(n):
        table[i][0] = x[i]
        table[i][1] = fx[i]

    # Handle derivative values
    for i in range(1, n):
        if x[i] == x[i - 1]:
            table[i][2] = fpx[i]
        else:
            table[i][2] = (table[i][1] - table[i - 1][1]) / (x[i] - x[i - 1])

    # Compute divided differences
    for j in range(3, n - 1):
        for i in range(j - 1, n):
            table[i][j] = (table[i][j - 1] - table[i - 1][j - 1]) / (
                x[i] - x[i - j + 1]
            )

    # Print table
    for i in range(n):
        print("[ ", end="")
        for j in range(n - 1):
            print(f"{table[i][j]: 12.8e} ", end="")
        print("]")

    print()


hermite()
print()


def cubic_spline():
    """
    Computes and prints the cubic spline interpolation matrices.
    """
    x = np.array([2, 5, 8, 10], dtype=float)
    f = np.array([3, 5, 7, 9], dtype=float)
    n = len(x)

    # Compute step sizes
    h = np.array([x[i + 1] - x[i] for i in range(n - 1)], dtype=float)

    # Construct matrix A
    A = np.zeros((n, n), dtype=float)
    A[0, 0], A[n - 1, n - 1] = 1.0, 1.0

    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]

    # Construct vector b
    b = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        b[i] = (3.0 / h[i]) * (f[i + 1] - f[i]) - (3.0 / h[i - 1]) * (f[i] - f[i - 1])

    # Solve for c in A*c = b
    c = np.linalg.solve(A, b)

    # Print matrix A
    for row in A:
        print("[", end="")
        for val in row:
            print(f"{val:3.0f}.", end="")
        print("]")

    # Print vector b
    print("[", end="")
    for val in b:
        print(f"{val:2.0f}.", end="")
    print("]")

    # Print vector c
    print("[0.", end="")
    for val in c[1:-1]:
        print(f" {val:.8f}", end="")
    print(" 0.]\n")


cubic_spline()
