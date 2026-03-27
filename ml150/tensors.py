# Tensor broadcasting
import numpy as np

def broadcast_ops(X: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes (X + b) * w using broadcasting.

    Args:
        X: Input matrix of shape (N, D)
        b: Bias vector of shape (D,)
        w: Weight vector of shape (N,)

    Returns:
        Resulting matrix of shape (N, D)
    """
    # Your code here
    sum = X + b
    activation = sum * w[:, np.newaxis] # w(N,) --> (N, 1) --(broadcast)--> (N, D)
    return activation

X = np.array([[1, 2], 
     [3, 4]])  # Shape (2, 2)
b = np.array([10, 20])  # Shape (2,)
w = np.array([0.5, 2])  # Shape (2,)

# print(broadcast_ops(X, b, w))

#########################################################################################################

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using 3 nested loops.
    """
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    # Initialize result matrix with zeros
    C = np.zeros((M, N))

    # Your code here
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]

    return C

# print("naive: ", matmul_naive(A, B))

def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using vectorized operations.
    """
    # Your code here
    return A @ B

# print("matmul_vectorized", matmul_vectorized(A, B))

#########################################################################################################

# Element wise operations:
from typing import Dict

a = np.array([1.0, 2.0])  # Shape (2,)
b = np.array([0.0, 2.0]) 

def elementwise_ops(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes element-wise add, mul, and safe div.
    
    Args:
        a: First tensor
        b: Second tensor (same shape)
        
    Returns:
        Dictionary with keys "add", "mul", "div"
    """
    epsilon = 1e-8

    assert a.shape == b.shape, "shapes must match"
    # Your code here
    add = np.add(a, b)

    mul = np.multiply(a, b)

    div = np.divide(a, b + epsilon)

    Results = {
        "add": add,
        "mul": mul,
        "div": div
    }

    return Results

# print(elementwise_ops(a, b))

#########################################################################################################

# Tensor reshaping and transposing

# Input: 1D vector with 24 elements
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
     
# Parameters
B = 1  # 1 image in batch
C = 2  # 2 channels
H = 3  # 3 rows (height)
W = 4  # 4 columns (width)

def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int) -> np.ndarray:
    """
    Reshapes flat x to (B, C, H, W) then transposes to (B, H, W, C).
    """
    # Your code here
    reshaped = x.reshape((B, C, H, W));

    transposed = np.transpose(reshaped, (0, 2, 3, 1))

    return transposed

# print(reshape_and_transpose(x, B, C, H, W).shape)

#########################################################################################################

# Reduction operations
from typing import Union

x = np.array([[1, 2, 3], 
     [4, 5, 6]])  # Shape (2, 3)
axis = 0  # Reduce along columns (second dimension)

def tensor_reductions(x: np.ndarray, axis: int) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes sum, mean, max, argmax along axis.
    """
    # Your code here
    sum = np.sum(x, axis=axis)

    mean = np.mean(x, axis=axis)

    max = np.max(x, axis=axis)

    argmax = np.argmax(x, axis=axis)

    results = {
        "sum": sum,
        "mean": mean,
        "max": max,
        "argmax": argmax
    }

    return results

# print(tensor_reductions(x, axis))

#########################################################################################################

# Vector norms

x = np.array([[3, 4],      # First vector: (3, 4)
     [1, -1]])     # Second vector: (1, -1)

def compute_norms(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes L1 and L2 norms for a batch of vectors.
    
    Args:
        x: Input matrix of shape (N, D)
        
    Returns:
        Dictionary with keys "l1" and "l2", each containing an array of shape (N,)
    """
    # Your code here
    l1_norm = np.linalg.norm(x, ord=1, axis=1)
    l2_norm = np.linalg.norm(x, ord=2, axis=1)

    results = {
        "l1_norm": l1_norm,
        "l2_norm": l2_norm
    }

    return results

# print(compute_norms(x))

#########################################################################################################

# vector_products

a = np.array([[1, 0, 0]])  # Unit vector along x-axis
b = np.array([[0, 1, 0]])  # Unit vector along y-axis

def vector_products(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes dot and cross products for batches of 3D vectors.
    
    Args:
        a: Shape (N, 3)
        b: Shape (N, 3)
        
    Returns:
        Dict with "dot" (N,) and "cross" (N, 3)
    """
    # Your code here
    # dot = np.dot(a, b)
    # dot = a @ b

    dot = np.sum(a * b, axis=1)

    cross = np.cross(a, b)

    results = {
        "dot": dot,
        "cross": cross
    }

    return results

# print(vector_products(a, b))

#########################################################################################################

# einsum

A = np.array([[1, 2],     # Shape (2, 2) - N=2, D=2
     [3, 4]])
B = np.array([[5, 6],     # Shape (2, 2) - D=2, M=2
     [7, 8]])

# Input: 2 batches, 2 heads, 3 sequence length, 4 dimensions
b, h, s, d = 2, 2, 3, 4

Q = np.random.randn(b, h, s, d)  # Shape (2, 2, 3, 4)
K = np.random.randn(b, h, s, d)  # Shape (2, 2, 3, 4)

def einsum_ops(A: np.ndarray, B: np.ndarray, Q: np.ndarray, K: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:

    transpose = np.einsum("ij->ji", A)

    sum = np.einsum("ij->", A)

    col_sum = np.einsum("ij->j", A)

    matmul = np.einsum("ik,kj->ij", A, B)

    batch_matmul = np.einsum("bhid,bhjd->bhij", Q, K)

    results = {
        "transpose": transpose,
        "sum": sum,
        "col_sum": col_sum,
        "matmul": matmul,
        "batch_matmul": batch_matmul
    }

    return results

print(einsum_ops(A, B, Q, K))
