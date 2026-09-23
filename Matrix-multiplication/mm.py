import numpy as np

class MatrixMultiplication:
    def __init__(self, k, dim = 1, irregular = False):
        self.input_k = k
        self.dim = 1
        self.mat_A = None
        self.mat_B = None
        if not irregular: self.dim = 2 ** k
        else: self.dim = dim
    
    def take_input_from_file(self, file_name):
        with open(file_name, "r") as input_file:
            lines = [line.strip() for line in input_file if line.strip()]

        if not lines:
            raise ValueError("The input file is empty.")

        dimension = int(lines[0])
        expected_lines = 2 * dimension + 1
        if len(lines) != expected_lines:
            raise ValueError(
                f"Expected {expected_lines} non-empty lines, found {len(lines)}."
            )

        matrices = []
        for start in (1, dimension + 1):
            matrix = []
            for line in lines[start:start + dimension]:
                row = [int(value) for value in line.split()]
                if len(row) != dimension:
                    raise ValueError("Every matrix row must contain n numbers.")
                matrix.append(row)
            matrices.append(matrix)

        self.Mat_A, self.Mat_B = matrices
        self.mat_A, self.mat_B = self.Mat_A, self.Mat_B
        
    def naive_mm(self):
        mat_C = [[] for _ in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.dim):
                sum = 0
                for k in range(self.dim):
                    sum += self.mat_A[i][k] * self.mat_B[k][j]
                mat_C[i].append(sum)
        return mat_C
    
    def strassen_mm(self, dimension, mat_A, mat_B):
        matrix_a = np.asarray(mat_A, dtype=np.float64)
        matrix_b = np.asarray(mat_B, dtype=np.float64)

        padded_dimension = 1
        while padded_dimension < dimension:
            padded_dimension *= 2

        if padded_dimension != dimension:
            padded_a = np.zeros((padded_dimension, padded_dimension))
            padded_b = np.zeros((padded_dimension, padded_dimension))
            padded_a[:dimension, :dimension] = matrix_a
            padded_b[:dimension, :dimension] = matrix_b
            matrix_a, matrix_b = padded_a, padded_b

        cutoff = 32

        def multiply(left, right):
            size = left.shape[0]
            if size <= cutoff:
                return left @ right

            half = size // 2
            left_11 = left[:half, :half]
            left_12 = left[:half, half:]
            left_21 = left[half:, :half]
            left_22 = left[half:, half:]
            right_11 = right[:half, :half]
            right_12 = right[:half, half:]
            right_21 = right[half:, :half]
            right_22 = right[half:, half:]

            p = multiply(left_11 + left_22, right_11 + right_22)
            q = multiply(left_21 + left_22, right_11)
            r = multiply(left_11, right_12 - right_22)
            s = multiply(left_22, right_21 - right_11)
            t = multiply(left_11 + left_12, right_22)
            u = multiply(left_21 - left_11, right_11 + right_12)
            v = multiply(left_12 - left_22, right_21 + right_22)

            result_11 = p + s - t + v
            result_12 = r + t
            result_21 = q + s
            result_22 = p - q + r + u
            return np.vstack((np.hstack((result_11, result_12)),
                              np.hstack((result_21, result_22))))

        return multiply(matrix_a, matrix_b)[:dimension, :dimension].tolist()

    def pad_matrix(self, matrix):
        matrix = [row.tolist() if isinstance(row, np.ndarray) else list(row) for row in matrix]
        for row in matrix:
            row.append(0)
        matrix.append([0] * len(matrix[0]))
        return matrix