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
        additions = 0
        multiplications = 0
        for i in range(self.dim):
            for j in range(self.dim):
                sum = 0
                for k in range(self.dim):
                    sum += self.mat_A[i][k] * self.mat_B[k][j]
                    multiplications += 1
                    if k > 0:
                        additions += 1
                mat_C[i].append(sum)
        return mat_C, additions, multiplications
    
    def strassen_mm(self, dimension, mat_A, mat_B, arbitrary_dim=False):
        matrix_a = np.asarray(mat_A, dtype=np.float64)
        matrix_b = np.asarray(mat_B, dtype=np.float64)

        if arbitrary_dim:
            padded_dimension = 2 ** int(np.ceil(np.log2(dimension)))
            if padded_dimension != dimension:
                padded_a = np.zeros((padded_dimension, padded_dimension))
                padded_b = np.zeros((padded_dimension, padded_dimension))
                padded_a[:dimension, :dimension] = matrix_a
                padded_b[:dimension, :dimension] = matrix_b
                matrix_a, matrix_b = padded_a, padded_b

        def multiply(left, right):
            size = left.shape[0]
            if size == 1:
                return np.dot(left, right), 0, 1

            additions = 0
            multiplications = 0

            def add(left_matrix, right_matrix):
                return left_matrix + right_matrix, left_matrix.size

            def subtract(left_matrix, right_matrix):
                return left_matrix - right_matrix, left_matrix.size

            half = size // 2
            left_11 = left[:half, :half]
            left_12 = left[:half, half:]
            left_21 = left[half:, :half]
            left_22 = left[half:, half:]
            right_11 = right[:half, :half]
            right_12 = right[:half, half:]
            right_21 = right[half:, :half]
            right_22 = right[half:, half:]

            left_sum, count = add(left_11, left_22)
            additions += count
            right_sum, count = add(right_11, right_22)
            additions += count
            p, p_additions, p_multiplications = multiply(left_sum, right_sum)

            left_sum, count = add(left_21, left_22)
            additions += count
            q, q_additions, q_multiplications = multiply(left_sum, right_11)

            right_difference, count = subtract(right_12, right_22)
            additions += count
            r, r_additions, r_multiplications = multiply(left_11, right_difference)

            right_difference, count = subtract(right_21, right_11)
            additions += count
            s, s_additions, s_multiplications = multiply(left_22, right_difference)

            left_sum, count = add(left_11, left_12)
            additions += count
            t, t_additions, t_multiplications = multiply(left_sum, right_22)

            left_difference, count = subtract(left_21, left_11)
            additions += count
            right_sum, count = add(right_11, right_12)
            additions += count
            u, u_additions, u_multiplications = multiply(left_difference, right_sum)

            left_difference, count = subtract(left_12, left_22)
            additions += count
            right_sum, count = add(right_21, right_22)
            additions += count
            v, v_additions, v_multiplications = multiply(left_difference, right_sum)

            additions += sum((p_additions, q_additions, r_additions, s_additions,
                              t_additions, u_additions, v_additions))
            multiplications += sum((p_multiplications, q_multiplications, r_multiplications,
                                    s_multiplications, t_multiplications, u_multiplications,
                                    v_multiplications))

            result_11 = p + s - t + v
            result_12 = r + t
            result_21 = q + s
            result_22 = p - q + r + u
            additions += 8 * half ** 2
            return (np.vstack((np.hstack((result_11, result_12)),
                               np.hstack((result_21, result_22)))),
                    additions, multiplications)

        result, additions, multiplications = multiply(matrix_a, matrix_b)
        return result[:dimension, :dimension].tolist(), additions, multiplications
