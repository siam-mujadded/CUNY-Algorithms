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
        if dimension == 1:
            return [[mat_A[0][0] * mat_B[0][0]]]
        
        modified_dimension = dimension
        
        if dimension % 2 != 0: modified_dimension += 1
        
        half = modified_dimension // 2
            
        if modified_dimension > dimension:
            mat_A = self.pad_matrix(mat_A)
            mat_B = self.pad_matrix(mat_B)
        
        A_11 = [row[:half] for row in mat_A[:half]]
        A_12 = [row[half:] for row in mat_A[:half]]
        A_21 = [row[:half] for row in mat_A[half:]]
        A_22 = [row[half:] for row in mat_A[half:]]
        B_11 = [row[:half] for row in mat_B[:half]]
        B_12 = [row[half:] for row in mat_B[:half]]
        B_21 = [row[:half] for row in mat_B[half:]]
        B_22 = [row[half:] for row in mat_B[half:]]
        P = self.strassen_mm(half, np.add(A_11, A_22), np.add(B_11, B_22))
        Q = self.strassen_mm(half, np.add(A_21, A_22), B_11)
        R = self.strassen_mm(half, A_11, np.subtract(B_12, B_22))
        S = self.strassen_mm(half, A_22, np.subtract(B_21, B_11))
        T = self.strassen_mm(half, np.add(A_11, A_12), B_22)
        U = self.strassen_mm(half, np.subtract(A_21, A_11), np.add(B_11, B_12))
        V = self.strassen_mm(half, np.subtract(A_12, A_22), np.add(B_21, B_22))
        
        C_11 = np.add(np.add(P, S), np.subtract(V, T))
        C_12 = np.add(R, T)
        C_21 = np.add(Q, S)
        C_22 = np.add(np.subtract(np.add(P, R), Q), U)
        
        C = [list(C_11[i]) + list(C_12[i]) for i in range(half)]
        C.extend(list(C_21[i]) + list(C_22[i]) for i in range(half))
        C = [C[i][:dimension] for i in range(dimension)]
        return C

    def pad_matrix(self, matrix):
        matrix = [row.tolist() if isinstance(row, np.ndarray) else list(row) for row in matrix]
        for row in matrix:
            row.append(0)
        matrix.append([0] * len(matrix[0]))
        return matrix