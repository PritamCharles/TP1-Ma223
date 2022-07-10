import numpy as np
import src.gauss_method as gm
import src.lu_method as lm

# Partie 1
# Question 1
print(tp01.ReductionGauss(np.array([[2, 5, 6, 7], [4, 11, 9, 12], [-2, -8, 7, 3]])), "\n")

# Question 2
print(tp01.ResolutionSysTriSup(tp01.ReductionGauss(np.array([[2, 5, 6, 7], [4, 11, 9, 12], [-2, -8, 7, 3]]))), "\n")

# Question 3
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])
print(tp01.Gauss(A, B), "\n")


# Partie 2
# Question 1
print(tp01.DecompositionLU(np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])), "\n")

# Question 2
A = np.array([[2, 5], [4, 9], [-2, -8]])
L = tp01.DecompositionLU(A)[0]
U = tp01.DecompositionLU(A)[1]
B = np.array([[7], [12], [3]])
print(tp01.ResolutionLU(L, U, B), "\n")


# Partie 3
# Question 1
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])
print(tp01.GaussChoixPivotPartiel(A, B), "\n")

# Question 2
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])
print(tp01.GaussChoixPivotTotal(A, B))
