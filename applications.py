from src.gauss_method import GaussMethod
from src.lu_method import LUMethod
import numpy as np

gauss = GaussMethod()
lu = LUMethod()

print("Partie 1")
print("Question 1")
print(gauss.GaussReduction(np.array([[2, 5, 6, 7], [4, 11, 9, 12], [-2, -8, 7, 3]])), "\n")

print("Question 2")
print(gauss.SysTriSupResolution(gauss.GaussReduction(np.array([[2, 5, 6, 7], [4, 11, 9, 12], [-2, -8, 7, 3]]))), "\n")

print("Question 3")
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])
print(gauss.Gauss(A, B), "\n")


print("Partie 2")
print("Question 1")
print(lu.LUDecomposition(np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])), "\n")

print("Question 2")
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
L = lu.LUDecomposition(A)[0]
U = lu.LUDecomposition(A)[1]
B = np.array([[7], [12], [3]])
print(lu.LUResolution(L, U, B), "\n")


print("Partie 3")
print("Question 1")
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])
print(gauss.GaussPartialPivotChoice(A, B), "\n")

print("Question 2")
A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])
print(gauss.GaussTotalPivotChoice(A, B))
