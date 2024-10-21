import numpy as np

def lu_decomposition(A):
    """
    Realiza la factorización LU de la matriz A usando el método de Doolittle.
    
    Parámetros:
    - A: Matriz de entrada (nxn)
    
    Retorna:
    - L: Matriz triangular inferior
    - U: Matriz triangular superior
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Proceso de Doolittle
    for i in range(n):
        # Calcular los elementos de U
        for j in range(i, n):
            U[i][j] = A[i][j] - np.sum(L[i, :i] * U[:i, j])

        # Calcular los elementos de L
        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - np.sum(L[j, :i] * U[:i, i])) / U[i][i]

    # Diagonal de L es 1
    for i in range(n):
        L[i][i] = 1

    return L, U

# Definir la matriz A
A = np.array([[36, 3, -44, 5],
              [-5, -45, 10, -21],
              [6, 82, 57, 5],
              [12, 3, -8, -42]], dtype=float)

# Realizar la factorización LU
L, U = lu_decomposition(A)

# Mostrar los resultados
print("Matriz L:")
print(L)
print("\nMatriz U:")
print(U)

# Obtener el valor de U(2,3) (tercera columna, segunda fila)
u_23 = U[1, 2]
print(f"\nValor de U(2,3): {u_23}")
