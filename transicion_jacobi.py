import numpy as np

def matriz_transicion_jacobi(A):
    """
    Calcula la matriz de transición T del método de Jacobi para la matriz A.
    
    Parámetros:
    - A: Matriz de coeficientes del sistema de ecuaciones (nxn)
    
    Retorna:
    - T: Matriz de transición del método de Jacobi
    """
    # Extraer la matriz diagonal D
    D = np.diag(np.diag(A))
    
    # Extraer la parte estrictamente inferior L
    L = np.tril(A, -1)
    
    # Extraer la parte estrictamente superior U
    U = np.triu(A, 1)
    
    # Calcular la matriz de transición T = -D^(-1) * (L + U)
    D_inv = np.linalg.inv(D)  # Inversa de la diagonal
    T = -np.dot(D_inv, (L + U))
    
    return T

# Ejemplo de uso: Matriz de coeficientes A
A = np.array([[3, -5, -8], 
              [2, 4, 6], 
              [3, 4, -12]], dtype=float)

# Calcular la matriz de transición T del método de Jacobi
T = matriz_transicion_jacobi(A)
print("Matriz de transición T del método de Jacobi:\n", T)
