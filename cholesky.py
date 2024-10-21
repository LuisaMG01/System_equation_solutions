import numpy as np

def cholesky_factorization(A):
    """
    Realiza la factorización de Cholesky de una matriz A (simétrica y definida positiva).
    
    Parámetros:
    - A: Matriz cuadrada de coeficientes (nxn)
    
    Retorna:
    - U: Matriz triangular superior de la factorización A = U^T * U
    """
    # Verificamos que A sea simétrica y definida positiva
    if not np.allclose(A, A.T):
        raise ValueError("La matriz A no es simétrica.")
    
    # Factorización de Cholesky usando la función numpy
    U = np.linalg.cholesky(A).T  # numpy devuelve la matriz L, pero queremos la triangular superior U, por eso la transponemos
    
    return U

# Ejemplo de matriz A dada por el usuario
A = np.array([[36, 3],
              [3, 5]], dtype=float)

# Realizar la factorización de Cholesky
U = cholesky_factorization(A)

# Imprimir la matriz U y el valor U(1,2)
print("Matriz triangular superior U de la factorización de Cholesky:")
print(U)

# El valor U(1,2) está en la posición (2,1) de la matriz U en Python (índices comienzan en 0)
print(f"El valor U(1,2) es: {U[1,0]}")
