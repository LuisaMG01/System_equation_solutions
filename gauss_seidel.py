import numpy as np

def gauss_seidel(A, b, x0, tol=1e-5, max_iterations=100):
    """
    Método de Gauss-Seidel para resolver el sistema Ax = b.
    
    Parámetros:
    - A: matriz de coeficientes
    - b: vector de términos independientes
    - x0: valor inicial
    - tol: tolerancia para el error
    - max_iterations: número máximo de iteraciones
    
    Retorna:
    - x: solución del sistema
    - iteraciones: número de iteraciones realizadas
    """
    n = len(b)
    x = x0.copy()
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        
        # Actualizar cada variable
        for i in range(n):
            sum_ = b[i]
            for j in range(n):
                if j != i:
                    sum_ -= A[i][j] * x[j]
            x[i] = sum_ / A[i][i]
        
        # Calcular el error
        error = np.linalg.norm(x - x_old, ord=np.inf)
        
        if error < tol:
            return x, iteration + 1
    
    return x, max_iterations

# Definir la matriz de coeficientes A y el vector b
A = np.array([[3, -5, -8],
              [2, 4, 6],
              [3, 4, -12]], dtype=float)

b = np.array([-15, 12, 8], dtype=float)

# Valor inicial
x0 = np.array([1, 1, 1], dtype=float)

# Resolver el sistema
solucion, iteraciones = gauss_seidel(A, b, x0)

# Imprimir el resultado
print(f"Solución: {solucion}")
print(f"Número de iteraciones: {iteraciones}")
print(f"Valor de x1: {solucion[0]}")
