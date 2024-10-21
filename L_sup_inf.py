import numpy as np

def resolver_sistema(L, b, tipo='inferior'):
    """
    Resuelve el sistema Lz = b donde L es una matriz triangular (inferior o superior)
    y b es un vector dado.
    
    Parámetros:
    - L: Matriz triangular (inferior o superior), de tamaño nxn
    - b: Vector solución de tamaño n
    - tipo: 'inferior' para triangular inferior (sustitución progresiva)
            'superior' para triangular superior (sustitución regresiva)
    
    Retorna:
    - z: Vector solución del sistema
    """
    n = len(b)
    z = np.zeros_like(b, dtype=float)
    
    if tipo == 'inferior':
        # Sustitución progresiva
        for i in range(n):
            suma = sum(L[i, j] * z[j] for j in range(i))
            z[i] = (b[i] - suma) / L[i, i]
    elif tipo == 'superior':
        # Sustitución regresiva
        for i in range(n-1, -1, -1):
            suma = sum(L[i, j] * z[j] for j in range(i+1, n))
            z[i] = (b[i] - suma) / L[i, i]
    else:
        raise ValueError("El tipo de matriz debe ser 'inferior' o 'superior'.")
    
    return z

# Ejemplo de uso
L_inferior = np.array([[2, 0, 0], 
                       [3, 1, 0], 
                       [1, -2, 1]], dtype=float)

L_superior = np.array([[2, 3, 1], 
                       [0, 1, -2], 
                       [0, 0, 1]], dtype=float)

b = np.array([2, 4, 5], dtype=float)

# Resolver para matriz triangular inferior
z_inferior = resolver_sistema(L_inferior, b, tipo='inferior')
print("Solución para matriz triangular inferior:", z_inferior)

# Resolver para matriz triangular superior
z_superior = resolver_sistema(L_superior, b, tipo='superior')
print("Solución para matriz triangular superior:", z_superior)
