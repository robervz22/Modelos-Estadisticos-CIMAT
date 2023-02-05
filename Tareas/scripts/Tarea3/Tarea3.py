# (Tarea 3 Modelos Estadisticos)
import numpy as np
import sympy as sym

##############
# PROBLEMA 1 #
##############
X=np.array([[1,0],[2,-1],[1,2]])
y1, y2, y3 = sym.symbols('y1,y2,y3')
Y=np.array([y1,y2,y3])
XTX_inv=np.linalg.inv(X.T@X)
print("Los estimadores de theta y phi son: ")
print(XTX_inv@ X.T@ Y)

##############
# PROBLEMA 2 #
##############
X=np.array([[1,-1,1],[1,0,-2],[1,1,1]])
XTX_inv=np.linalg.inv(X.T@X)
print("\n\nLos estimadores de b0, b1 y b2 son: ")
print(XTX_inv@ X.T@ Y)

