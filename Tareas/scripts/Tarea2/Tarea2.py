# (Tarea 2 Modelos Estadisticos)
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statistics as st
from scipy.stats import t

##############
# PROBLEMA 2 #
##############

# Datos Pesos vs Frecuencias
X=np.array([90,86,67,89,81,75])
Y=np.array([62,45,40,55,64,53])

print("PROBLEMA 2:\n")

# Inciso a)
figA= plt.figure()
ax=figA.add_subplot(111)
ax.set_xticks(np.arange(min(X)-1,max(X)+1,2))
ax.set_yticks(np.arange(min(Y)-1,max(Y)+1,2))
ax.plot(X,Y,'^r')
plt.xlabel("Peso (kg)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
figA.savefig("GraficaDatos.png")

# Inciso b)
b1 =lambda X,Y: sum((X-np.mean(X,dtype=np.float64))*(Y-np.mean(Y,dtype=np.float64)))/sum((X-np.mean(X,dtype=np.float64))**2)
b0=lambda X,Y,p:np.mean(Y,dtype=np.float64)-p*np.mean(X,dtype=np.float64)
estY=lambda X,b0,b1:b0+b1*X

print("Los coeficientes son b0=%.4f y b1=%.4f" %(b0(X,Y,b1(X,Y)),b1(X,Y)))

pendiente=b1(X,Y)
intercepto=b0(X,Y,pendiente)

figB=plt.figure()
ax=figB.add_subplot(111)
ax.set_xticks(np.arange(min(X)-1,max(X)+1,2))
ax.set_yticks(np.arange(min(Y)-1,max(Y)+1,2))
ax.plot(X,Y,'^r')
domX=np.arange(min(X),max(X)+1)
ax.plot(domX,estY(domX,intercepto,pendiente),color='slateblue')
hatY=estY(X,intercepto,pendiente)
ax.plot(X,hatY,'ob')
plt.xlabel("Peso (kg)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
figB.savefig('lineaRegress.png')


# Inciso c)
Xprima=np.delete(X,2)
Yprima=np.delete(Y,2)
print("\nLos coeficientes son b0=%.4f y b1=%.4f (quitando (67,40))" %(b0(Xprima,Yprima,b1(Xprima,Yprima)),b1(Xprima,Yprima)))


est88=estY(88,intercepto,pendiente) # Respuesta media

def mssRes(X,Y):
    beta1=b1(X,Y)
    beta0=b0(X,Y,beta1)
    hatY=estY(X,beta0,beta1)
    return sum((Y-hatY)**2)/(len(X)-2)

# Varianza de estimacion y prediccion
def varY(x,X,Y,i):
    if i==1: # Estimacion
        aux=(x-np.mean(X,dtype=np.float64))**2/(sum((X-np.mean(X,dtype=np.float64))**2))
        return (1/len(X)+aux)*mssRes(X,Y)
    if i==2: # Prediccion
        aux=(x-np.mean(X,dtype=np.float64))**2/(sum((X-np.mean(X,dtype=np.float64))**2))
        return (1+1/len(X)+aux)*mssRes(X,Y)

# Intervalos de confianza de 95%
def confIntervalY(x,X,Y,alpha,i): # i=1 Estimacion, i=2 Prediccion
    hatY=estY(x,b0(X,Y,b1(X,Y)),b1(X,Y))
    desv=mt.sqrt(varY(x,X,Y,i))
    aux=t.interval(1-alpha,len(X)-2)
    cfInterval=[hatY+aux[0]*desv,hatY+aux[1]*desv]
    return cfInterval

# Respuesta al inciso d)
print("\nLa estimacion en X=88 es: %.4f" %(est88))
print("\nEl intervalo de confianza para la estimacion es: ", confIntervalY(88,X,Y,0.05,1))

# Respuesta al inciso e)
print("\nEl intervalo de confianza para la prediccion es: ", confIntervalY(88,X,Y,0.05,2))

##############
# PROBLEMA 3 #
##############

from scipy.stats.stats import pearsonr 
print("\n\nPROBLEMA 3:\n")

hatY=estY(X,b0(X,Y,b1(X,Y)),b1(X,Y))

#inciso a)
print("La correlacion entre Xi y Yi es: %.4f" %(pearsonr(X,Y)[0]))

#inciso b)
print("\nLa correlacion entre Yi y hatYi es: %.4f" %(pearsonr(Y,hatY)[0]))

#inciso c)
print("\nLa correlacion entre X y hatYi es: %.4f" %(pearsonr(X,hatY)[0]))

# Coeficiente de determinación
R2=((pendiente**2)*sum((X-np.mean(X,dtype=np.float64))**2))/sum((Y-np.mean(Y,dtype=np.float64))**2)
print("\n\nEl coeficiente de determinacion es: %.4f" %(R2))
print("\nLa correlacion de Xi y Yi al cuadrado es: %.4f" %(pearsonr(X,Y)[0]**2))

##############
# PROBLEMA 8 #
##############

print("\n\nPROBLEMA 8:\n")

Xoz=np.array([0.02,0.07,0.11,0.15])
Yred=np.array([242,237,231,201])

def mssReg(X,Y):
    value=(b1(X,Y)**2)*(sum((X-np.mean(X,dtype=np.float64))**2))
    return value

def varb0(X,Y):
    sigma2=mssRes(X,Y)
    aux=((np.mean(X,dtype=np.float64))**2)/sum((X-np.mean(X,dtype=np.float64))**2)
    value=(1/len(X)+aux)*sigma2
    return value

def varb1(X,Y):
    sigma2=mssRes(X,Y)
    value=sigma2/sum((X-np.mean(X,dtype=np.float64))**2)
    return value
# Regresion sin centrar
print("\nModelo sin centrar:")

print("\nIntercepto: %.4f, Pendiente: %.4f" %(b0(Xoz,Yred,b1(Xoz,Yred)),b1(Xoz,Yred)))

print("\nSS(Res): %.4f, SS(Reg): %.4f" %((len(Xoz)-2)*mssRes(Xoz,Yred),mssReg(Xoz,Yred)))

print("\nMSS(Res): %.4f, MSS(Reg): %.4f" %(mssRes(Xoz,Yred),mssReg(Xoz,Yred)))

print("\nVarianza intercepto: %.4f, Varianza pendiente: %.4f" %(varb0(Xoz,Yred),varb1(Xoz,Yred))) 

# Regresion centrada
XozC=Xoz-np.mean(Xoz,dtype=np.float64)

print("\n\nModelo centrado:")

print("\nIntercepto: %.4f, Pendiente: %.4f" %(b0(XozC,Yred,b1(XozC,Yred)),b1(XozC,Yred)))

print("\nSS(Res): %.4f, SS(Reg): %.4f" %((len(XozC)-2)*mssRes(XozC,Yred),mssReg(XozC,Yred)))

print("\nMSS(Res): %.4f, MSS(Reg): %.4f" %(mssRes(XozC,Yred),mssReg(XozC,Yred)))

print("\nVarianza intercepto: %.4f, Varianza pendiente: %.4f" %(varb0(XozC,Yred),varb1(XozC,Yred))) 

