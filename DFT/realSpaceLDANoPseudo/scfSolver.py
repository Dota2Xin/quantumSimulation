from solveSchrodinger import *

def scfLoop():
    return 0

def calcEnergy(densityOld, eigenstates, kineticOperator, externalPotential, hartreePotential, exchangeCorrelation):
    T=0
    densityNew=np.zeros_like(densityOld)
    for i in range(len(eigenstates)):
        T+=np.dot(eigenstates[i], kineticOperator@eigenstates[i])
        densityNew+=eigenstates[i]**2.0


    return 0
