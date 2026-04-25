import numpy as np
from readPseudopotentials import *


def kineticEnergyCalcsVE(atoms, dicts, maxOrbitals):
    Tbar=np.zeros((len(atoms), maxOrbitals, maxOrbitals))

    for i in range(len(atoms)):
        dict=dicts[i]
        waveFuncs, lValues, mValues, indexDict=getWavefunctionsVE(dict)
        rGrid=getRadius(dict)
        drGrid=getRadiusDiff(dict)
        for j in range(len(mValues)):
            for k in range(len(mValues)):
                Tbar[i][j][k]=kineticEnergyIntegral(waveFuncs[indexDict[j]], mValues[j], lValues[indexDict[j]], waveFuncs[indexDict[k]], mValues[k], lValues[indexDict[k]], rGrid, drGrid)

    return Tbar

def kineticEnergyCalcsAE(atoms, dicts, maxOrbitals):
    Tbar=np.zeros((len(atoms), maxOrbitals, maxOrbitals))

    for i in range(len(atoms)):
        dict=dicts[i]
        waveFuncs, lValues, mValues, indexDict=getWavefunctionsAE(dict)
        rGrid=getRadius(dict)
        drGrid=getRadiusDiff(dict)
        for j in range(len(mValues)):
            for k in range(len(mValues)):
                Tbar[i][j][k]=kineticEnergyIntegral(waveFuncs[indexDict[j]], mValues[j], lValues[indexDict[j]], waveFuncs[indexDict[k]], mValues[k], lValues[indexDict[k]], rGrid, drGrid)

    return Tbar

def kineticEnergyIntegral(content1,m1, l1,content2,m2, l2, rGrid, drGrid):
    if m1!=m2 or l1!=l2:
        return 0
    integrand2=(content1/rGrid)*(content2/rGrid)*(l1*l1)*(l1+1)*(l1+1)
    term2=np.sum(integrand2*drGrid)

    integrand1=(np.diff(content1)/np.diff(rGrid))*(np.diff(content2)/np.diff(rGrid))
    term1=np.sum(integrand1*np.diff(rGrid))
    return 0.5*(term1+term2)

