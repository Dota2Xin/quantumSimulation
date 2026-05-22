from solveSchrodinger import *
from handleSCF import *
import matplotlib.pyplot as plt




def main():
    #DONE TESTING, MOVE ON TO NON-LOCAL PSEUDOPOTENTIAL NOW
    # Simple Cubic Helium System
    a=10.0

    initialConditions = {
        # 1. Kinetic energy cutoff (Hartree)
        # A value of 10.0 is small enough for testing but captures basic features.
        'ecutwfc': 10.0,

        # 2. Lattice Vectors (Bohr)
        # A 6x6x6 Bohr cube provides enough space to avoid interaction with images.
        'latticeVecs': np.array([
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a]
        ]),

        # 3. Atomic Information (Helium)
        'atomicPositions': np.array([
            [a/2,a/2,a/2]  # Center of the cell
        ]),
        'atomicNumbers': np.array([2]),  # Helium
        'atomicMasses': np.array([4.0026]),

        # 4. Calculation Parameters
        'nBand': 1,  # He has 2 electrons, which fill 1 spin-degenerate band
        'bzSetting': "Single",
        'rC': 0.2,  # Potential cutoff radius (Bohr)
        'tol':1e-2
    }
    #_, currEnergy = mainSCFLoop(initialConditions)
    #return 0
    daArr=np.linspace(-0.1,0.1, 21)
    energies=np.zeros_like(daArr)
    for i in range(len(daArr)):
        da=daArr[i]
        aTemp=a+da
        latticeVecs=np.asarray([
            [aTemp, 0.0, 0.0],
            [0.0, aTemp, 0.0],
            [0.0, 0.0, aTemp]
        ])
        atomicPositions=np.asarray([
            [aTemp/2,aTemp/2,aTemp/2]  # Center of the cell
        ])
        print("SWAG SWAG SWAG")
        initialConditions['latticeVecs']=latticeVecs
        initialConditions['atomicPositions']=atomicPositions
        _, currEnergy= mainSCFLoop(initialConditions)
        energies[i]=currEnergy

    plt.plot(daArr+a, energies)
    plt.show()
    return 0

if __name__=="__main__":
    main()