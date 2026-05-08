from solveSchrodinger import *
from handleSCF import *

def main():
    # Simple Cubic Helium System
    initialConditions = {
        # 1. Kinetic energy cutoff (Hartree)
        # A value of 10.0 is small enough for testing but captures basic features.
        'ecutwfc': 10.0,

        # 2. Lattice Vectors (Bohr)
        # A 6x6x6 Bohr cube provides enough space to avoid interaction with images.
        'latticeVecs': np.array([
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0]
        ]),

        # 3. Atomic Information (Helium)
        'atomicPositions': np.array([
            [3.0, 3.0, 3.0]  # Center of the cell
        ]),
        'atomicNumbers': np.array([2]),  # Helium
        'atomicMasses': np.array([4.0026]),

        # 4. Calculation Parameters
        'nBand': 1,  # He has 2 electrons, which fill 1 spin-degenerate band
        'bzSetting': "Single",
        'rC': 0.5,  # Potential cutoff radius (Bohr)
    }

    mainSCFLoop(initialConditions)
    return 0

if __name__=="__main__":
    main()