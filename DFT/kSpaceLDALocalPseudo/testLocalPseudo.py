from blockDavidson import *
from handleSCF import *

def actHamiltonian(vec, *args):
    # args[0] is the hamiltonianArgs dictionary
    hamiltonianArgs = args[0]
    H = hamiltonianArgs['H']
    return H @ vec

def main():
    # 1. Davidson Parameters
    l = 5  # Number of target eigenvalues
    m = 20  # Max subspace size

    # 2. Set up the 3D grid
    ecutwfc = 1.0  # Kept small to keep dense matrix size manageable for the exact solver
    reciprocalVecs = np.eye(3)
    qGridSmall, n1Arr, n2Arr, n3Arr = makeSmallGrid(ecutwfc, reciprocalVecs)

    stateSize = len(n1Arr) * len(n2Arr) * len(n3Arr)
    print(f"State space dimension: {stateSize}")

    # 3. Construct hamiltonianArgs
    k = np.array([0.1, 0.2, 0.3])
    # V needs to be a 3D grid matching the spatial dimensions
    VPotential = np.random.rand(len(n1Arr), len(n2Arr), len(n3Arr))

    hamiltonianArgs = {
        'k': k,
        'V': VPotential
    }

    # 4. Generate the exact dense matrix H to test against.
    # To test convergence reliably, we ensure H has the exact diagonal that
    # getDiagonal calculates, plus some symmetric off-diagonal noise.
    HDiag = getDiagonal(qGridSmall, hamiltonianArgs)

    # Create symmetric noise
    noise = np.random.rand(stateSize, stateSize) * 0.05
    noise = (noise + noise.T) / 2

    # Build H
    H = np.diag(HDiag) + noise

    # Pass the dense H matrix through the args so actHamiltonian can use it
    hamiltonianArgs['H'] = H

    # 5. Run the Block-Davidson algorithm
    print(f"Running Block-Davidson for lowest {l} eigenvalues...")
    daveVal, daveEig = blockDavidson(l, m, stateSize, qGridSmall, actHamiltonian, hamiltonianArgs)

    # 6. Run the exact solver for comparison
    print("Running Exact Eigensolver...")
    realVal, realEig = getExact(H)

    # 7. Output results
    print("\n--- Results ---")
    print("Block-Davidson Eigenvalues:")
    print(daveVal)
    print("\nExact Eigenvalues:")
    print(realVal[:l])

    # Print the maximum absolute error for quick verification
    diff = np.abs(daveVal - realVal[:l])
    print(f"\nMax difference: {np.max(diff):.2e}")

    return 0


if __name__ == "__main__":
    main()