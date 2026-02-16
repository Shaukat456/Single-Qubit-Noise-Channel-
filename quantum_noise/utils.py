"""
Utility functions and helper tools for quantum noise simulation.

Includes visualization, analysis, and helper functions.
"""

import numpy as np
from typing import List, Tuple
from .core import DensityMatrix, PAULI_X, PAULI_Y, PAULI_Z


def bloch_vector(rho: DensityMatrix) -> np.ndarray:
    """
    Convert density matrix to Bloch vector representation.

    For a single qubit, the density matrix can be written as:
    ρ = (I + σ·b)/2

    where σ = (σₓ, σᵧ, σᵤ) and b = (bₓ, bᵧ, bᵤ) is the Bloch vector.

    Parameters:
        rho: Density matrix.

    Returns:
        3D Bloch vector (bₓ, bᵧ, bᵤ).

    Properties:
        - Pure states: ||b|| = 1 (on Bloch sphere surface)
        - Mixed states: ||b|| < 1 (inside Bloch sphere)
        - Maximally mixed: b = (0, 0, 0) (at sphere center)
    """
    bx = rho.expectation(PAULI_X)
    by = rho.expectation(PAULI_Y)
    bz = rho.expectation(PAULI_Z)
    return np.array([bx, by, bz])


def from_bloch_vector(b: np.ndarray) -> DensityMatrix:
    """
    Create density matrix from Bloch vector.

    Parameters:
        b: 3D vector (bₓ, bᵧ, bᵤ) with ||b|| ≤ 1.

    Returns:
        Corresponding density matrix.
    """
    if len(b) != 3:
        raise ValueError(f"Bloch vector must be 3D, got {len(b)}D")

    if np.linalg.norm(b) > 1.0 + 1e-10:
        raise ValueError(
            f"Bloch vector must satisfy ||b|| ≤ 1, got {np.linalg.norm(b)}"
        )

    bx, by, bz = b
    I = np.eye(2, dtype=complex)
    rho = (I + bx * PAULI_X + by * PAULI_Y + bz * PAULI_Z) / 2
    return DensityMatrix(rho)


def fidelity(rho1: DensityMatrix, rho2: DensityMatrix) -> float:
    """
    Compute fidelity between two density matrices.

    Formula: F(ρ,σ) = Tr(√(√ρ σ √ρ))²

    Also: F(ρ,σ) = min(1, Tr(ρσ) + √((1-Tr(ρ²))(1-Tr(σ²))))  (for single qubit)

    Properties:
        - F = 1: Identical states
        - F = 0: Orthogonal states
        - 0 ≤ F ≤ 1 always

    Parameters:
        rho1, rho2: Density matrices to compare.

    Returns:
        Fidelity value in [0, 1].
    """
    # Eigendecomposition of rho1
    evals, evecs = np.linalg.eigh(rho1.matrix)

    # Construct √ρ1
    sqrt_evals = np.sqrt(np.maximum(evals, 0))
    sqrtrho1 = evecs @ np.diag(sqrt_evals) @ evecs.conj().T

    # Compute √ρ1 σ √ρ1
    M = sqrtrho1 @ rho2.matrix @ sqrtrho1

    # Eigenvalues of M
    M_evals = np.linalg.eigvalsh(M)

    # Fidelity = (Tr(√M))²
    fid = (np.sum(np.sqrt(np.maximum(M_evals, 0)))) ** 2
    return float(np.real(fid))


def purity_loss(rho: DensityMatrix) -> float:
    """
    Compute how much purity the state has lost (relative to a pure state).

    Formula: ΔP = 1 - Tr(ρ²)

    Returns:
        0 for pure states, 0.5 for maximally mixed single qubit.
    """
    return 1.0 - rho.purity()


def coherence(rho: DensityMatrix) -> float:
    """
    Measure of coherence (off-diagonal elements).

    L1 norm of off-diagonal elements: Σᵢⱼ |ρᵢⱼ| (i≠j)

    Returns:
        0 for incoherent (diagonal) states.
        Non-zero for states with superposition.
    """
    coh = 0.0
    for i in range(2):
        for j in range(2):
            if i != j:
                coh += np.abs(rho.matrix[i, j])
    return float(coh)


def purity_trajectory(
    initial_rho: DensityMatrix, channels_list: List, num_steps: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track purity as channels are applied repeatedly.

    Useful for studying decoherence over time.

    Parameters:
        initial_rho: Starting density matrix.
        channels_list: List of channels to apply repeatedly.
        num_steps: Number of iterations.

    Returns:
        (steps, purities): Arrays of step numbers and corresponding purities.

    Example:
        channel = AmplitudeDamping(gamma=0.1)
        steps, purities = purity_trajectory(ground_state(), [channel], 50)
        plot(steps, purities)
    """
    purity_history = [initial_rho.purity()]
    current_rho = initial_rho

    for _ in range(num_steps):
        for channel in channels_list:
            current_rho = channel.apply(current_rho)
        purity_history.append(current_rho.purity())

    return np.arange(len(purity_history)), np.array(purity_history)


def characterize_channel(channel, state_list: List[DensityMatrix]) -> None:
    """
    Comprehensive analysis of a channel's effect.

    Prints detailed statistics about how the channel affects different states.

    Parameters:
        channel: Quantum channel to analyze.
        state_list: List of test states (e.g., [|0⟩, |1⟩, |+⟩, mixed]).
    """
    print(f"\nChannel Characterization: {channel}")
    print("=" * 60)

    for i, rho_in in enumerate(state_list):
        rho_out = channel.apply(rho_in)

        print(f"\nState {i+1}:")
        print(f"  Input purity:  {rho_in.purity():.4f}")
        print(f"  Output purity: {rho_out.purity():.4f}")
        print(f"  Purity loss:   {purity_loss(rho_out):.4f}")
        print(f"  Fidelity:      {fidelity(rho_in, rho_out):.4f}")
        print(f"  Entropy:       {rho_out.entropy():.4f} bits")


def channel_fidelity(channel, reference_states: List[DensityMatrix] = None) -> float:
    """
    Average fidelity of a channel against reference states.

    Measures how much the channel preserves states on average.

    Parameters:
        channel: Quantum channel.
        reference_states: States to test (defaults to {|0⟩, |1⟩, |+⟩}).

    Returns:
        Average fidelity.
    """
    if reference_states is None:
        from .core import ground_state, excited_state, plus_state

        reference_states = [ground_state(), excited_state(), plus_state()]

    fidelities = []
    for rho in reference_states:
        rho_out = channel.apply(rho)
        fidelities.append(fidelity(rho, rho_out))

    return float(np.mean(fidelities))


def kraus_rank(channel) -> int:
    """
    Return the number of Kraus operators for this channel.

    Rank 1 channels are unitary (reversible).
    Higher rank = more information loss.
    """
    return len(channel.get_kraus_operators())


def process_matrix(channel, dimensions: int = 2) -> np.ndarray:
    """
    Compute the Choi matrix (process matrix) for a channel.

    For analysis and visualization purposes.

    The Choi matrix is a 4×4 matrix representation of the quantum channel
    in the vectorized density matrix space.

    Parameters:
        channel: Quantum channel.
        dimensions: Hilbert space dimension (2 for single qubit).

    Returns:
        d² × d² Choi matrix.
    """
    d = dimensions

    # Create basis states |i⟩⟨j|
    basis = []
    for i in range(d):
        for j in range(d):
            psi = np.zeros(d, dtype=complex)
            psi[i] = 1
            phi = np.zeros(d, dtype=complex)
            phi[j] = 1
            basis.append(np.outer(psi, np.conj(phi)))

    # Apply channel to each basis state
    choi = np.zeros((d * d, d * d), dtype=complex)
    for idx, rho_basis in enumerate(basis):
        rho_in = DensityMatrix(rho_basis)
        rho_out = channel.apply(rho_in)
        choi[idx, :] = rho_out.matrix.flatten()

    return choi


# Example usage
if __name__ == "__main__":
    from .core import ground_state, excited_state, plus_state, maximally_mixed_state
    from .channels import AmplitudeDamping, PhaseDamping, Depolarizing

    print("Quantum Noise Utilities Example")
    print("=" * 50)

    # Test Bloch vector
    rho = plus_state()
    b = bloch_vector(rho)
    print(f"\nBloch vector of |+⟩: {b}")
    print(f"Bloch vector norm: {np.linalg.norm(b):.4f}")

    # Test fidelity
    rho1 = ground_state()
    rho2 = excited_state()
    print(f"\nFidelity(|0⟩, |1⟩): {fidelity(rho1, rho2):.4f}")

    # Test channel characterization
    channel = AmplitudeDamping(gamma=0.3)
    characterize_channel(
        channel,
        [ground_state(), excited_state(), plus_state(), maximally_mixed_state()],
    )
