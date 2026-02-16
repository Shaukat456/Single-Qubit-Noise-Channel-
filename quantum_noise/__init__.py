"""
Quantum Noise Simulation Library

A from-scratch implementation of single-qubit quantum noise channels using only NumPy.
No built-in quantum libraries (Qiskit, Cirq, QuTiP) are used.

Main exports:
- DensityMatrix: Quantum state representation
- Channels: AmplitudeDamping, PhaseDamping, Depolarizing, BitFlip
- State helpers: ground_state, excited_state, plus_state, maximally_mixed_state
"""

from .core import (
    DensityMatrix,
    state_vector_to_density_matrix,
    maximally_mixed_state,
    ground_state,
    excited_state,
    plus_state,
    apply_unitary,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    IDENTITY,
)

from .channels import (
    QuantumChannel,
    AmplitudeDamping,
    PhaseDamping,
    Depolarizing,
    BitFlip,
    PhaseDampingRandom,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "DensityMatrix",
    # State creation
    "ground_state",
    "excited_state",
    "plus_state",
    "maximally_mixed_state",
    "state_vector_to_density_matrix",
    # Channels
    "QuantumChannel",
    "AmplitudeDamping",
    "PhaseDamping",
    "Depolarizing",
    "BitFlip",
    "PhaseDampingRandom",
    # Utilities
    "apply_unitary",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "IDENTITY",
]
