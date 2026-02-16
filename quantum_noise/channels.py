"""
Single-Qubit Quantum Noise Channels

This module implements quantum noise channels using Kraus operators.
Each channel is a Completely Positive, Trace-Preserving (CPTP) map.

Theory:
-------
A quantum channel ℰ transforms density matrices via Kraus operators:

    ℰ(ρ) = Σᵢ Kᵢ ρ Kᵢ†  (Kraus representation)

where Σᵢ Kᵢ† Kᵢ = I (trace-preserving condition)

Each Kᵢ represents a possible physical process outcome.
The sum over all outcomes recovers the full evolution.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from .core import DensityMatrix


class QuantumChannel(ABC):
    """
    Abstract base class for quantum channels.
    
    A quantum channel is a linear map that transforms density matrices.
    It must be:
    - Linear: ℰ(aρ + bσ) = aℰ(ρ) + bℰ(σ)
    - Completely Positive: Preserves positive-semidefiniteness
    - Trace-Preserving: Tr(ℰ(ρ)) = Tr(ρ) = 1
    """
    
    @abstractmethod
    def get_kraus_operators(self) -> List[np.ndarray]:
        """
        Return the Kraus operators {K₁, K₂, ...} for this channel.
        
        Returns:
            List of 2x2 Kraus operator matrices.
        """
        pass
    
    def apply(self, rho: DensityMatrix) -> DensityMatrix:
        """
        Apply the quantum channel to a density matrix.
        
        Formula: ℰ(ρ) = Σᵢ Kᵢ ρ Kᵢ†
        
        Parameters:
            rho: Input density matrix.
            
        Returns:
            Transformed density matrix.
        """
        kraus_ops = self.get_kraus_operators()
        rho_out = np.zeros((2, 2), dtype=complex)
        
        for K in kraus_ops:
            K_dag = np.conj(K.T)
            rho_out += K @ rho.matrix @ K_dag
        
        return DensityMatrix(rho_out)
    
    def verify_cptp(self, tolerance: float = 1e-10) -> bool:
        """
        Verify this is a valid CPTP map.
        
        Checks:
        1. Σᵢ Kᵢ† Kᵢ = I (trace preservation)
        
        Parameters:
            tolerance: Numerical tolerance for verification.
            
        Returns:
            True if channel is CPTP, False otherwise.
        """
        kraus_ops = self.get_kraus_operators()
        
        # Check Σᵢ Kᵢ† Kᵢ = I
        sum_ops = np.zeros((2, 2), dtype=complex)
        for K in kraus_ops:
            K_dag = np.conj(K.T)
            sum_ops += K_dag @ K
        
        identity = np.eye(2, dtype=complex)
        return np.allclose(sum_ops, identity, atol=tolerance)


class AmplitudeDamping(QuantumChannel):
    """
    Amplitude Damping Channel (Generational Damping)
    
    **Physical Interpretation**:
    Energy dissipation to the environment (T₁ relaxation).
    Excited state decays to ground state at rate γ.
    
    **Kraus Operators**:
    K₀ = [[1,     0    ],
          [0,  √(1-γ)]]
    
    K₁ = [[0, √γ],
          [0,  0 ]]
    
    **Effect on basis states**:
    - ℰ(|0⟩⟨0|) = |0⟩⟨0|  (ground state unchanged)
    - ℰ(|1⟩⟨1|) = (1-γ)|1⟩⟨1| + γ|0⟩⟨0|  (decay to ground)
    - ℰ(|+⟩⟨+|) = mixture (superposition decays)
    
    **Parameters**:
        gamma: Damping parameter (0 ≤ γ ≤ 1)
               - γ = 0: No noise
               - γ = 1: Complete decay to ground state
    """
    
    def __init__(self, gamma: float):
        """
        Initialize amplitude damping channel.
        
        Parameters:
            gamma: Damping rate (0 to 1).
        """
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """
        Get Kraus operators for amplitude damping.
        
        Returns:
            List of 2 Kraus operators: [K₀, K₁]
        """
        # K₀ = [[1, 0], [0, √(1-γ)]]
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - self.gamma)]
        ], dtype=complex)
        
        # K₁ = [[0, √γ], [0, 0]]
        K1 = np.array([
            [0, np.sqrt(self.gamma)],
            [0, 0]
        ], dtype=complex)
        
        return [K0, K1]
    
    def __repr__(self) -> str:
        return f"AmplitudeDamping(γ={self.gamma:.4f})"


class PhaseDamping(QuantumChannel):
    """
    Phase Damping Channel (Pure Dephasing)
    
    **Physical Interpretation**:
    Loss of quantum coherence without energy dissipation (T₂ relaxation).
    Off-diagonal elements of density matrix decay.
    
    **Kraus Operators**:
    K₀ = [[1,     0    ],
          [0,  √(1-λ)]]
    
    K₁ = [[0,  0 ],
          [0, √λ]]
    
    **Effect on basis states**:
    - ℰ(|0⟩⟨0|) = |0⟩⟨0|  (unaffected)
    - ℰ(|1⟩⟨1|) = |1⟩⟨1|  (unaffected)
    - ℰ(|+⟩⟨+|) = mixture of |0⟩ and |1⟩  (coherence lost)
    
    **Difference from Amplitude Damping**:
    - Amplitude damping: Energy loss (|1⟩ → |0⟩)
    - Phase damping: Coherence loss (no preferred direction)
    
    **Parameters**:
        lam: Dephasing parameter (0 ≤ λ ≤ 1)
             - λ = 0: Perfect coherence
             - λ = 1: Complete dephasing to incoherent mixture
    """
    
    def __init__(self, lam: float):
        """
        Initialize phase damping channel.
        
        Parameters:
            lam: Dephasing rate (0 to 1).
        """
        if not (0 <= lam <= 1):
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = lam
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """
        Get Kraus operators for phase damping.
        
        Returns:
            List of 2 Kraus operators: [K₀, K₁]
        """
        # K₀ = [[1, 0], [0, √(1-λ)]]
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - self.lam)]
        ], dtype=complex)
        
        # K₁ = [[0, 0], [0, √λ]]
        K1 = np.array([
            [0, 0],
            [0, np.sqrt(self.lam)]
        ], dtype=complex)
        
        return [K0, K1]
    
    def __repr__(self) -> str:
        return f"PhaseDamping(λ={self.lam:.4f})"


class Depolarizing(QuantumChannel):
    """
    Depolarizing Channel
    
    **Physical Interpretation**:
    Random, uniform noise. With probability p, the state is replaced with
    the maximally mixed state I/2. This models general "bit-flip + phase-flip" errors.
    
    **Kraus Operators** (general single-qubit depolarizing):
    K₀ = √(1 - 3p/4) * I
    K₁ = √(p/4) * σₓ
    K₂ = √(p/4) * σᵧ
    K₃ = √(p/4) * σᵤ
    
    Where σₓ, σᵧ, σᵤ are Pauli matrices.
    
    **Effect**:
    - p = 0: No noise, ℰ(ρ) = ρ
    - p = 1: Complete depolarization, ℰ(ρ) = I/2 (maximally mixed)
    - 0 < p < 1: Partial depolarization
    
    **Why "Depolarizing"?**:
    The output state is a mixture of the input and the maximally mixed state:
    ℰ(ρ) = (1 - p)ρ + p(I/2)
    
    The "polarization" (coherence) of the state decreases.
    
    **Parameters**:
        p: Depolarization parameter (0 ≤ p ≤ 1)
    """
    
    def __init__(self, p: float):
        """
        Initialize depolarizing channel.
        
        Parameters:
            p: Depolarization rate (0 to 1).
        """
        if not (0 <= p <= 1):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """
        Get Kraus operators for depolarizing channel.
        
        Returns:
            List of 4 Kraus operators: [K₀, K₁, K₂, K₃]
        """
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # K₀ = √(1 - 3p/4) * I
        K0 = np.sqrt(1 - 3 * self.p / 4) * I
        
        # K₁ = √(p/4) * σₓ
        K1 = np.sqrt(self.p / 4) * X
        
        # K₂ = √(p/4) * σᵧ
        K2 = np.sqrt(self.p / 4) * Y
        
        # K₃ = √(p/4) * σᵤ
        K3 = np.sqrt(self.p / 4) * Z
        
        return [K0, K1, K2, K3]
    
    def __repr__(self) -> str:
        return f"Depolarizing(p={self.p:.4f})"


class BitFlip(QuantumChannel):
    """
    Bit-Flip Channel
    
    **Physical Interpretation**:
    With probability p, the qubit flips: |0⟩ ↔ |1⟩.
    This models single bit-flip errors due to random quantum jumps.
    
    **Kraus Operators**:
    K₀ = √(1-p) * I
    K₁ = √p * σₓ
    
    Where σₓ = [[0, 1], [1, 0]] is the Pauli-X matrix (bit-flip operator).
    
    **Effect on basis states**:
    - ℰ(|0⟩⟨0|) = (1-p)|0⟩⟨0| + p|1⟩⟨1|
    - ℰ(|1⟩⟨1|) = p|0⟩⟨0| + (1-p)|1⟩⟨1|
    - ℰ(|±⟩⟨±|) = |±⟩⟨±|  (eigenstates of X unchanged)
    
    **Key difference from Depolarizing**:
    - Bit-flip: Only X component of error
    - Depolarizing: All three Pauli errors equally
    
    **Parameters**:
        p: Bit-flip probability (0 ≤ p ≤ 1)
    """
    
    def __init__(self, p: float):
        """
        Initialize bit-flip channel.
        
        Parameters:
            p: Bit-flip probability (0 to 1).
        """
        if not (0 <= p <= 1):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """
        Get Kraus operators for bit-flip channel.
        
        Returns:
            List of 2 Kraus operators: [K₀, K₁]
        """
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # K₀ = √(1-p) * I
        K0 = np.sqrt(1 - self.p) * I
        
        # K₁ = √p * σₓ
        K1 = np.sqrt(self.p) * X
        
        return [K0, K1]
    
    def __repr__(self) -> str:
        return f"BitFlip(p={self.p:.4f})"


class PhaseDampingRandom(QuantumChannel):
    """
    Random Phase Damping (Dephasing on Both Axes)
    
    **Physical Interpretation**:
    Combination of phase damping on |±⟩ and |±ᵢ⟩ bases.
    Models random phase flip without affecting populations.
    
    **Kraus Operators**:
    K₀ = √(1-λ) * I
    K₁ = √λ * Z
    
    Where Z is the phase-flip operator (Pauli-Z).
    
    **Effect**:
    - Off-diagonal elements decay by √(1-λ)
    - Populations (diagonal) unchanged
    - ℰ(ρ) = (1-λ)ρ + λ ZρZ
    
    **Parameters**:
        lam: Phase flip probability (0 ≤ λ ≤ 1)
    """
    
    def __init__(self, lam: float):
        """
        Initialize random phase damping channel.
        
        Parameters:
            lam: Phase flip probability (0 to 1).
        """
        if not (0 <= lam <= 1):
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = lam
    
    def get_kraus_operators(self) -> List[np.ndarray]:
        """
        Get Kraus operators for random phase damping.
        
        Returns:
            List of 2 Kraus operators: [K₀, K₁]
        """
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # K₀ = √(1-λ) * I
        K0 = np.sqrt(1 - self.lam) * I
        
        # K₁ = √λ * Z
        K1 = np.sqrt(self.lam) * Z
        
        return [K0, K1]
    
    def __repr__(self) -> str:
        return f"PhaseDampingRandom(λ={self.lam:.4f})"


def compose_channels(channel1: QuantumChannel, channel2: QuantumChannel) -> QuantumChannel:
    """
    Compose two quantum channels sequentially.
    
    The composition ℰ ∘ ℰ' applies channel1 first, then channel2:
    (ℰ ∘ ℰ')(ρ) = ℰ(ℰ'(ρ))
    
    Parameters:
        channel1: First channel to apply.
        channel2: Second channel to apply.
        
    Returns:
        Composition (not returned as a formal channel, use apply sequentially).
        
    Note:
        This is a convenience function. To apply multiple channels:
        rho_new = channel2.apply(channel1.apply(rho))
    """
    pass  # This is shown for completeness; actual use is shown above
