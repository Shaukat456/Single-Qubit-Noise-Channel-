"""
Comprehensive Tests for Quantum Noise Channels

These tests verify:
1. Channels are valid CPTP maps
2. Physical effects match theoretical predictions
3. Channel composition works correctly
4. Edge cases are handled properly
"""

import numpy as np
import pytest
from quantum_noise import (
    DensityMatrix,
    AmplitudeDamping,
    PhaseDamping,
    Depolarizing,
    BitFlip,
    PhaseDampingRandom,
    ground_state,
    excited_state,
    plus_state,
    maximally_mixed_state,
    state_vector_to_density_matrix,
    PAULI_X,
    PAULI_Z,
)


class TestDensityMatrix:
    """Test density matrix operations."""
    
    def test_normalization(self):
        """Density matrices must have trace = 1."""
        rho = ground_state()
        assert np.isclose(np.trace(rho.matrix), 1.0)
    
    def test_purity_pure_state(self):
        """Pure states have purity = 1."""
        rho = ground_state()
        assert np.isclose(rho.purity(), 1.0)
    
    def test_purity_mixed_state(self):
        """Maximally mixed state has purity = 0.5."""
        rho = maximally_mixed_state()
        assert np.isclose(rho.purity(), 0.5)
    
    def test_entropy_pure_state(self):
        """Pure states have entropy = 0."""
        rho = ground_state()
        assert np.isclose(rho.entropy(), 0.0, atol=1e-10)
    
    def test_entropy_mixed_state(self):
        """Maximally mixed single qubit has entropy = 1."""
        rho = maximally_mixed_state()
        assert np.isclose(rho.entropy(), 1.0, atol=1e-10)
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        rho = ground_state()
        # <0|Z|0> = 1
        assert np.isclose(rho.expectation(PAULI_Z), 1.0)
        
        rho = excited_state()
        # <1|Z|1> = -1
        assert np.isclose(rho.expectation(PAULI_Z), -1.0)


class TestAmplitudeDamping:
    """Test amplitude damping (energy dissipation) channel."""
    
    def test_cptp(self):
        """Verify amplitude damping is a valid CPTP map."""
        channel = AmplitudeDamping(gamma=0.3)
        assert channel.verify_cptp()
    
    def test_no_noise(self):
        """No noise (gamma=0) leaves state unchanged."""
        rho = ground_state()
        channel = AmplitudeDamping(gamma=0.0)
        rho_out = channel.apply(rho)
        assert np.allclose(rho.matrix, rho_out.matrix)
    
    def test_complete_damping(self):
        """Complete damping (gamma=1) drives to ground state."""
        rho = excited_state()
        channel = AmplitudeDamping(gamma=1.0)
        rho_out = channel.apply(rho)
        assert np.allclose(rho_out.matrix, ground_state().matrix)
    
    def test_ground_state_unchanged(self):
        """Ground state is unaffected by amplitude damping."""
        rho = ground_state()
        channel = AmplitudeDamping(gamma=0.5)
        rho_out = channel.apply(rho)
        assert np.allclose(rho.matrix, rho_out.matrix)
    
    def test_excited_state_decay(self):
        """Excited state decays toward ground state."""
        rho = excited_state()
        gamma = 0.3
        channel = AmplitudeDamping(gamma=gamma)
        rho_out = channel.apply(rho)
        
        # After damping: (1-γ)|1⟩⟨1| + γ|0⟩⟨0|
        expected = (1 - gamma) * excited_state().matrix + gamma * ground_state().matrix
        assert np.allclose(rho_out.matrix, expected)
    
    def test_purity_decrease(self):
        """Amplitude damping reduces purity (creates mixedness)."""
        rho = excited_state()
        channel = AmplitudeDamping(gamma=0.3)
        rho_out = channel.apply(rho)
        
        # Pure excited state → mixed state (lower purity)
        assert rho_out.purity() < rho.purity()


class TestPhaseDamping:
    """Test phase damping (dephasing) channel."""
    
    def test_cptp(self):
        """Verify phase damping is a valid CPTP map."""
        channel = PhaseDamping(lam=0.3)
        assert channel.verify_cptp()
    
    def test_no_noise(self):
        """No noise (lambda=0) leaves state unchanged."""
        rho = plus_state()
        channel = PhaseDamping(lam=0.0)
        rho_out = channel.apply(rho)
        assert np.allclose(rho.matrix, rho_out.matrix)
    
    def test_basis_states_unchanged(self):
        """Basis states |0⟩, |1⟩ unaffected by phase damping."""
        for rho in [ground_state(), excited_state()]:
            channel = PhaseDamping(lam=0.5)
            rho_out = channel.apply(rho)
            assert np.allclose(rho.matrix, rho_out.matrix)
    
    def test_coherence_decay(self):
        """Off-diagonal elements (coherence) decay."""
        rho = plus_state()  # Has off-diagonal elements
        channel = PhaseDamping(lam=0.3)
        rho_out = channel.apply(rho)
        
        # Coherence should decrease
        original_coherence = np.abs(rho.matrix[0, 1])
        final_coherence = np.abs(rho_out.matrix[0, 1])
        assert final_coherence < original_coherence


class TestDepolarizing:
    """Test depolarizing channel (uniform noise)."""
    
    def test_cptp(self):
        """Verify depolarizing is a valid CPTP map."""
        channel = Depolarizing(p=0.3)
        assert channel.verify_cptp()
    
    def test_no_noise(self):
        """No noise (p=0) leaves state unchanged."""
        rho = ground_state()
        channel = Depolarizing(p=0.0)
        rho_out = channel.apply(rho)
        assert np.allclose(rho.matrix, rho_out.matrix)
    
    def test_complete_depolarization(self):
        """Complete noise (p=1) gives maximally mixed state."""
        rho = ground_state()
        channel = Depolarizing(p=1.0)
        rho_out = channel.apply(rho)
        
        # Should be I/2
        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(rho_out.matrix, expected, atol=1e-10)
    
    def test_purity_decrease(self):
        """Depolarizing decreases purity."""
        rho = ground_state()
        channel = Depolarizing(p=0.5)
        rho_out = channel.apply(rho)
        assert rho_out.purity() < rho.purity()
    
    def test_mixture_formula(self):
        """ℰ(ρ) = (1-p)ρ + p(I/2)."""
        rho = plus_state()
        p = 0.3
        channel = Depolarizing(p=p)
        rho_out = channel.apply(rho)
        
        I = np.eye(2, dtype=complex)
        expected = (1 - p) * rho.matrix + p * (I / 2)
        assert np.allclose(rho_out.matrix, expected, atol=1e-10)


class TestBitFlip:
    """Test bit-flip channel."""
    
    def test_cptp(self):
        """Verify bit-flip is a valid CPTP map."""
        channel = BitFlip(p=0.3)
        assert channel.verify_cptp()
    
    def test_no_noise(self):
        """No noise (p=0) leaves state unchanged."""
        rho = ground_state()
        channel = BitFlip(p=0.0)
        rho_out = channel.apply(rho)
        assert np.allclose(rho.matrix, rho_out.matrix)
    
    def test_ground_excited_swap(self):
        """With p=0.5, |0⟩ and |1⟩ mix equally."""
        rho = ground_state()
        channel = BitFlip(p=0.5)
        rho_out = channel.apply(rho)
        
        # Should be (0.5|0⟩⟨0| + 0.5|1⟩⟨1|) = I/2
        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(rho_out.matrix, expected, atol=1e-10)
    
    def test_plus_state_invariant(self):
        """Plus state |+⟩ is eigenstate of X (bit-flip operator)."""
        rho = plus_state()
        channel = BitFlip(p=0.3)
        rho_out = channel.apply(rho)
        # |+⟩ should be unchanged by bit flips
        assert np.allclose(rho.matrix, rho_out.matrix)


class TestPhaseDampingRandom:
    """Test random phase damping channel."""
    
    def test_cptp(self):
        """Verify phase damping random is a valid CPTP map."""
        channel = PhaseDampingRandom(lam=0.3)
        assert channel.verify_cptp()
    
    def test_populations_preserved(self):
        """Populations (diagonal elements) unchanged."""
        rho = plus_state()
        channel = PhaseDampingRandom(lam=0.3)
        rho_out = channel.apply(rho)
        
        # Diagonal elements should be preserved
        assert np.isclose(rho.matrix[0, 0], rho_out.matrix[0, 0])
        assert np.isclose(rho.matrix[1, 1], rho_out.matrix[1, 1])


class TestChannelProperties:
    """Test mathematical properties of channels."""
    
    def test_linearity_amplitude_damping(self):
        """Channels are linear: ℰ(aρ + bσ) = aℰ(ρ) + bℰ(σ)."""
        channel = AmplitudeDamping(gamma=0.3)
        rho1 = ground_state()
        rho2 = excited_state()
        
        # Mixed state as convex combination
        a, b = 0.6, 0.4  # a + b = 1
        rho_mixed = (a * rho1.matrix + b * rho2.matrix)
        rho_mixed = DensityMatrix(rho_mixed)
        
        # Apply channel to mixture
        result_direct = channel.apply(rho_mixed)
        
        # Apply channel to components and mix
        result_composed = (
            a * channel.apply(rho1).matrix + 
            b * channel.apply(rho2).matrix
        )
        result_composed = DensityMatrix(result_composed)
        
        assert np.allclose(result_direct.matrix, result_composed.matrix, atol=1e-10)
    
    def test_sequential_application(self):
        """Can apply channels sequentially."""
        rho = ground_state()
        
        # Apply two channels in sequence
        channel1 = AmplitudeDamping(gamma=0.2)
        channel2 = PhaseDamping(lam=0.1)
        
        rho_out = channel2.apply(channel1.apply(rho))
        
        # Result should be a valid density matrix
        assert np.isclose(np.trace(rho_out.matrix), 1.0)
        assert np.all(np.linalg.eigvalsh(rho_out.matrix) >= -1e-10)
    
    def test_parameter_edge_cases(self):
        """Test channels at parameter boundaries."""
        channels = [
            (AmplitudeDamping(gamma=0.0), AmplitudeDamping(gamma=1.0)),
            (PhaseDamping(lam=0.0), PhaseDamping(lam=1.0)),
            (Depolarizing(p=0.0), Depolarizing(p=1.0)),
            (BitFlip(p=0.0), BitFlip(p=1.0)),
        ]
        
        for ch_min, ch_max in channels:
            # Both should be valid CPTP maps
            assert ch_min.verify_cptp()
            assert ch_max.verify_cptp()


class TestStateTransformations:
    """Test state transformations through channels."""
    
    def test_superposition_decoherence(self):
        """Superposition states become mixed under noise."""
        rho = plus_state()
        
        channel = AmplitudeDamping(gamma=0.3)
        rho_out = channel.apply(rho)
        
        # Purity should decrease (becomes more mixed)
        assert rho_out.purity() < rho.purity()
    
    def test_entropy_increase(self):
        """Noise generally increases entropy."""
        rho = ground_state()
        
        initial_entropy = rho.entropy()
        
        channel = Depolarizing(p=0.3)
        rho_out = channel.apply(rho)
        final_entropy = rho_out.entropy()
        
        # Entropy should increase
        assert final_entropy > initial_entropy


class TestInvalidInputs:
    """Test error handling."""
    
    def test_invalid_gamma(self):
        """Amplitude damping rejects invalid gamma."""
        with pytest.raises(ValueError):
            AmplitudeDamping(gamma=-0.1)
        with pytest.raises(ValueError):
            AmplitudeDamping(gamma=1.1)
    
    def test_invalid_lambda(self):
        """Phase damping rejects invalid lambda."""
        with pytest.raises(ValueError):
            PhaseDamping(lam=-0.1)
        with pytest.raises(ValueError):
            PhaseDamping(lam=1.1)
    
    def test_invalid_p(self):
        """Depolarizing and BitFlip reject invalid p."""
        with pytest.raises(ValueError):
            Depolarizing(p=-0.1)
        with pytest.raises(ValueError):
            Depolarizing(p=1.1)
        with pytest.raises(ValueError):
            BitFlip(p=-0.1)
        with pytest.raises(ValueError):
            BitFlip(p=1.1)


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_channels.py -v
    pytest.main([__file__, "-v"])
