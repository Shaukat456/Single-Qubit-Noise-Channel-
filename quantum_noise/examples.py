"""
Usage Examples for Quantum Noise Channels

Demonstrates common patterns and workflows.
"""

import numpy as np
from quantum_noise import (
    # State creation
    ground_state,
    excited_state,
    plus_state,
    maximally_mixed_state,
    state_vector_to_density_matrix,
    # Channels
    AmplitudeDamping,
    PhaseDamping,
    Depolarizing,
    BitFlip,
    # Utilities
    bloch_vector,
    from_bloch_vector,
    fidelity,
    coherence,
    purity_trajectory,
    characterize_channel,
)


# ============================================================================
# Example 1: Very Basic - Apply a single channel
# ============================================================================


def example_1_basic():
    """Simple example: apply amplitude damping to ground state."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Channel Application")
    print("=" * 70)

    # Create initial state
    rho = ground_state()
    print(f"\nInitial state: |0⟩ (ground state)")
    print(f"Purity: {rho.purity():.4f} (1.0 = pure state)")
    print(f"Entropy: {rho.entropy():.4f} bits (0.0 = pure state)")

    # Create a noise channel
    gamma = 0.3  # Energy loss parameter
    channel = AmplitudeDamping(gamma=gamma)
    print(f"\nApplying amplitude damping with γ = {gamma}")
    print("(Models energy dissipation to environment)")

    # Apply channel
    rho_noisy = channel.apply(rho)
    print(f"\nAfter noise:")
    print(f"Purity: {rho_noisy.purity():.4f} (decreased from original)")
    print(f"Entropy: {rho_noisy.entropy():.4f} bits")

    # Ground state is special - should be unaffected
    print(f"\nNote: |0⟩ is unaffected by amplitude damping (lowest energy)")


# ============================================================================
# Example 2: Compare different noise channels
# ============================================================================


def example_2_compare_channels():
    """Compare effects of different channels on the same initial state."""
    print("\n" + "=" * 70)
    print("Example 2: Comparing Different Noise Channels")
    print("=" * 70)

    # Start with exciting state (vulnerable to noise)
    rho = excited_state()
    print(f"\nInitial state: |1⟩ (excited state)")
    print(f"Initial purity: {rho.purity():.4f}")

    # Define different channels
    channels = {
        "Amplitude Damping (γ=0.3)": AmplitudeDamping(gamma=0.3),
        "Phase Damping (λ=0.3)": PhaseDamping(lam=0.3),
        "Depolarizing (p=0.3)": Depolarizing(p=0.3),
        "Bit-Flip (p=0.3)": BitFlip(p=0.3),
    }

    print(f"\nApplying noise channels (all with ~0.3 strength):\n")
    print(f"{'Channel':<30} {'Final Purity':<15} {'Entropy':<15}")
    print("-" * 60)

    for name, channel in channels.items():
        rho_out = channel.apply(rho)
        print(f"{name:<30} {rho_out.purity():<15.4f} {rho_out.entropy():<15.4f}")


# ============================================================================
# Example 3: Sequential application (time evolution under noise)
# ============================================================================


def example_3_sequential_channels():
    """Apply channels sequentially to simulate decoherence over time."""
    print("\n" + "=" * 70)
    print("Example 3: Sequential Channel Application (Decoherence)")
    print("=" * 70)

    # Start with superposition
    rho = plus_state()
    print(f"\nInitial state: |+⟩ = (|0⟩ + |1⟩)/√2 (superposition)")
    print(f"Initial purity: {rho.purity():.4f}")
    print(f"Initial coherence: {coherence(rho):.4f}")

    # Simulate repeated application of noise
    channel = PhaseDamping(lam=0.2)
    print(f"\nApplying phase damping (λ=0.2) repeatedly:")

    print(f"\n{'Step':<6} {'Purity':<12} {'Coherence':<12} {'Entropy':<12}")
    print("-" * 45)

    for step in range(6):
        print(
            f"{step:<6} {rho.purity():<12.4f} {coherence(rho):<12.4f} {rho.entropy():<12.4f}"
        )
        rho = channel.apply(rho)

    print(f"\nNotice: Coherence decays to zero (superposition destroyed)")
    print(f"         Purity decreases (state becomes more mixed)")


# ============================================================================
# Example 4: Use Bloch vectors (geometric representation)
# ============================================================================


def example_4_bloch_vectors():
    """Visualize quantum states on the Bloch sphere."""
    print("\n" + "=" * 70)
    print("Example 4: Bloch Vector Representation")
    print("=" * 70)

    states = {
        "|0⟩": ground_state(),
        "|1⟩": excited_state(),
        "|+⟩": plus_state(),
        "Mixed": maximally_mixed_state(),
    }

    print(f"\n{'State':<10} {'Bloch Vector (bx, by, bz)':<35} {'||b||':<10}")
    print("-" * 55)

    for name, rho in states.items():
        b = bloch_vector(rho)
        norm = np.linalg.norm(b)
        print(f"{name:<10} ({b[0]:7.4f}, {b[1]:7.4f}, {b[2]:7.4f})     {norm:<10.4f}")

    print(f"\nKey insight:")
    print(f"  Pure states: ||b|| = 1.0 (on Bloch sphere surface)")
    print(f"  Mixed states: ||b|| < 1.0 (inside Bloch sphere)")
    print(f"  Maximally mixed: ||b|| = 0 (sphere center)")


# ============================================================================
# Example 5: Measure fidelity (state similarity)
# ============================================================================


def example_5_fidelity():
    """Measure how similar states are using fidelity."""
    print("\n" + "=" * 70)
    print("Example 5: Fidelity (State Similarity Measure)")
    print("=" * 70)

    rho_ideal = ground_state()

    print(f"\nIdeal state: |0⟩")
    print(f"\nApplying noise and measuring fidelity:\n")

    # Different amounts of noise
    gammas = [0.0, 0.1, 0.2, 0.5, 1.0]

    print(f"{'γ (damping)':<15} {'Fidelity(|0⟩, noisy)':<25}")
    print("-" * 40)

    for gamma in gammas:
        channel = AmplitudeDamping(gamma=gamma)
        rho_noisy = channel.apply(rho_ideal)
        f = fidelity(rho_ideal, rho_noisy)
        print(f"{gamma:<15.2f} {f:<25.4f}")

    print(f"\nInterpretation:")
    print(f"  F = 1.0: Identical states (no error)")
    print(f"  F = 0.5: Significant error")
    print(f"  F = 0.0: Completely different (orthogonal)")


# ============================================================================
# Example 6: Purity as noise indicator
# ============================================================================


def example_6_purity_tracking():
    """Track purity loss over repeated applications."""
    print("\n" + "=" * 70)
    print("Example 6: Purity Degradation Under Repeated Noise")
    print("=" * 70)

    # Start with pure state
    rho = excited_state()
    print(f"\nStarting with |1⟩ (pure state)")

    # Apply amplitude damping repeatedly
    channel = AmplitudeDamping(gamma=0.15)

    print(f"\nApplying amplitude damping (γ=0.15) repeatedly:\n")
    print(f"{'Application':<15} {'Purity':<15} {'Purity Loss':<15}")
    print("-" * 45)

    for step in range(8):
        purity = rho.purity()
        loss = 1.0 - purity
        print(f"{step:<15} {purity:<15.4f} {loss:<15.4f}")
        rho = channel.apply(rho)


# ============================================================================
# Example 7: Verify CPTP property
# ============================================================================


def example_7_verify_cptp():
    """Verify that channels are valid CPTP maps."""
    print("\n" + "=" * 70)
    print("Example 7: Verifying CPTP (Completely Positive Trace-Preserving)")
    print("=" * 70)

    channels = [
        AmplitudeDamping(gamma=0.3),
        PhaseDamping(lam=0.5),
        Depolarizing(p=0.4),
        BitFlip(p=0.2),
    ]

    print(f"\nAll channels must satisfy: Σᵢ Kᵢ† Kᵢ = I\n")
    print(f"{'Channel':<30} {'Is CPTP?':<15}")
    print("-" * 45)

    for channel in channels:
        is_cptp = channel.verify_cptp()
        status = "✓ Valid" if is_cptp else "✗ Invalid"
        print(f"{str(channel):<30} {status:<15}")

    print(f"\nIf all valid, channels preserve quantum information correctly!")


# ============================================================================
# Example 8: Create custom states with state vectors
# ============================================================================


def example_8_custom_states():
    """Create custom quantum states from state vectors."""
    print("\n" + "=" * 70)
    print("Example 8: Creating Custom Quantum States")
    print("=" * 70)

    # Create arbitrary superposition: |ψ⟩ = (1/√2)|0⟩ + (i/√2)|1⟩
    psi = np.array([1, 1j]) / np.sqrt(2)
    rho = state_vector_to_density_matrix(psi)

    print(f"\nCreated state: |ψ⟩ = (|0⟩ + i|1⟩)/√2")
    print(f"Purity: {rho.purity():.4f} (pure state)")
    print(f"Bloch vector: {bloch_vector(rho)}")

    # Create equal superposition
    psi2 = np.array([1, -1]) / np.sqrt(2)
    rho2 = state_vector_to_density_matrix(psi2)

    print(f"\nCreated state: |ψ'⟩ = (|0⟩ - |1⟩)/√2")
    print(f"Purity: {rho2.purity():.4f}")
    print(f"Fidelity between the two: {fidelity(rho, rho2):.4f}")


# ============================================================================
# Main: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "QUANTUM NOISE SIMULATION - USAGE EXAMPLES".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    example_1_basic()
    example_2_compare_channels()
    example_3_sequential_channels()
    example_4_bloch_vectors()
    example_5_fidelity()
    example_6_purity_tracking()
    example_7_verify_cptp()
    example_8_custom_states()

    print("\n" + "=" * 70)
    print("Examples completed! Review the output to understand quantum noise.")
    print("=" * 70)
    print("\n📚 For more details, see:")
    print("  - README.md: Overview and quick start")
    print("  - THEORY.md: Detailed mathematical theory")
    print("  - quantum_noise/channels.py: Implementation details")
    print("\n")
