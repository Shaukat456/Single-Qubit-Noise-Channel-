# Quantum Noise Simulation Library

A from-scratch implementation of single-qubit quantum noise channels using only NumPy and SciPy.

## Overview

This library provides tools to simulate quantum noise through **Kraus operators** and **density matrices**. Unlike pre-built quantum libraries (Qiskit, Cirq, QuTiP), this implementation is built from first principles for educational clarity and complete control.

## Key Concepts

### Density Matrices
A density matrix ρ describes the quantum state of a system:
- **Pure state**: ρ = |ψ⟩⟨ψ| has rank 1 and purity = 1
- **Mixed state**: A statistical mixture of pure states (models noise/decoherence)
- **Maximally mixed state**: ρ = I/2 represents complete uncertainty (purity = 0.5)

### Quantum Channels (CPTP Maps)

A quantum channel ℰ transforms density matrices:
- **Input**: ρ (input state)
- **Output**: ℰ(ρ) = Σᵢ Kᵢ ρ Kᵢ† (Kraus representation)
- **CPTP**: Completely Positive, Trace-Preserving

#### Why Kraus Operators?
The Kraus representation provides a **physical interpretation**:
- Each Kᵢ represents a possible "outcome" or physical process
- Kᵢ ρ Kᵢ† is the post-measurement state if outcome i occurred
- Σᵢ Kᵢ† Kᵢ = I (trace preservation)

### Single-Qubit Noise Channels

#### 1. **Amplitude Damping** (Generational Damping)
**Physical**: Energy loss to environment (T₁ decay)
- Model: |1⟩ → |0⟩ at rate γ
- Kraus operators:
  - K₀ = [[1, 0], [0, √(1-γ)]]
  - K₁ = [[0, √γ], [0, 0]]
- Effect: Excited state loses energy, ground state unaffected

#### 2. **Phase Damping** (Dephasing)
**Physical**: Loss of quantum coherence (T₂ decay)
- Model: Superposition states lose off-diagonal elements
- Kraus operators:
  - K₀ = [[1, 0], [0, √(1-λ)]]
  - K₁ = [[0, 0], [0, √λ]]
- Effect: |±⟩ → mixture of |0⟩ and |1⟩

#### 3. **Depolarizing Channel**
**Physical**: Random bit-flip noise
- Model: Replace state with uniform mixture with probability p
- Kraus operators:
  - K₀ = √(1 - 3p/4) I
  - K₁ = √(p/4) σₓ
  - K₂ = √(p/4) σᵧ
  - K₃ = √(p/4) σᵧ
- Effect: State becomes more mixed as p increases

#### 4. **Bit-Flip Channel**
**Physical**: Spontaneous bit-flips (errors)
- Model: |0⟩ ↔ |1⟩ at probability p
- Kraus operators:
  - K₀ = √(1-p) I
  - K₁ = √p σₓ
- Effect: Population swaps between |0⟩ and |1⟩

## Architecture

```
quantum_noise/
├── __init__.py           # Package exports
├── core.py              # Density matrix operations
├── channels.py          # Quantum channel implementations
└── utils.py             # Helper functions
```

## Usage Examples

```python
from quantum_noise import ground_state, AmplitudeDamping, PhaseDamping

# Create initial state: |0⟩
rho = ground_state()

# Apply amplitude damping (energy loss)
channel = AmplitudeDamping(gamma=0.3)
rho_noisy = channel.apply(rho)

# Measure purity to quantify noise effect
print(f"Original purity: {rho.purity()}")
print(f"After noise: {rho_noisy.purity()}")
```

## Theory References

- **Density Matrices**: Standard representation in quantum mechanics textbooks
- **Kraus Operators**: Discovered by Karl Kraus (1971)
- **CPTP Maps**: Choi's theorem on completely positive maps
- **Standard reference**: Wilde, "Quantum Information Theory", Cambridge University Press

## Implementation Details

All operations use NumPy for:
- Matrix operations
- Eigenvalue decomposition
- Linear algebra

No external quantum libraries are used - this is a pure mathematical implementation.
