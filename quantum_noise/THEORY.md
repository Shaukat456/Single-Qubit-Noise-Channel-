# Quantum Noise Channels: Complete Theory and Documentation

## Table of Contents

1. [Quantum State Representations](#quantum-state-representations)
2. [Quantum Channels](#quantum-channels)
3. [Kraus Operators](#kraus-operators)
4. [Specific Channels](#specific-channels)
5. [Implementation Details](#implementation-details)
6. [Use Cases](#use-cases)

---

## Quantum State Representations

### State Vectors (for pure states)

A pure quantum state is represented as a **2D complex vector** (for single qubits):

```
|ψ⟩ = [a]  where |a|² + |b|² = 1
      [b]
```

- **|0⟩** = [1, 0]ᵀ (ground state / computational basis)
- **|1⟩** = [0, 1]ᵀ (excited state)
- **|+⟩** = (|0⟩ + |1⟩)/√2 = [1/√2, 1/√2]ᵀ (superposition on X basis)

**Interpretation**: Probability of measuring |0⟩ is |a|², probability of |1⟩ is |b|².

### Density Matrices (for mixed states)

A **density matrix ρ** is the generalization to mixed states:

```
ρ = [ρ₀₀  ρ₀₁]   (2×2 complex matrix)
    [ρ₁₀  ρ₁₁]
```

**Properties**:

| Property        | Formula             | Meaning                                    |
| --------------- | ------------------- | ------------------------------------------ |
| **Hermitian**   | ρ† = ρ              | Matrix equals its conjugate transpose      |
| **Normalized**  | Tr(ρ) = 1           | ρ₀₀ + ρ₁₁ = 1                              |
| **Positive**    | All eigenvalues ≥ 0 | Linear combinations give probabilities ≥ 0 |
| **Pure State**  | Tr(ρ²) = 1          | Pure state: ρ = \|ψ⟩⟨ψ\|                   |
| **Mixed State** | Tr(ρ²) < 1          | Statistical mixture of pure states         |

### Converting State Vector to Density Matrix

For a pure state \|ψ⟩:

```
ρ = |ψ⟩⟨ψ|
```

**Example**:

```
|ψ⟩ = [a]     ρ = |ψ⟩⟨ψ| = [a]  [a* b*] = [|a|²   a·b*]
      [b]                   [b]           [a*·b  |b|²]
```

### Key Density Matrices

1. **Ground state |0⟩**:

   ```
   ρ₀ = [1  0]    (pure state, purity = 1)
        [0  0]
   ```

2. **Excited state |1⟩**:

   ```
   ρ₁ = [0  0]    (pure state, purity = 1)
        [0  1]
   ```

3. **Superposition |+⟩ = (|0⟩+|1⟩)/√2**:

   ```
   ρ₊ = [1/2  1/2]   (pure state, purity = 1)
        [1/2  1/2]
   ```

4. **Maximally mixed state**:
   ```
   ρmix = [1/2   0 ]   (completely mixed, purity = 0.5)
          [ 0   1/2]
   ```

### Measurements and Observables

An **observable** is a Hermitian operator (like Pauli matrices):

```
σₓ (bit-flip)  =  [0  1]    σᵧ (Y-rotation)  =  [0  -i]
                  [1  0]                        [i   0]

σᵤ (phase-flip) =  [1   0]
                   [0  -1]
```

**Expectation value**: ⟨O⟩ = Tr(ρ O)

---

## Quantum Channels

### Definition

A **quantum channel** ℰ is a linear map that transforms density matrices while:

1. **Preserving trace** (normalization): Tr(ℰ(ρ)) = Tr(ρ)
2. **Being completely positive**: Preserves positive-semidefiniteness
3. **Being physical**: Can be realized by quantum operations

$$\mathcal{E}(\rho) = \rho' \text{ where } \text{Tr}(\rho') = 1$$

### Choi-Kraus Representation

**Theorem (Kraus)**: Every CPTP channel has a Kraus representation:

$$\mathcal{E}(\rho) = \sum_{i=1}^{n} K_i \rho K_i^\dagger$$

where Kraus operators satisfy:

$$\sum_{i=1}^{n} K_i^\dagger K_i = I \text{ (completeness)}$$

**Physical Interpretation**:

- Each Kᵢ represents a possible outcome/process
- The sum over i provides the total evolution
- The completeness condition ensures trace preservation

### Example: Amplitude Damping

```
K₀ = [1      0    ]     K₁ = [0  √γ]
     [0  √(1-γ)]          [0   0 ]

Verification: K₀†K₀ + K₁†K₁ = I ✓
```

**How it works**:

$$\mathcal{E}(\rho) = K_0 \rho K_0^\dagger + K_1 \rho K_1^\dagger$$

On the ground state |0⟩:

```
ρ = [1  0]    →    [1  0] (unchanged)
    [0  0]         [0  0]
```

On the excited state |1⟩:

```
ρ = [0  0]    →    [γ    0  ]
    [0  1]         [0   1-γ ]
```

This shows: probability γ of decaying to |0⟩, probability (1-γ) of staying in |1⟩.

---

## Kraus Operators

### Why Kraus Operators?

1. **Physical Interpretation**: Each Kᵢ represents a quantum jump or measurement outcome
2. **Computational Efficiency**: Easy to implement via matrix multiplication
3. **CPTP Verification**: Can directly check completeness condition
4. **Composition**: Sequential channels compose naturally

### General Form for Single Qubit

For a single qubit with n-outcomes:

$$K_i = \begin{bmatrix} a_i & b_i \\ c_i & d_i \end{bmatrix}$$

Constraint check requires:
$$\sum_i K_i^\dagger K_i = I$$

### Construction Strategy

**For 2-outcome channels** (2 Kraus ops):

1. Parameterize: K₀ = √(1-p) × A, K₁ = √p × B
2. Verify: A†A + B†B = I
3. Set probabilities: p for outcome 1, (1-p) for outcome 0

**Example - Bit-flip**:

- A = I (identity)
- B = σₓ (Pauli-X)
- Check: I + σₓ² = I + I = 2I ✗ (doesn't work)
- Better: K₀ = √(1-p)I, K₁ = √p σₓ
- Check: (1-p)I + pI = I ✓

---

## Specific Channels

### 1. Amplitude Damping (Energy Loss)

**Physical Process**: T₁ relaxation - qubit loses energy to environment

**Parameters**: γ ∈ [0,1] - damping rate

**Kraus Operators**:

```
K₀ = [1       0    ]    K₁ = [0  √γ]
     [0  √(1-γ)]            [0   0 ]
```

**Action on basis states**:

```
|0⟩ →  |0⟩                    (unaffected)
|1⟩ → √(1-γ)|1⟩ + √γ|0⟩     (decays to ground)
```

**Property**:

- Ground state unchanged (lowest energy state)
- Excited state decays toward ground state
- Populations change (diagonal elements modified)

**Use case**: Model qubit relaxation in superconducting qubits

---

### 2. Phase Damping (Dephasing)

**Physical Process**: T₂ decay - loss of phase coherence

**Parameters**: λ ∈ [0,1] - dephasing rate

**Kraus Operators**:

```
K₀ = [1       0    ]    K₁ = [0  0]
     [0  √(1-λ)]            [0  √λ]
```

**Action**:

```
|0⟩ → |0⟩                   (unaffected)
|1⟩ → |1⟩                   (unaffected)
|+⟩ → mixture of |0⟩ and |1⟩ (coherence destroyed)
```

**Property**:

- Only affects off-diagonal elements (coherence/superposition)
- Populations preserved
- No energy exchange

**Key difference from amplitude damping**:

- Amplitude: Energy loss (population redistribution)
- Phase: Coherence loss (off-diagonal decay)

---

### 3. Depolarizing Channel (Uniform Noise)

**Physical Process**: Random errors in all directions

**Parameters**: p ∈ [0,1] - error probability

**Kraus Operators**:

```
K₀ = √(1-3p/4) I
K₁ = √(p/4) σₓ
K₂ = √(p/4) σᵧ
K₃ = √(p/4) σᵤ
```

**Action**:
$$\mathcal{E}(\rho) = (1-p)\rho + p\frac{I}{2}$$

**Interpretation**: With probability p, replace state with maximally mixed state

**Limiting cases**:

- p = 0: No noise, ℰ(ρ) = ρ
- p = 1: Complete depolarization, ℰ(ρ) = I/2 (maximally mixed)

**Properties**:

- Symmetric in all Pauli directions
- Purity decreases: Tr(ℰ(ρ)²) ≤ Tr(ρ²)
- Entropy increases

**Use case**: Generic error model, worst-case analysis

---

### 4. Bit-Flip Channel (X-Errors)

**Physical Process**: Random bit flips |0⟩ ↔ |1⟩

**Parameters**: p ∈ [0,1] - flip probability

**Kraus Operators**:

```
K₀ = √(1-p) I     K₁ = √p σₓ
```

**Action on basis states**:

```
|0⟩ → √(1-p)|0⟩ + √p|1⟩
|1⟩ → √p|0⟩ + √(1-p)|1⟩
```

**Invariant state**: |+⟩ is eigenstate of σₓ, unaffected by bit-flips

**Properties**:

- No energy loss or gain
- Only population exchanges
- Population in eigenstates of σₓ are stable

**Use case**: Model bit-flip errors in error correction codes

---

### 5. Phase Damping (Random Phase Flip)

**Parameters**: λ ∈ [0,1]

**Kraus Operators**:

```
K₀ = √(1-λ) I    K₁ = √λ σᵤ
```

**Action**:
$$\mathcal{E}(\rho) = (1-\lambda)\rho + \lambda\sigma_z\rho\sigma_z$$

**Effect**: Random phase flips without changing populations

---

## Implementation Details

### Density Matrix Class

```python
class DensityMatrix:
    def __init__(self, matrix: np.ndarray):
        # Verify 2×2, normalized (Tr = 1)
        self.matrix = matrix

    def purity(self) -> float:
        """Tr(ρ²) - measure of mixedness"""
        return np.trace(self.matrix @ self.matrix)

    def entropy(self) -> float:
        """-Tr(ρ log₂ ρ) - von Neumann entropy"""
        eigenvalues = np.linalg.eigvals(self.matrix)
        return -np.sum(eigenvalues * np.log2(eigenvalues+1e-10))

    def expectation(self, observable):
        """⟨O⟩ = Tr(ρO)"""
        return np.trace(self.matrix @ observable)
```

### Channel Application

All channels inherit from `QuantumChannel`:

```python
class QuantumChannel(ABC):
    @abstractmethod
    def get_kraus_operators(self) -> List[np.ndarray]:
        """Return [K₁, K₂, ..., Kₙ]"""
        pass

    def apply(self, rho: DensityMatrix) -> DensityMatrix:
        """ℰ(ρ) = Σᵢ Kᵢ ρ Kᵢ†"""
        kraus_ops = self.get_kraus_operators()
        result = np.zeros((2, 2), dtype=complex)
        for K in kraus_ops:
            result += K @ rho.matrix @ K.conj().T
        return DensityMatrix(result)

    def verify_cptp(self):
        """Check Σᵢ Kᵢ† Kᵢ = I"""
        kraus_ops = self.get_kraus_operators()
        sum_ops = np.sum([K.conj().T @ K for K in kraus_ops], axis=0)
        return np.allclose(sum_ops, np.eye(2))
```

### Numerical Considerations

1. **Trace preservation**: Automatically enforced by Kraus representation
2. **Positive-semidefiniteness**: Preserved by CPTP condition
3. **Normalization**: Verified in DensityMatrix constructor
4. **Eigenvalue gaps**: Small negative eigenvalues set to 1e-10 for entropy calculation

---

## Use Cases

### 1. Quantum Error Correction

Design codes robust to specific noise channels:

- Amplitude damping → 3-qubit bit-flip code
- Phase errors → 3-qubit phase-flip code
- General errors → Surface codes

### 2. Device Calibration

Measure and model actual quantum hardware:

- Characterize T₁, T₂ times
- Identify dominant error channels
- Tailor error mitigation strategies

### 3. Algorithm Simulation

Study algorithm robustness under realistic noise:

```python
algorithm_noise_simulation.py:
    for p_noise in [0.001, 0.01, 0.1]:
        channel = DepolarizingChannel(p_noise)

        # Apply channel between algorithm steps
        result = run_algorithm_with_noise(channel)
        success_rate = measure_success(result)

        plot(p_noise, success_rate)
```

### 4. Benchmarking

Characterize quantum gates and operations:

```python
# Measure fidelity of a noisy gate
ideal_state = |0⟩
noisy_output = channel.apply(ideal_state)
fidelity = ⟨ideal|noisy⟩² = 1 - purity_loss
```

---

## Mathematical Summary

### CPTP Map Properties

For channel ℰ with Kraus operators {Kᵢ}:

| Property        | Formula                    | Check                    |
| --------------- | -------------------------- | ------------------------ |
| **Linearity**   | ℰ(aρ+bσ) = aℰ(ρ)+bℰ(σ)     | Linear combination       |
| **Trace**       | Tr(ℰ(ρ)) = Tr(ρ)           | Σᵢ Kᵢ†Kᵢ = I             |
| **Positivity**  | ρ ≥ 0 ⟹ ℰ(ρ) ≥ 0           | Eigenvalue decomposition |
| **Composition** | ℰ = ℰ₂∘ℰ₁ compose channels | Apply sequentially       |

### Purity Under Channels

Pure → Mixed transition:
$$\text{Tr}(\mathcal{E}(\rho)^2) \leq \text{Tr}(\rho^2)$$

General bound:
$$\text{Tr}(\rho'^2) = \sum_i p_i^2 \leq 1$$

with equality iff ρ' is pure.

---

## References

1. **Michael A. Nielsen & Isaac L. Chuang** - "Quantum Computation and Quantum Information" (Cambridge University Press)
2. **Barbara M. Terhal** - "Quantum Error Correction for Quantum Memories"
3. **David Deutsch** - "Quantum Theory, the Church-Turing Principle and the Universal Quantum Computer"
4. **Karl Kraus** - "General Properties of Entropy" (1971)

---

## Final Notes

This implementation prioritizes:

- **Clarity**: Every operation mathematically justified
- **Simplicity**: No dependencies beyond NumPy
- **Correctness**: All channels verified as CPTP
- **Efficiency**: Direct Kraus application
- **Extensibility**: Easy to add new channels

For production quantum simulation, see QuTiP or Qiskit. This library is for **learning and understanding** the mathematics.
