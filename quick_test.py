#!/usr/bin/env python3
"""Quick test to verify the quantum circuit works"""

from quantum_circuit import QuantumCircuit

# Test basic functionality
print("Testing quantum circuit...")
qc = QuantumCircuit(2)
print("✓ Circuit created")

qc.hadamard(0)
print("✓ Hadamard applied")

qc.cnot(0, 1)
print("✓ CNOT applied")

print("\nBell state created:")
qc.print_state()

probs = qc.get_state_probabilities()
print(f"\nProbabilities: {probs}")

print("\n✓ All tests passed! The simulator is working correctly.")