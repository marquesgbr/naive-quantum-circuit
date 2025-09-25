"""
Example usage of the naive quantum circuit simulator.

This script demonstrates how to use the QuantumCircuit class to create
quantum circuits, apply gates, and perform measurements.
"""

from quantum_circuit import QuantumCircuit


def example_single_qubit():
    """Example with a single qubit: apply Hadamard gate and measure."""
    print("=== Single Qubit Example ===")
    
    # Create a 1-qubit circuit
    qc = QuantumCircuit(1)
    print("Initial state:")
    qc.print_state()
    
    # Apply Hadamard gate to create superposition
    qc.hadamard(0)
    print("\nAfter Hadamard gate:")
    qc.print_state()
    
    # Measure the qubit
    result = qc.measure(0)
    print(f"\nMeasurement result: {result}")
    print("\nState after measurement:")
    qc.print_state()
    
    print("\n" + "="*50 + "\n")


def example_two_qubits():
    """Example with two qubits: create Bell state with Hadamard and CNOT."""
    print("=== Two Qubits Bell State Example ===")
    
    # Create a 2-qubit circuit
    qc = QuantumCircuit(2)
    print("Initial state:")
    qc.print_state()
    
    # Create Bell state: |00⟩ + |11⟩
    qc.hadamard(0)  # Put first qubit in superposition
    print("\nAfter Hadamard on qubit 0:")
    qc.print_state()
    
    qc.cnot(0, 1)   # Entangle with CNOT
    print("\nAfter CNOT(0,1) - Bell state created:")
    qc.print_state()
    
    # Measure all qubits
    result = qc.measure()
    print(f"\nMeasurement result: {result}")
    print("\nState after measurement:")
    qc.print_state()
    
    print("\n" + "="*50 + "\n")


def example_pauli_x():
    """Example demonstrating Pauli-X (NOT) gate."""
    print("=== Pauli-X Gate Example ===")
    
    # Create a 2-qubit circuit
    qc = QuantumCircuit(2)
    print("Initial state:")
    qc.print_state()
    
    # Apply Pauli-X to flip qubit 1
    qc.pauli_x(1)
    print("\nAfter Pauli-X on qubit 1:")
    qc.print_state()
    
    # Apply Hadamard to qubit 0
    qc.hadamard(0)
    print("\nAfter Hadamard on qubit 0:")
    qc.print_state()
    
    print("\n" + "="*50 + "\n")


def example_three_qubits():
    """Example with three qubits showing more complex operations."""
    print("=== Three Qubits Example ===")
    
    # Create a 3-qubit circuit
    qc = QuantumCircuit(3)
    print("Initial state:")
    qc.print_state()
    
    # Apply gates to create a more complex state
    qc.hadamard(0)
    qc.hadamard(1)
    qc.cnot(0, 2)
    
    print("\nAfter H(0), H(1), CNOT(0,2):")
    qc.print_state()
    
    # Show circuit history
    print(f"\nCircuit history: {qc.get_circuit_history()}")
    
    # Multiple measurements
    print("\nPerforming multiple measurements:")
    for i in range(3):
        # Reset and recreate the same state
        qc.reset()
        qc.hadamard(0)
        qc.hadamard(1)
        qc.cnot(0, 2)
        
        result = qc.measure()
        print(f"Measurement {i+1}: {result['outcome']}")
    
    print("\n" + "="*50 + "\n")


def main():
    """Run all examples."""
    print("Naive Quantum Circuit Simulator Examples")
    print("=" * 50)
    
    example_single_qubit()
    example_two_qubits()
    example_pauli_x()
    example_three_qubits()
    
    print("Examples completed!")


if __name__ == "__main__":
    main()