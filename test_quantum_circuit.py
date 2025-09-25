"""
Simple tests for the quantum circuit simulator.
"""

import numpy as np
from quantum_circuit import QuantumCircuit


def test_initialization():
    """Test circuit initialization."""
    print("Testing initialization...")
    
    # Test 1-qubit circuit
    qc1 = QuantumCircuit(1)
    assert qc1.n_qubits == 1
    assert qc1.n_states == 2
    assert np.isclose(qc1.state_vector[0], 1.0)
    assert np.isclose(qc1.state_vector[1], 0.0)
    
    # Test 2-qubit circuit
    qc2 = QuantumCircuit(2)
    assert qc2.n_qubits == 2
    assert qc2.n_states == 4
    assert np.isclose(qc2.state_vector[0], 1.0)
    for i in range(1, 4):
        assert np.isclose(qc2.state_vector[i], 0.0)
    
    print("✓ Initialization tests passed")


def test_pauli_x():
    """Test Pauli-X gate."""
    print("Testing Pauli-X gate...")
    
    # Test on single qubit
    qc = QuantumCircuit(1)
    qc.pauli_x(0)
    
    assert np.isclose(qc.state_vector[0], 0.0)
    assert np.isclose(qc.state_vector[1], 1.0)
    
    # Apply again to return to original state
    qc.pauli_x(0)
    assert np.isclose(qc.state_vector[0], 1.0)
    assert np.isclose(qc.state_vector[1], 0.0)
    
    print("✓ Pauli-X tests passed")


def test_hadamard():
    """Test Hadamard gate."""
    print("Testing Hadamard gate...")
    
    qc = QuantumCircuit(1)
    qc.hadamard(0)
    
    # Should create equal superposition
    expected_amplitude = 1.0 / np.sqrt(2)
    assert np.isclose(qc.state_vector[0], expected_amplitude)
    assert np.isclose(qc.state_vector[1], expected_amplitude)
    
    print("✓ Hadamard tests passed")


def test_cnot():
    """Test CNOT gate."""
    print("Testing CNOT gate...")
    
    # Test CNOT with control in |0⟩ state
    qc = QuantumCircuit(2)
    qc.cnot(0, 1)
    
    # Should remain |00⟩
    assert np.isclose(qc.state_vector[0], 1.0)  # |00⟩
    for i in range(1, 4):
        assert np.isclose(qc.state_vector[i], 0.0)
    
    # Test CNOT with control in |1⟩ state
    qc.reset()
    qc.pauli_x(0)  # Set control to |1⟩
    qc.cnot(0, 1)
    
    # Should be |11⟩
    assert np.isclose(qc.state_vector[0], 0.0)  # |00⟩
    assert np.isclose(qc.state_vector[1], 0.0)  # |01⟩
    assert np.isclose(qc.state_vector[2], 0.0)  # |10⟩
    assert np.isclose(qc.state_vector[3], 1.0)  # |11⟩
    
    print("✓ CNOT tests passed")


def test_bell_state():
    """Test creation of Bell state."""
    print("Testing Bell state creation...")
    
    qc = QuantumCircuit(2)
    qc.hadamard(0)
    qc.cnot(0, 1)
    
    # Should create (|00⟩ + |11⟩) / √2
    expected_amplitude = 1.0 / np.sqrt(2)
    assert np.isclose(qc.state_vector[0], expected_amplitude)  # |00⟩
    assert np.isclose(qc.state_vector[1], 0.0)                # |01⟩
    assert np.isclose(qc.state_vector[2], 0.0)                # |10⟩
    assert np.isclose(qc.state_vector[3], expected_amplitude)  # |11⟩
    
    print("✓ Bell state tests passed")


def test_probabilities():
    """Test probability calculations."""
    print("Testing probability calculations...")
    
    qc = QuantumCircuit(2)
    qc.hadamard(0)
    qc.cnot(0, 1)
    
    probs = qc.get_state_probabilities()
    
    # Bell state should have equal probabilities for |00⟩ and |11⟩
    assert len(probs) == 2
    assert '00' in probs
    assert '11' in probs
    assert np.isclose(probs['00'], 0.5)
    assert np.isclose(probs['11'], 0.5)
    
    print("✓ Probability tests passed")


def test_reset():
    """Test circuit reset functionality."""
    print("Testing reset...")
    
    qc = QuantumCircuit(2)
    qc.hadamard(0)
    qc.cnot(0, 1)
    
    # Verify state is changed
    assert not np.isclose(qc.state_vector[0], 1.0)
    
    qc.reset()
    
    # Should be back to |00⟩
    assert np.isclose(qc.state_vector[0], 1.0)
    for i in range(1, 4):
        assert np.isclose(qc.state_vector[i], 0.0)
    
    assert len(qc.circuit_history) == 0
    
    print("✓ Reset tests passed")


def test_error_handling():
    """Test error handling for invalid operations."""
    print("Testing error handling...")
    
    qc = QuantumCircuit(2)
    
    # Test invalid qubit indices
    try:
        qc.hadamard(2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        qc.cnot(0, 2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        qc.cnot(0, 0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test invalid circuit size
    try:
        QuantumCircuit(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Error handling tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running quantum circuit tests...")
    print("=" * 40)
    
    test_initialization()
    test_pauli_x()
    test_hadamard()
    test_cnot()
    test_bell_state()
    test_probabilities()
    test_reset()
    test_error_handling()
    
    print("=" * 40)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()