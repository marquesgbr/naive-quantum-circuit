"""
Naive Quantum Circuit Simulator

A simple implementation of a quantum circuit simulator using numpy.
Supports basic quantum gates (Hadamard, Pauli-X, CNOT) and measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import random


class QuantumCircuit:
    """
    A naive quantum circuit simulator that represents quantum states as state vectors
    and implements basic quantum gates and measurements.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize a quantum circuit with n qubits in the |0> state.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        
        # Initialize all qubits in |0> state
        self.state_vector = np.zeros(self.n_states, dtype=complex)
        self.state_vector[0] = 1.0  # |00...0> state
        
        # Keep track of applied gates for visualization
        self.circuit_history = []
    
    def _validate_qubit(self, qubit: int) -> None:
        """Validate that qubit index is valid."""
        if not 0 <= qubit < self.n_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.n_qubits-1}]")
    
    def hadamard(self, qubit: int) -> None:
        """
        Apply Hadamard gate to the specified qubit.
        H = (1/√2) * [[1, 1], [1, -1]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
        """
        self._validate_qubit(qubit)
        
        # Hadamard matrix
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Apply the gate
        self._apply_single_qubit_gate(H, qubit)
        self.circuit_history.append(f"H({qubit})")
    
    def pauli_x(self, qubit: int) -> None:
        """
        Apply Pauli-X (NOT) gate to the specified qubit.
        X = [[0, 1], [1, 0]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
        """
        self._validate_qubit(qubit)
        
        # Pauli-X matrix
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Apply the gate
        self._apply_single_qubit_gate(X, qubit)
        self.circuit_history.append(f"X({qubit})")
    
    def cnot(self, control: int, target: int) -> None:
        """
        Apply CNOT gate with specified control and target qubits.
        
        Args:
            control: Index of the control qubit
            target: Index of the target qubit
        """
        self._validate_qubit(control)
        self._validate_qubit(target)
        
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        # Apply CNOT gate
        self._apply_cnot_gate(control, target)
        self.circuit_history.append(f"CNOT({control},{target})")
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the specified qubit."""
        new_state = np.zeros_like(self.state_vector)
        
        # Iterate through all basis states
        for i in range(self.n_states):
            # Extract the bit value of the target qubit
            qubit_value = (i >> qubit) & 1
            
            # Apply the gate matrix
            for new_qubit_value in range(2):
                # Calculate the new state index
                if qubit_value != new_qubit_value:
                    new_i = i ^ (1 << qubit)  # Flip the target qubit bit
                else:
                    new_i = i
                
                new_state[new_i] += gate_matrix[new_qubit_value, qubit_value] * self.state_vector[i]
        
        self.state_vector = new_state
    
    def _apply_cnot_gate(self, control: int, target: int) -> None:
        """Apply CNOT gate between control and target qubits."""
        new_state = np.zeros_like(self.state_vector)
        
        for i in range(self.n_states):
            # Extract control and target bit values
            control_bit = (i >> control) & 1
            target_bit = (i >> target) & 1
            
            if control_bit == 1:
                # Flip target bit
                new_i = i ^ (1 << target)
            else:
                # Keep the same state
                new_i = i
            
            new_state[new_i] = self.state_vector[i]
        
        self.state_vector = new_state
    
    def measure(self, qubit: Optional[int] = None) -> Dict:
        """
        Measure the quantum circuit. If qubit is specified, measure only that qubit.
        Otherwise, measure all qubits.
        
        Args:
            qubit: Optional qubit index to measure. If None, measure all qubits.
            
        Returns:
            Dictionary with measurement results and probabilities
        """
        if qubit is not None:
            return self._measure_single_qubit(qubit)
        else:
            return self._measure_all_qubits()
    
    def _measure_single_qubit(self, qubit: int) -> Dict:
        """Measure a single qubit."""
        self._validate_qubit(qubit)
        
        # Calculate probabilities for |0> and |1>
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.n_states):
            qubit_value = (i >> qubit) & 1
            prob = abs(self.state_vector[i]) ** 2
            
            if qubit_value == 0:
                prob_0 += prob
            else:
                prob_1 += prob
        
        # Simulate measurement outcome
        if random.random() < prob_0:
            outcome = 0
            self._collapse_state_single_qubit(qubit, 0)
        else:
            outcome = 1
            self._collapse_state_single_qubit(qubit, 1)
        
        return {
            'qubit': qubit,
            'outcome': outcome,
            'probabilities': {'0': prob_0, '1': prob_1}
        }
    
    def _measure_all_qubits(self) -> Dict:
        """Measure all qubits simultaneously."""
        probabilities = {}
        
        # Calculate probabilities for each basis state
        for i in range(self.n_states):
            prob = abs(self.state_vector[i]) ** 2
            if prob > 1e-10:  # Only include non-zero probabilities
                binary_state = format(i, f'0{self.n_qubits}b')
                probabilities[binary_state] = prob
        
        # Simulate measurement outcome
        rand_val = random.random()
        cumulative_prob = 0.0
        outcome = None
        
        for state, prob in probabilities.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                outcome = state
                break
        
        if outcome is None:
            outcome = list(probabilities.keys())[0]
        
        # Collapse the state
        outcome_index = int(outcome, 2)
        self.state_vector = np.zeros_like(self.state_vector)
        self.state_vector[outcome_index] = 1.0
        
        return {
            'outcome': outcome,
            'probabilities': probabilities
        }
    
    def _collapse_state_single_qubit(self, qubit: int, outcome: int) -> None:
        """Collapse the state after measuring a single qubit."""
        new_state = np.zeros_like(self.state_vector)
        norm = 0.0
        
        for i in range(self.n_states):
            qubit_value = (i >> qubit) & 1
            if qubit_value == outcome:
                new_state[i] = self.state_vector[i]
                norm += abs(self.state_vector[i]) ** 2
        
        # Normalize the state
        if norm > 0:
            new_state /= np.sqrt(norm)
        
        self.state_vector = new_state
    
    def get_state_probabilities(self) -> Dict[str, float]:
        """
        Get the probabilities of all basis states.
        
        Returns:
            Dictionary mapping binary state strings to their probabilities
        """
        probabilities = {}
        
        for i in range(self.n_states):
            prob = abs(self.state_vector[i]) ** 2
            if prob > 1e-10:  # Only include non-zero probabilities
                binary_state = format(i, f'0{self.n_qubits}b')
                probabilities[binary_state] = prob
        
        return probabilities
    
    def visualize_state(self) -> None:
        """Visualize the current quantum state as a bar chart."""
        probabilities = self.get_state_probabilities()
        
        if not probabilities:
            print("No non-zero probability states to visualize")
            return
        
        states = list(probabilities.keys())
        probs = list(probabilities.values())
        
        plt.figure(figsize=(max(8, len(states)), 6))
        bars = plt.bar(states, probs)
        plt.xlabel('Quantum States')
        plt.ylabel('Probability')
        plt.title(f'Quantum State Probabilities ({self.n_qubits} qubits)')
        plt.xticks(rotation=45 if len(states) > 8 else 0)
        
        # Add probability values on top of bars
        for bar, prob in zip(bars, probs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def print_state(self) -> None:
        """Print the current quantum state in a readable format."""
        print(f"Quantum Circuit State ({self.n_qubits} qubits):")
        print("=" * 40)
        
        probabilities = self.get_state_probabilities()
        
        if not probabilities:
            print("All states have zero probability")
            return
        
        for state, prob in sorted(probabilities.items()):
            amplitude = self.state_vector[int(state, 2)]
            print(f"|{state}⟩: {amplitude:.3f} (probability: {prob:.3f})")
    
    def reset(self) -> None:
        """Reset the circuit to the |00...0> state."""
        self.state_vector = np.zeros(self.n_states, dtype=complex)
        self.state_vector[0] = 1.0
        self.circuit_history = []
    
    def get_circuit_history(self) -> List[str]:
        """Get the history of applied gates."""
        return self.circuit_history.copy()