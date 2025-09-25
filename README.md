# Naive Quantum Circuit Simulator

Um simulador ingênuo de circuitos quânticos implementado em Python usando NumPy. Este projeto fornece uma implementação simples de um simulador de circuitos quânticos que suporta gates quânticos básicos e medições.

## Funcionalidades

- **Simulação de circuitos quânticos** com número arbitrário de qubits
- **Gates quânticos básicos**:
  - Hadamard (H) - cria superposição
  - Pauli-X (X) - gate NOT quântico
  - CNOT - gate controlado para criar emaranhamento
- **Medições** de qubits individuais ou todos os qubits
- **Visualização de estados** quânticos
- **Cálculo de probabilidades** para todos os estados base

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/marquesgbr/naive-quantum-circuit.git
cd naive-quantum-circuit
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso Básico

```python
from quantum_circuit import QuantumCircuit

# Criar um circuito de 2 qubits
qc = QuantumCircuit(2)

# Aplicar gates
qc.hadamard(0)  # Hadamard no qubit 0
qc.cnot(0, 1)   # CNOT entre qubits 0 e 1

# Visualizar o estado
qc.print_state()

# Medir todos os qubits
resultado = qc.measure()
print(resultado)
```

## Exemplos

### Estado de Bell
Criando o estado de Bell clássico (|00⟩ + |11⟩)/√2:

```python
qc = QuantumCircuit(2)
qc.hadamard(0)  # Cria superposição no qubit 0
qc.cnot(0, 1)   # Emaranha os qubits
qc.print_state()
```

### Superposição com um qubit
```python
qc = QuantumCircuit(1)
qc.hadamard(0)  # (|0⟩ + |1⟩)/√2
qc.print_state()
```

### Gate NOT quântico
```python
qc = QuantumCircuit(1)
qc.pauli_x(0)   # |0⟩ → |1⟩
qc.print_state()
```

## Executando os Exemplos

Para ver o simulador em ação:

```bash
python example.py
```

Para executar os testes:

```bash
python test_quantum_circuit.py
```

## Estrutura do Projeto

- `quantum_circuit.py` - Classe principal QuantumCircuit
- `example.py` - Exemplos de uso do simulador
- `test_quantum_circuit.py` - Testes básicos
- `requirements.txt` - Dependências do projeto

## API da Classe QuantumCircuit

### Construtor
- `QuantumCircuit(n_qubits)` - Inicializa circuito com n qubits no estado |0...0⟩

### Gates Quânticos
- `hadamard(qubit)` - Aplica gate Hadamard
- `pauli_x(qubit)` - Aplica gate Pauli-X (NOT)
- `cnot(control, target)` - Aplica gate CNOT

### Medições
- `measure(qubit=None)` - Mede qubit específico ou todos os qubits
- `get_state_probabilities()` - Retorna probabilidades de todos os estados

### Visualização
- `print_state()` - Imprime o estado atual
- `visualize_state()` - Mostra gráfico de barras das probabilidades

### Utilitários
- `reset()` - Reinicia para o estado |0...0⟩
- `get_circuit_history()` - Retorna histórico de gates aplicados

## Dependências

- **numpy** - Operações matemáticas e álgebra linear
- **qiskit** - Framework de computação quântica (para compatibilidade futura)
- **matplotlib** - Visualização de estados quânticos

## Limitações

Este é um simulador "ingênuo" destinado a fins educacionais. Limitações incluem:

- Simulação clássica (não é um computador quântico real)
- Exponencialmente caro em memória para muitos qubits
- Implementação básica sem otimizações avançadas
- Apenas gates básicos implementados

## Licença

MIT License - veja o arquivo LICENSE para detalhes.