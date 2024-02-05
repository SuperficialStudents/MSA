from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import TaperedPauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import Pauli
import numpy as np
from functools import reduce
import warnings
warnings.filterwarnings('ignore')


class VQE:
    def __init__(self, ansatz, backend=None, nshots=1024, **kwargs):
        self.ansatz = ansatz
        self.qubits = ansatz.num_qubits
        self.backend = backend if backend is not None else Aer.get_backend('qasm_simulator')
        self.nshots = nshots
        self.seed = kwargs.get('seeds', None)
        self.optimizers = {
            'sgd': self.sgd, 
            'momentum': self.momentum,
            'adam': self.adam
        }
        self.basis = {"0": np.array([1, 0]), "1": np.array([0, 1])}
    
    def sgd(self, lossFunc, getGrad, init_params, opt_params, **options):
        isAdaptive = options['adaptive'] if 'adaptive' in options else True
        
        params = init_params
        
        iterstep = 0
        loss = 1e4
        next_loss = lossFunc(params)
        while np.abs(loss - next_loss) > opt_params['threshold'] and iterstep < opt_params['maxiter']:
            loss = next_loss
            
            grad = getGrad(params)
            # if (iterstep+1) % 10 == 0:
            #     print(grad)
            
            lr = opt_params['lr'] * (iterstep + 1) ** (-0.602) if isAdaptive else opt_params['lr'] 
            params = params - lr * grad
            
            iterstep += 1
            next_loss = lossFunc(params)
        return next_loss, params
    
    def momentum(self, lossFunc, getGrad, init_params, opt_params, **options):
        isAdaptive = options['adaptive'] if 'adaptive' in options else True
        beta = options['beta'] if 'beta' in options else 0.9
        
        params = init_params
        last_grad = np.zeros_like(params)
        
        iterstep = 0
        loss = 1e4
        next_loss = lossFunc(params)
        while np.abs(loss - next_loss) > opt_params['threshold'] and iterstep < opt_params['maxiter']:
            loss = next_loss
            
            next_grad = getGrad(params)
            m_grad = last_grad * beta + next_grad * (1 - beta)
            
            lr = opt_params['lr'] * (iterstep + 1) ** (-0.602) if isAdaptive else opt_params['lr']
            params = params - lr * m_grad
            
            iterstep += 1
            last_grad = m_grad
            next_loss = lossFunc(params)
        return next_loss, params

    def adam(self, lossFunc, getGrad, init_params, opt_params, **options):
        isAdaptive = options['adaptive'] if 'adaptive' in options else True
        beta1 = options['beta1'] if 'beta1' in options else 0.9
        beta2 = options['beta2'] if 'beta2' in options else 0.999
        
        params = init_params
        m_grad, v_grad = np.zeros_like(params), np.zeros_like(params)
        
        iterstep = 0
        loss = 1e4
        next_loss = lossFunc(params)
        while np.abs(loss - next_loss) > opt_params['threshold'] and iterstep < opt_params['maxiter']:
            loss = next_loss
            
            iterstep += 1
            
            g_grad = getGrad(params)
            
            m_grad = beta1 * m_grad + (1 - beta1) * g_grad
            v_grad = beta2 * v_grad + (1 - beta2) * np.power(g_grad, 2)
            
            m_grad_avg = m_grad / (1 - beta1 ** iterstep)
            v_grad_avg = v_grad / (1 - beta2 ** iterstep)
            
            grad = m_grad_avg / (np.sqrt(v_grad_avg) + 1e-16)
            
            lr = opt_params['lr'] * (iterstep + 1) ** (-0.602) if isAdaptive else opt_params['lr'] 
            params = params - lr * grad

            # last_grad = m_grad
            
            next_loss = lossFunc(params)
        return next_loss, params

    def getEnergy(self, counts, Z_ops):
        for eigv in counts.keys():
            basis_tmp = reduce(np.kron, [self.basis[c] for c in eigv[::-1]])
            sign = (Z_ops @ basis_tmp) @ basis_tmp
            counts[eigv] = sign * counts[eigv]
        exp_val = sum(counts.values()) / self.nshots
        return exp_val

    def getLossAndGrad(self, Hamiltonian, **options):
        jac = options.get('jac', 'param-shift')
        if jac not in set({'param-shift', 'spsa'}):
            raise ValueError(f"The parameter jac value {jac} is invalid, please input (param-shift, spsa).")

        def getLoss(params):
            if isinstance(Hamiltonian, (TaperedPauliSumOp, PauliSumOp)):
            # if isinstance(Hamiltonian, TaperedPauliSumOp) or isinstance(Hamiltonian, PauliSumOp):
                qubit_op = Hamiltonian._primitive
            else:
                qubit_op = Hamiltonian

            if isinstance(params, list):
                total_energy = [0.0] * len(params)
            else:
                total_energy = 0.0
            coef_identify = 0.0
            for coef, paulis in zip(qubit_op.coeffs, qubit_op.paulis):
                if paulis == Pauli('I'*self.qubits):
                    coef_identify = coef
                    continue
                qc_pauli = QuantumCircuit(self.qubits)
                Z_ops = ''
                for i, pauli in enumerate(str(paulis)[::-1]):
                    if pauli in ['X', 'Y']:
                        qc_pauli.h(i)
                    if pauli == 'Y':
                        qc_pauli.sdg(i)
                    if pauli == 'I':
                        Z_ops += 'I'
                    else:
                        Z_ops += 'Z'
                Z_ops = Pauli(Z_ops).to_matrix()
                ansatz = self.ansatz.compose(qc_pauli, range(self.qubits))
                ansatz.measure_all()

                if isinstance(params, list):
                    ansatz = [ansatz.bind_parameters(param) for param in params]
                else:
                    ansatz = ansatz.bind_parameters(params)

                results = self.backend.run(ansatz, seed_simulator=self.seed, shots=self.nshots).result()
                counts = results.get_counts()

                if isinstance(counts, list):
                    total_energy = [total_energy[i] + coef * self.getEnergy(cts, Z_ops) for i, cts in enumerate(counts)]
                else:
                    energy = self.getEnergy(counts, Z_ops)
                    total_energy += coef * energy

            return np.real_if_close(np.asarray(total_energy) + coef_identify)
        
        def getGrad(params):
            """
            the parameter-shift rule
            """
            dim = params.size
            
            ei = np.identity(dim)
            plus_shifts = (params + np.pi / 2 * ei).tolist()  # boradcast
            minus_shifts = (params - np.pi / 2 * ei).tolist()
            
            doubleLoss = getLoss(plus_shifts + minus_shifts)  # (2*num_qubits, num_qubits)
            grad = (doubleLoss[:dim] - doubleLoss[dim:]) / 2
            return grad
        
        def getGradbySPSA(params):
            alpha = options['alpha'] if 'alpha' in options else 0.5
            c = options['c'] if 'c' in options else 0.02
            
            dim = self.ansatz.num_parameters
            
            delta = np.random.binomial(1, 0.5, size=dim)
            delta = np.where(delta==0, -1, delta)
            
            plus_shifts = params + c * delta
            minus_shifts = params - c * delta
            
            doubleLoss = getLoss([plus_shifts, minus_shifts])
            grad = (doubleLoss[0] - doubleLoss[1]) / (2 * c * delta)
            return grad
        
        return (getLoss, getGradbySPSA) if jac == 'spsa' else (getLoss, getGrad)
    
    def getMinEnergy(self, Hamiltonian, init_params, optimizer, **options):

        lossFunc, getGrad = self.getLossAndGrad(Hamiltonian, **options)
        if optimizer in set({'sgd', 'momentum', 'adam'}):
            threshold = options['threshold'] if 'threshold' in options else 1e-8
            maxiter = options['maxiter'] if 'maxiter' in options else 100
            lr = options['lr'] if 'lr' in options else 0.05
            opt_params = {'threshold': threshold, 'maxiter': maxiter, 'lr': lr}
            
            energy, params = self.optimizers[optimizer](lossFunc, getGrad, init_params, opt_params, **options)
            return energy, params
        else:
            raise ValueError(f"The parameter optimizer value {optimizer} is invalid, please input (sgd, momentum or adam).")
    
    def run(self, qubit_op, init_params=None, optimizer=None, **options):
        if init_params is None:
            init_params = np.random.randn(self.ansatz.num_parameters)
        
        minEnergy, params = self.getMinEnergy(qubit_op, init_params, optimizer, **options)
        
        return minEnergy, params


if __name__ == '__main__':
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
    from qiskit_nature.second_q.transformers import FreezeCoreTransformer
    from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
    from qiskit.opflow.converters import TwoQubitReduction
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit.circuit.library import EfficientSU2

    def get_qubit_op(molecule, remove_orbitals=None):
        driver = PySCFDriver.from_molecule(molecule, basis='sto3g')
        problem = driver.run()
        transform = FreezeCoreTransformer(remove_orbitals)
        problem = transform.transform(problem)
        hamiltonian = problem.hamiltonian.second_q_op()

        mapper = ParityMapper()
        converter = QubitConverter(mapper, two_qubit_reduction=True)
        reducer = TwoQubitReduction(problem.num_particles)

        qubit_op = converter.convert(hamiltonian)
        qubit_op = reducer.convert(qubit_op)

        return hamiltonian, qubit_op


    def getAnsatzCircuit(num_qubits, entanglement='linear'):
        qc = QuantumCircuit(num_qubits)

        # the init state
        # 待续

        # EfficientSU2
        qc.append(EfficientSU2(num_qubits=num_qubits, entanglement=entanglement), range(num_qubits))

        # qc.measure_all()

        return qc.decompose().decompose()


    molecule = MoleculeInfo(['H', 'H'], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], charge=0, multiplicity=1)
    _, qubit_op = get_qubit_op(molecule, remove_orbitals=None)

    ansatz = getAnsatzCircuit(qubit_op.num_qubits, entanglement='linear')
    params = np.random.random(ansatz.num_parameters) * 0.1
    
    vqe = VQE(ansatz, backend=None, nshots=1024)
    energy, params = vqe.run(qubit_op, init_params=params, optimizer='adam', jac='spsa', c=0.02, maxiter=200, lr=1.0, isAdaptive=True, threshold=1e-15)
    print(f' Energy: {energy}\n optimal parameter:\n {params}')