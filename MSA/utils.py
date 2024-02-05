import numpy as np
from functools import reduce
from itertools import product
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt


class MSAINFO:
    def __init__(self, seqs, p):
        """
        input
        -------------------
        seqs: the list of multi-sequences, e.g. ['AGG', 'ACGG', 'AT']
        p: the hyperparameter used for constraints
        """
        self.seqs = seqs
        self.num_seqs = len(self.seqs)
        self.S = self.get_seqs_length()
        self.L = max(self.S)
        self.encoding_dims = self.L * self.num_seqs
        self.weights = self.preprocessing()
        self.p = p
        
    def get_seqs_length(self,):
        return np.array([len(seq) for seq in self.seqs])
    
    def preprocessing(self):
        weights = {}
        for si in range(self.num_seqs):
            for sj in range(si+1, self.num_seqs):
                weights[(si, sj)] = np.zeros((self.L, self.L))
                for i, letter_i in enumerate(self.seqs[si]):
                    for j, letter_j in enumerate(self.seqs[sj]):
                        if letter_i == letter_j:
                            weights[(si, sj)][i,j] = -1
                        else:
                            weights[(si, sj)][i,j] = 1
        return weights


def getLoss(MSAINFO):
    
    def state2order(x):
        """
        x: should be a binary string, e.g. '111010011100', and the length should be N*L, 
            where N means the number of sequences, and L means the length of the longest sequence.
        """
        if isinstance(x, str):
            x = list(map(int, list(x)))
        seq2order = []
        x_o = np.reshape(x, (-1, MSAINFO.L))
        x_c = np.cumsum(x_o, axis=1)
        for i in range(x_o.shape[0]):
            x_1 = np.where(x_c[i,:]<=MSAINFO.S[i], 0, -1)
            x_2 = np.where(x_o[i,:]==1, 0, -1)
            x_3 = np.where((x_c[i,:]<=MSAINFO.S[i])&(x_c[i,:]>0), x_c[i,:]-1, 0)
            x_3 = np.multiply(x_3, x_o[i,:])
            each_seq = x_1 + x_2 + x_3
            seq2order.append(each_seq)
        seq2order = np.array(seq2order)
        seq2order_flag = np.where(seq2order>-1, 1, 0)
        return seq2order, seq2order_flag, x_o
    
    def get_score(x):
        seq2encoding, order_flag, x_reshape = state2order(x)
        score = 0
        for i in range(MSAINFO.num_seqs):
            s1 = seq2encoding[i, :]
            x_id = order_flag[i, :]
            for j in range(i+1, MSAINFO.num_seqs):
                s2 = seq2encoding[j, :]
                y_id = order_flag[j, :]
                
                w_ij = MSAINFO.weights[(i,j)]
                score += (x_id * y_id * w_ij[s1, s2]).sum()
        
        c = np.power(x_reshape.sum(axis=1) - MSAINFO.S, 2).sum()
        return score + MSAINFO.p * c
    
    return get_score


def getHamiltonian(MSAINFO):
    basis_space = gen_basis_space(MSAINFO.encoding_dims)
    get_score = getLoss(MSAINFO)
    Hamiltonian = np.zeros((2**MSAINFO.encoding_dims, 2**MSAINFO.encoding_dims))
    for i, basis_state in enumerate(basis_space):
        # basis_vec = state2vec(basis_state)
        eig_value = get_score(basis_state)
        Hamiltonian[i,i] = eig_value
        # Hamiltonian += eig_value * np.matmul(basis_vec, basis_vec.T)
    return Hamiltonian


def state2vec(x):
    basis_state = {
        '0' : np.array([[1], [0]]),
        '1' : np.array([[0], [1]])
    }
    return reduce(np.kron, [basis_state[xx] for xx in x])


def gen_basis_space(dims):
    res = []
    def dfs(path):
        if len(path) == dims:
            res.append(path)
            return
        dfs(path + '0')
        dfs(path + '1')
    dfs('')
    return res


Paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

def PaulisDecomposition(Hamiltonian: np.ndarray):
    order = len(Hamiltonian.shape)
    if order != 2:
        raise ValueError("The order of Hamiltonian is not equal to 2.")
    dims = Hamiltonian.shape[0]
    if Hamiltonian.shape != (dims, dims):
        raise ValueError("The shape of Hamiltonian is not like (dims, dims).")
    num_qubits = int(np.log2(dims))
    if not np.allclose(num_qubits - np.log2(dims), 0):
        raise ValueError("The dimension of Hamiltonian is not 2^{num_qubits}.")
    
    Decomposition = {}
    for paulis in product(Paulis.keys(), repeat=num_qubits):
        pauli_basis = reduce(np.kron, [Paulis[p] for p in paulis])
        coef = np.real_if_close(Hamiltonian.reshape(-1) @ pauli_basis.reshape(-1)) / dims
        if not np.allclose(coef, 0):
            Decomposition[''.join(paulis)] = coef
    return Decomposition


def getHamiltonianDiag(MSAINFO):
    basis_space = gen_basis_space(MSAINFO.encoding_dims)
    get_score = getLoss(MSAINFO)
    # Hamiltonian = np.zeros((2**MSAINFO.encoding_dims, 2**MSAINFO.encoding_dims))
    Hamiltonian = np.zeros(2**MSAINFO.encoding_dims)
    for i, basis_state in enumerate(basis_space):
        # basis_vec = state2vec(basis_state)
        eig_value = get_score(basis_state)
        Hamiltonian[i] = eig_value
        # Hamiltonian += eig_value * np.matmul(basis_vec, basis_vec.T)
    return Hamiltonian


PaulisIZ = {
        "I": np.array([1, 1]),
        "Z": np.array([1, -1]),
    }

def PaulisIZDecomposition(Hamiltonian: np.ndarray):
    """
    Hamiltonian: it should be a diagonal matrix.
    """
    dims = Hamiltonian.shape[0]
    num_qubits = int(np.log2(dims))
    if not np.allclose(num_qubits - np.log2(dims), 0):
        raise ValueError("The dimension of Hamiltonian is not 2^{num_qubits}.")
    
    Decomposition = {}
    for paulis in product(PaulisIZ.keys(), repeat=num_qubits):
        pauli_basis = reduce(np.kron, [PaulisIZ[p] for p in paulis])
        coef = np.real_if_close(Hamiltonian.reshape(-1) @ pauli_basis.reshape(-1)) / dims
        if not np.allclose(coef, 0):
            Decomposition[''.join(paulis)] = coef
    return Decomposition


def unparameterized2parameterized(circ, name='theta'):
    """
    used with "tq2qiskit" (from torchquantum.plugin import tq2qiskit)
    """
    circ_ = circ.copy()
    theta = ParameterVector(name, length=0)
    all_ins, params = [], []
    for ins in circ_._data:
        curr_ins_params = ins.operation.params
        if curr_ins_params:
            params.append(curr_ins_params)
            num_parameters = len(curr_ins_params)
            theta.resize(len(theta)+num_parameters)
            ins.operation.params = theta[-num_parameters:]
        all_ins.append(ins)
    return QuantumCircuit.from_instructions(all_ins), params


def draw_from_dict(dictdata, ideal_state=None, feasible_state=None, reps=None, is_save=False, heng=0):
    by_value = sorted(dictdata.items(), key=lambda x:x[1], reverse=True)
    x, y = [], []
    ideal_i, ideal_y = [], []
    feasible_i, feasible_y = [], []
    for i, d in enumerate(by_value):
        if ideal_state is not None and d[0] in ideal_state:
            ideal_i.append(i)
            ideal_y.append(d[1])
        if feasible_state is not None and d[0] in feasible_state:
            feasible_i.append(i)
            feasible_y.append(d[1])
        x.append(d[0])
        y.append(d[1])
    plt.figure(dpi=100, figsize=(9,9))
    if heng == 0:
        plt.bar(range(len(y)), y, color='grey', alpha=0.5, label='other')
        plt.xticks(range(len(y)), x, rotation=90, fontsize=14)
        plt.yticks(fontsize=18)
        plt.ylabel('counts', fontsize=24)
        plt.bar(feasible_i, feasible_y, color='orange', label='feasible')
        plt.bar(ideal_i, ideal_y, color='red', label='optimal')
    elif heng == 1:
        plt.barh(range(len(y)), y, color='grey', alpha=0.5, label='other')
        plt.yticks(range(len(y)), x, fontsize=14)
        plt.xticks(fontsize=18)
        plt.xlabel('counts', fontsize=24)
        plt.barh(feasible_i, feasible_y, color='orange', label='feasible')
        plt.barh(ideal_i, ideal_y, color='red', label='optimal')
    if reps is not None:
        plt.title(f"reps={reps}", fontsize=24)
    plt.legend(prop={'size': 24})
    if is_save:
        if reps:
            plt.savefig(f'reps_{reps}.png')
        else:
            plt.savefig('reps.png')
    plt.show()

if __name__ == "__main__":
    pass




