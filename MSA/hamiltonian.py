import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from itertools import product
from .utils import MSAINFO, getHamiltonianDiag

class Hamiltonian2PauliString(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.PaulisIZ = {
            'I': torch.tensor([1, 1], dtype=torch.float32, device='cuda'),
            'Z': torch.tensor([1, -1], dtype=torch.float32, device='cuda')
        }

    def forward(self, Hamiltonian):
        """
        Hamiltonian: it should be a diagonal matrix.
        """
        dims = Hamiltonian.shape[0]
        num_qubits = int(np.log2(dims))
        if not np.allclose(num_qubits - np.log2(dims), 0):
            raise ValueError("The dimension of Hamiltonian is not 2^{num_qubits}.")

        Hamiltonian = Hamiltonian.float()
        Decomposition = {}
        for paulis in product(self.PaulisIZ.keys(), repeat=num_qubits):
            pauli_basis = reduce(torch.kron, [self.PaulisIZ[p] for p in paulis])
            coef = torch.dot(Hamiltonian,  pauli_basis) / dims
            Decomposition[''.join(paulis)] = float(coef.cpu().numpy())
        return Decomposition

    
if __name__ == "__main__":
    seqs = ['ACGG', 'AGG', 'AT']
    msa = MSAINFO(seqs, p=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    H2P = Hamiltonian2PauliString()
    h2p = H2P.to(device)

    H = torch.tensor(getHamiltonianDiag(msa))
    H = H.to(device)

    paulis_operators = h2p(H)