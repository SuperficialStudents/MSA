import numpy as np
import torch
import torchquantum as tq
from typing import Any, Iterable, List
from torchquantum.util import pauli_string_to_matrix
from torchquantum.algorithm.hamiltonian import parse_hamiltonian_file

def qiskit2tq_op_history(circ):
    if getattr(circ, "_layout", None) is not None:
        try:
            p2v_orig = circ._layout.final_layout.get_physical_bits().copy()
        except:
            p2v_orig = circ._layout.get_physical_bits().copy()
        p2v = {}
        for p, v in p2v_orig.items():
            if v.register.name == "q":
                p2v[p] = v.index
            else:
                p2v[p] = f"{v.register.name}.{v.index}"
    else:
        p2v = {}
        for p in range(circ.num_qubits):
            p2v[p] = p

    ops = []
    for gate in circ.data:
        op_name = gate[0].name
        wires = list(map(lambda x: x.index, gate[1]))
        wires = [p2v[wire] for wire in wires]
        # sometimes the gate.params is ParameterExpression class
        init_params = (
            list(map(float, gate[0].params)) if len(gate[0].params) > 0 else None
        )
        # print(op_name,)

        if op_name in [
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "sx",
            "cx",
            "cz",
            "cy",
            "swap",
            "cswap",
            "ccx",
        ]:
            ops.append(
                {
                "name": op_name,  # type: ignore
                "wires": np.array(wires),
                "params": None,
                "inverse": False,
                "trainable": False,
            }
            )
        elif op_name in [
            "rx",
            "ry",
            "rz",
            "rxx",
            "xx",
            "ryy",
            "yy",
            "rzz",
            "zz",
            "rzx",
            "zx",
            "p",
            "cp",
            "crx",
            "cry",
            "crz",
            "u1",
            "cu1",
            "u2",
            "u3",
            "cu3",
            "u",
            "cu",
        ]:
            ops.append(
                {
                "name": op_name,  # type: ignore
                "wires": np.array(wires),
                "params": init_params,
                "inverse": False,
                "trainable": True
            })
        elif op_name in ["barrier", "measure"]:
            continue
        else:
            raise NotImplementedError(
                f"{op_name} conversion to tq is currently not supported."
            )
    return ops


class VQE(object):   
    def __init__(self, hamil, ansatz, train_configs) -> None:

        self.ansatz = ansatz
        self.n_wires = ansatz.n_wires
        
        self.train_configs = train_configs
        self.n_epochs = self.train_configs.get("n_epochs", 100)
        self.n_steps = self.train_configs.get("n_steps", 10)
        self.optimizer_name = self.train_configs.get("optimizer", "Adam")
        self.scheduler_name = self.train_configs.get("scheduler", "CosineAnnealingLR")
        self.lr = self.train_configs.get("lr", 0.1)
        self.device = self.train_configs.get("device", "cpu")

        self.hamil = hamil.to(self.device)
        self.ansatz = self.ansatz.to(self.device)

    def get_expval(self, qdev):
        
        states = qdev.get_states_1d()
        expval = torch.dot(torch.multiply(states.conj().flatten(), self.hamil), states.flatten()).real
        return expval

    def get_loss(self):
        """Calculate the loss function.
        
        Returns:
            float: loss value
        """
        
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=1,
            device=self.device,
        )
        self.ansatz(qdev)
        expval = self.get_expval(qdev)
        return expval
    
    def train(self):
        """Train the VQE model.

        Returns:
            float: final loss value
        """
        
        optimizer = getattr(torch.optim, self.optimizer_name)(self.ansatz.parameters(), lr=self.lr)
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, T_max=self.n_epochs)
        loss = None
        for epoch in range(self.n_epochs):
            for step in range(self.n_steps):
                loss = self.get_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
            lr_scheduler.step()
        return loss.detach().cpu().item()


## if wanna use the VQE in torchquantum.algorithm, the hamiltonian should be used.
class Hamiltonian(object):
    """Hamiltonian class."""
    def __init__(self,
                 coeffs: List[float],
                 paulis: List[str],
                 endianness: str = "big",
                 ) -> None:
        """Initialize the Hamiltonian.
        Args:
            coeffs: The coefficients of the Hamiltonian.
            paulis: The operators of the Hamiltonian, described in strings.
            endianness: The endianness of the operators. Default is big. Qubit 0 is the most significant bit.
        Example:

        .. code-block:: python
            coeffs = [1.0, 1.0]
            paulis = ["ZZ", "ZX"]
            hamil = tq.Hamiltonian(coeffs, paulis)

        """
        if endianness not in ["big", "little"]:
            raise ValueError("Endianness must be either big or little.")
        if len(coeffs) != len(paulis):
            raise ValueError("The number of coefficients and operators must be the same.")
        for op in paulis:
            if len(op) != len(paulis[0]):
                raise ValueError("The length of each operator must be the same.")
            for char in op:
                if char not in ["X", "Y", "Z", "I"]:
                    raise ValueError("The operator must be a string of X, Y, Z, and I.")
        
        self.n_wires = len(paulis[0])
        self.coeffs = coeffs
        self.paulis = paulis
        self.endianness = endianness
        if self.endianness == "little":
            self.paulis = [pauli[::-1] for pauli in self.paulis]
    
    @property
    def hamil_info(self):
        hamil_list = []
        for coef, pauli in zip(self.coeffs, self.paulis):
            if not np.allclose(coef, 0):
                hamil_list.append({"pauli_string": pauli, "coeff": coef})
        return {"hamil_list": hamil_list}

    @property
    def matrix(self) -> torch.Tensor:
        """Return the matrix of the Hamiltonian."""
        return self.get_matrix()
    
    def get_matrix(self) -> torch.Tensor:
        """Return the matrix of the Hamiltonian."""
        matrix = self.coeffs[0] * pauli_string_to_matrix(self.paulis[0])
        for coeff, pauli in zip(self.coeffs[1:], self.paulis[1:]):
            matrix += coeff * pauli_string_to_matrix(pauli)

        return matrix
    
    def __repr__(self) -> str:
        """Return the representation string."""
        return f"{self.__class__.__name__}({self.coeffs}, {self.paulis}, {self.endianness})"
    
    def __len__(self) -> int:
        """Return the number of terms in the Hamiltonian."""
        return len(self.coeffs)
    
    @classmethod
    def from_file(cls, file_path: str) -> Any:
        """Initialize the Hamiltonian from a file.
        Args:
            file_path: The path to the file.
        Example:

        .. code-block:: python
            hamil = tq.Hamiltonian.from_file("hamiltonian.txt")

        Example of the hamiltonian.txt file:
        h2 bk 2
        -1.052373245772859 I0 I1
        0.39793742484318045 I0 Z1
        -0.39793742484318045 Z0 I1
        -0.01128010425623538 Z0 Z1
        0.18093119978423156 X0 X1            

        """
        hamil_info = parse_hamiltonian_file(file_path)
        return cls(hamil_info["coeffs"], hamil_info["paulis"])    


if __name__ == "__main__":
    pass













