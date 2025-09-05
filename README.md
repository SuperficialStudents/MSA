# A Hybrid-Querying Quantum Optimization Model Validated with 16-qubits on an Ion Trap Quantum Computer for Life Science Applications
Quantum Computing for Multi-Sequences Alignment

The algorithm is implemented on both CPU and GPU, where the CPU version is based on qiskit; The GPU version is based on torchquantum.

## File
File `Hamiltonian` holds the Pauli string of the Hamiltonian;

File `MSA` contains the required core code, where \
    `hamiltonian.py` is used to calculate the Hamiltonian; \
    `tqm_fix.py` is the code from torchquantum; \
    `VQE_v4.py` is only used on the CPU.

## Installtion
Files `quantum.txt` & `torchquantum.txt` are the environment configurations required to run on the CPU and GPU, respectively.

## Example
Files `msa_cpu.ipynb` & `msa_gpu.ipynb` can be utilized directly to execute the corresponding examples.


## Article Link
- [Read the full article on arXiv](https://arxiv.org/abs/2506.01559)

## Citation
```
@article{chen2025hqqubo,
  title={hqQUBO: A Hybrid-querying Quantum Optimization Model Validated with 16-qubits on an Ion Trap Quantum Computer for Life Science Applications},
  author={Chen, Rong and Mei, Quan-Xin and Zhao, Wen-Ding and Yao, Lin and Yang, Hao-Xiang and Zhang, Shun-Yao and Chen, Jiao and Li, Hong-Lin},
  journal={arXiv preprint arXiv:2506.01559},
  year={2025}
}
```

## License

This project is open-sourced under the MIT license.
