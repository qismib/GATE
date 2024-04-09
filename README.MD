[![Python Version](https://img.shields.io/badge/Python-3-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45.1-blue.svg?style=flat-square&logo=qiskit&logoColor=white)](https://www.ibm.com/quantum/qiskit)

## GATE (Gaps Analysis via Time Evolution)

This tool computes the energy gaps of the SU(2) pure-gauge Yang-Mills theory on 2 distinct 2D spatial lattices, respectively 2 and 3 plaquettes long, with periodic boundary conditions, using quatum algorithms. See ["Real time evolution and a traveling excitation in SU(2) pure gauge theory on a quantum computer"](https://arxiv.org/abs/2210.11606), by S.A. Rahman et al., for a review of the problem. At first, the Variational Quantum Eigensolver (VQE) is used to obtain estimates of the eigenstates of the theory. Their time evolution is processed with a Discrete Fourier Transform to measure the energy gaps.
The graphs generated with this code have been included in the thesis "On the spectrum of the SU(2) Yang-Mills theory on a quantum computer" by Stefano Grippo.

## Contents

- `data/`: Some data provided for convenience. The scripts in this repository can either read this data to generate the corresponding graphs, or they can execute new simulations to generate new data from scratch. Notice that some simulations can be quite time consuming.

- `gaugeSpectrum/`: The scripts for the eigenstates estimation. 
	- `vqeMP.py`: finds the eigenstates by minimizing the variance of the states with respect to a given Hamiltonian (see ["Variational quantum eigensolvers by variance minimization"](https://arxiv.org/abs/2006.15781)). Set `d` to `4` for the 2-plaquette simulation, or to `8` for the 3-plaquette case;
	- `analisiDati.py`: generates visual information about the estimated eigenvectors for the 2-plaquette system, like their superposition with the true eigenvectors and an estimate of the eigenvalues of the Hamiltonian. The pictures are then put in `data/misureAutovaloriMPQ/`;
	- `analisiDati3PL.py`: the same as `analisiDati.py` for the 3-plaquette system;
	- `moduliHR.py`, `moduliHRCircuiti.py`, `valoreApettazionePauli.py`: contain useful functions;
	- `backupMP.py`: saves the eigenstates during the execution of the VQE.
	
- `plaquettesEvolution/`: The scripts for the time evolution. 
	- `Evoluzione2Pl.py`: outputs the graphs of the time evolution of the estimated eigenvectors for the 2-plaquette system. The pictures are saved in `data/frequenze2PL`;
	- `Evoluzione3Pl.py`: the same as `Evoluzione2Pl.py` for the 3-plaquette system;
	- `evConErrori.py`: outputs the graphs of a time evolution and its DFT when quantum fluctuations are taken into account. The pictures are saved in `data/errori`;
	- `analisiFrequenze.py`: generates the graphs of the DFT of the time evolution of the eigenvectors. Then it estimates the energy gaps from the peaks. Set `nPlacchette` to `2` or `3` to choose the number of plaquettes of the system.

- `LICENSE`: Project license.
- `README.MD`: This file.

