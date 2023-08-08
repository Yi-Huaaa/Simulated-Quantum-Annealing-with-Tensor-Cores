# Simulated Quantum Annealing (SQA)

## Introduction
Inspired by quantum annealing, simulated quantum annealing (SQA) mimics quantum tunneling effects on classical computers to perform annealing through a path-integral Monte Carlo simulation, which increases the potential to find the global optima faster than traditional annealing algorithms for large-size combinatorial optimization problems while today’s quantum annealing systems are of a limited number of qubits. As previous studies have accelerated SQA with Graphics Processing Unit (GPU) and specialized hardware such as Field Programmable Gate Array (FPGA), we propose an innovative parallelizing strategy called hierarchical update to vastly improve the efficiency of parallel computing, which is capable of accelerating state-of-the-art SQA implementations further by 7X-47.2X based on our case studies. Furthermore, we develop a tensorizing scheme to leverage the Tensor Cores on modern GPUs to deliver up to 1.83X of additional speedup. Overall, our work solves fully-connected Ising models faster than any previous SQA work. Our solution outperforms existing GPU-based solutions by 86.6X and FPGA-based solutions by 14X.

## Paper and Reference
* Our paper: [Accelerating Simulated Quantum Annealing with GPU and Tensor Cores](https://link.springer.com/chapter/10.1007/978-3-031-07312-0_9)
* Tohoku_fully_GPU: [A GPU-Based Quantum Annealing Simulator for Fully-Connected Ising Models Utilizing Spatial and Temporal Parallelism](https://ieeexplore.ieee.org/abstract/document/9057502)
* Tohoku_fully_FPGA: [Highly-Parallel FPGA Accelerator for Simulated Quantum Annealing](https://ieeexplore.ieee.org/document/8918417)
* Tohoku_king_graph: [An Ising Computer Based on Simulated Quantum Annealing by Path Integral Monte Carlo Method](https://ieeexplore.ieee.org/abstract/document/8123652?casa_token=j0NShE6p6_IAAAAA:Bq8VAVpfQEJlaEyyi-VFDN6NSTXV4b_ONO9bb3Jd11zilw-iEb-J5ckh5IUsXDgMYZ61Lb1A)
* [Accelerator Architecture for Simulated Quantum Annealing Based on Resource-Utilization-Aware Scheduling and its Implementation Using OpenCL](https://ieeexplore.ieee.org/abstract/document/8923263)
* About TFIM model: [Quantum Annealing in the Transverse Ising Model](https://arxiv.org/abs/cond-mat/9804280)
* [cublas](https://developer.nvidia.com/cublas)


## Some Parameter Defined in the Program
1. M: #Trotter
2. N: #Spin
3. $J_{i,j}$: coupling of $spin_{i}$ and $spin_{j}$
4. $\sigma_{i,k}$: $spin_{i}$ located on $#k$ trotter
Notes: For a deeper comprehension of the Hamiltonian and TFIM models, please refer to the paper and the accompanying references.

---

## Acceleration Strategy
* Detail [Link](https://yi-huaaa.github.io/2022/05/10/Accelerating%20Simulated%20Quantum%20Annealing%20with%20GPU%20and%20Tensor%20Cores/)
	- It elaborates on the operational mechanics of the mentioned SQA as delineated within the contents of our paper.
	- The "Analysis for HU" section in the link provides an account of the disparities in total computational complexity introduced by including the Hierarchical Update (HU) strategy, incorporating both additive and multiplicative computations. Through the comparative assessment of these computational disparities, a clearer comprehension of the underlying rationale for the acceleration of the SQA due to HU can be attained.
	- The provided link offers a pre-recorded video of the conference presentation and comprehensive reference materials in the form of presentation slides.

## Accuracy
### Adjustable Parameters
*  Including:
	*  $N$: #spin
 	*  $M$: #trotter
  	*  $M\_2$
 	*  $TIMES$: #round
 	*  $STEP$: annealing steps per round
 	*  ${G0}$ and ${beta}$ $\to$ according to Parameter configuration reference:
		- [An Ising computer based on simulated quantum annealing by path integral Monte Carlo method](https://ieeexplore.ieee.org/abstract/document/8123652)
* My initial approach to determine the optimal parameters settings was brute force.
	

### Something Needs to Know
* Our paper exclusively addresses the MAX-CUT problem on a fully connected Ising model; therefore, to tackle different Ising problems, one would need to independently map the QUBO form to the Ising problem and subsequently adapt and rewrite the Hamiltonian computation function accordingly in the program.

* The program exclusively supports the MatrixMarket format for text input, along with the MAX-CUT benchmark dataset Gset.
	- Gset is a collection of instances for the MAX-CUT problem.
 	- [Gset](https://web.stanford.edu/~yyye/yyye/Gset/)

* My approach was entirely brute-force to find out the best spin configuration in nature, as the flipping of spins was based on random probabilities. Consequently, it was possible for the flipping to occur even in regions with higher energy. Therefore, my initial strategy involved storing spin configurations in GPU global memory whenever a configuration with lower energy was obtained. One drawback of this approach was its slowdown due to the time-consuming data transfer process.
* The current program utilizes Tensor cores, but you can consider implementing an additional function that runs purely on CUDA cores.
* Depending on the GPU architecture, varying $M_2$ values are required, resulting in differing speeds. For a comprehensive analysis of $M_2$, please refer to the paper.


## More Improvement
* Currently, Tensor cores are utilized through calls to the cuBLAS library. However, it's also possible to implement custom PTX code without relying on libraries, aiming to optimize the program's pipeline as much as possible.
* Since spins can only take on two possibilities, ${+1}$ or ${-1}$, you can utilize the `int2` data type to store a spin. However, there isn't currently a library available to directly handle this. Therefore, the program needs to handle this aspect independently.

---

## Makefile usage

```
cd SQA-fully/  # should work in this directory
make           # compile all binaries
make speed     # test speed for different N,M
```
## run.sh
* three files inside
* parameters:
  * N = 1024 ~ 32768
  * M = 1 - 512
  * M2 = 1 - 1024
* run.sh -> get exe files
* com.sh -> get running time
* cut.sh -> grep out "avg time, gemm, judge" times
