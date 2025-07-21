## Phase-Field Method to Topology optimization 
In this project, we compare the performance of different PETSc TAO solvers using FEniCSx for phase-field topology optimization.

The project is organized into three main files:
- The implementation script
- A class that defines the optimization problem
- A utility file for plotting

The objective function consists of three terms:
- Compliance
- Perimeter regularization
- A volume penalty

The optimization is subject to a PDE constraint and a volume constraint.
