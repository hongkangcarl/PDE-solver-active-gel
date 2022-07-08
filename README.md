# Numerical solver of "active gel" PDEs
Numerical algorithms that solve "active gel" PDE on a 2D plane and on a unit sphere. Spectral methods are used, which transform the solutions into Fourier series in the 2D plane case and into spherical harmonics in the spherical case.

## Model equations

Active gel models are widely used to model actively contractile materials such as the actomyosin cortex of cells. The active gel is treated as a compressible viscous fluid, but with an active stress that depends on its density.

The force balance of the active stress, the viscous stress and the external drag determines the velocity field:

$$
\mu(\nabla^2 \mathbf v + \nabla(\nabla\cdot \mathbf v)) + \nabla \sigma_\text{act}(\rho) = \lambda \mathbf v
$$

And from the velocity field, the time evolution of the density field can be determined:

$$
\frac{\partial \rho}{\partial t} = - \nabla\cdot(\rho \mathbf v) + D\nabla^2 \rho + \frac{\rho_0 - \rho}{\tau_\text{turn}}
$$

These two equations form a self-contained system that can self-evolve. This system has intrinsic instability, since a region with high density exerts higher active stresses, which will pull more material into that region and further increase the density. Therefore, this model predicts formation of density patterns.

## Examples of numerical solutions

One solution on a 2D plane shows these dynamic patterns that evolve in time:

https://user-images.githubusercontent.com/63879978/178057055-0e5a05bf-1240-41e5-9678-a56227fa2687.mp4

Another solution on a unit spherical surface also shows similar dynamic patterns:

https://user-images.githubusercontent.com/63879978/178058456-6b139a8e-c885-4c10-b719-cc4fe43ed87c.mp4

## How to solve these complex PDEs numerically? Spectral methods!

Common numerical methods to solve differential equations include finite difference methods and spectral methods. Finite difference methods discretize the system into many points, and use the neighboring points compute the local solution. Spectral methods write the solutions of the differential equations into "basis functions" (for example, sin and cos on a 2D plane, and spherical harmonics on a spherical surface), and then solve for the coefficients.

Compared to finite difference methods that use only the local information to compute the solution, spectral methods use the global information, and thus are more accurate than the finite difference methods.

This repo contain two numerical programs to solve the active gel PDEs, one on a plane and the other one on a spherical surface, built from scratch using NumPy and SciPy. The planar case uses the Fast Fourier Transform (FFT) and the spherical case uses the Spherical Harmonic Transform (written in the SHT.py).
