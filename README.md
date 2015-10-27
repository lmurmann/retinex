# Retinex C++ Implementation

This is an implementation of the Retinex algorithm for
intrinsic image decomposition. The provided code computes
image gradients, and assembles a sparse linear "Ax = b" 
system. The system is solved using Eigen.

## Dependencies

 - Eigen for sparse linear solve
 - For imread: OpenCV

## Example results

Input
![inputt](https://raw.github.com/lmurmann/retinex/master/img/input.jpg)

Reflectance
![reflectance](https://raw.github.com/lmurmann/retinex/master/img/reflectance.jpg)
