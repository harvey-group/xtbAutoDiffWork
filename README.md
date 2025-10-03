This repository contains a variety of files relevant for doing AutoDiff work in Julia.

1. MM example. This WORKS.
A first code is contained in WorkingAutodiffMM.jl. This code evaluates the MM energy of a molecule, with the 'tiny' forcefield, and optimizes the structure.
The code is currentluy hardwired to run for a test case, cyclohexane, with initial coordinates in cyclohexane.mol2.
Parameters are hardwired to be read from tiny.parameters
The code does the following (see function 'main' in lines 350 and following:
a) Reads the coordinates and sets up an object of type Molec containing coordinates, lists of bonded pairs, etc.
b) reads the parameters
c) Prints out some stuff about the molecule.
d) Evaluates the Stretching energy with function StretchEnergy
e) Same for Bending, Torsional and vdW energy
f) The container function TotalEnergy is called to evaluate the total energy
g) Creates a "Closure" which reads molecular specification and MM terms as parameters.
h) This closure is in turn used to create the function "DirectEnergy" which takes as only input the coordinates.
i) The Julia AutoDiff library Zygote then can be used to compute the gradient, and this is evaluated and printed for testing purposes.
j) The function GeomOpt runs geometry optimization with BFGS, using the DirectEnergy / Zygote technique to get gradients.


