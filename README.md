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

2. xTB. Here there is working GFN1-xTB energy calculation, but the code does not work with autodifferentiation. The uploaded code tries
to use the library Zygote, but I also tried others and hit similar problems. 
My initial attempts did not work because the various parts of the xTB code were not careful enough with building various arrays such that
the AutoDiff libraries were confused. I rewrote many functions to involve straightforward assignment/comprehension statements such that 
arrays are built directly. My goal was to create a function "Lagrangian" which would take the final orbital coefficients, final orbital
energies, and the overlap matrix, charges, and so on, and re-evaluate the energy including the orthogonal terms. Then I would apply
AutoDiff to that function. The code new_xtb_julia.jl successfully calculates the energy (it requires to be run with "julia new_xtb_julia.jl {filename}.xyz",
I attach the required parameters file parameters3.dat and the sample input methanol.xyz) but then crashes. The LAgrangian energy
does NOT match the SCF energy so evenm the Lagrangian is built incorrectly. I abandoned the project at this point because I could not
get the AutoDiff to work. I am confident the Lagrangian can be fixed quite easily. The "hard nut" at this point for me is getting
the AutoDiff to work.
Update: I noticed an error in new_xtb_julia concerning the way the Lagrangian was expressed. After revising this, I get the modified code
newer_xtb_julia, in which the Lagrangian now comes out correct, equal to the energy. The automatic differentiation still does not work.



