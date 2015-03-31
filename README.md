basis_variance
==============

Tools to analyze basis function data from YANK simulations written with a special alchemy.py script.

Special version of YANK's alchemy.py script required to run. This script has been included in the 
directory. 

We recommend you look at the auto_schedule branch to see a version of the code which explicilty 
accounts for the cap potential on repulsive Lennard-Jones and accepts multiple alchemical schedules.

This is developmental code to test many schedules and alchemical switches. It is not optimized or 
intended for full production simulations, although it may work for them just as well. 

The Shirts Group is working on implementing these methods in proper simulation packages.

ncdata.py
==============
Module to load in and read the .nc files created by YANK with the special alchemy.py alchemical factory.

linfunctions.py
==============
Small module of functions to compute the basis functions values for given parameter. Creates classes 
which house functions to invoke the various switches, and their exact/numeric inverses on their 
[0,1] domain.

basisvariance.py
==============
Primary analysis scrit. Takes a complex and vacuum ncdata.py object + a class from linfunctions.py set of switches.
Allows computation of variances, dudl, and free energies from the ncdata classes.
