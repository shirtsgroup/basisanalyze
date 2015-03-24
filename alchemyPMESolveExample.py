#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Alchemical factory for free energy calculations that operates directly on OpenMM swig System objects.

DESCRIPTION

This module contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

The code in this module operates directly on OpenMM Swig-wrapped System objects for efficiency.

EXAMPLES

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

* Can we store serialized form of Force objects so that we can save time in reconstituting
  Force objects when we make copies?  We can even manipulate the XML representation directly.
* Allow protocols to automatically be resized to arbitrary number of states, to 
  allow number of states to be enlarged to be an integral multiple of number of GPUs.
* Add GBVI support to AlchemicalFactory.
* Add analytical dispersion correction to softcore Lennard-Jones, or find some other
  way to deal with it (such as simply omitting it from lambda < 1 states).
* Deep copy Force objects that don't need to be modified instead of using explicit 
  handling routines to copy data.  Eventually replace with removeForce once implemented?
* Can alchemically-modified System objects share unmodified Force objects to avoid overhead
  of duplicating Forces that are not modified?

NOTE

This is a modified version to test out a different potential energy function other than the Soft-Core models.
Please be aware of this if using in production.

This file in particular has an example of solving for the PME component of a ligand in an alchemical 
transformation

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import numpy
import copy
import time

import simtk.openmm as openmm
import simtk.unit as units

from sets import Set

#=============================================================================================
# AlchemicalState
#=============================================================================================

class AlchemicalState(object):
    """
    Alchemical state description.
    
    These parameters describe the parameters that affect computation of the energy.

    CAUTION: This variant is designed to support separate lambda for both repulsion and van der Waals forces, so there is an extra term in the class.

    TODO

    * Rework these structure members into something more general and flexible?
    * Add receptor modulation back in?
    
    """
        
    def __init__(self, relativeRestraints, ligandElectrostatics, ligandAttraction, ligandRepulsion, ligandCapToFull, ligandTorsions, ligandPMEFull=None):
        """
        Create an Alchemical state

        relativeRestraints (float)        # scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near pocket)
        ligandElectrostatics (float)      # scaling factor for ligand charges, intrinsic Born radii, and surface area term
        ligandPMEFull (float)              #Scaling factor of the PME component of the system
        ligandAttraction (float)        # scaling factor for attractive ligand Lennard-Jones well depth and radius
        ligandRepulsion (float)        # scaling factor for repulsive ligand Lennard-Jones well depth and radius
        ligandCapToFull (float)        # Scaling factor of the capped potential to the full uncapped potential
        ligandTorsions (float)            # scaling factor for ligand non-ring torsions

        """

        self.annihilateElectrostatics = True
        self.annihilateLennardJones = False

        self.relativeRestraints = relativeRestraints
        self.ligandElectrostatics = ligandElectrostatics
        self.ligandPMEFull = ligandPMEFull
        self.ligandAttraction = ligandAttraction
        self.ligandRepulsion = ligandRepulsion
        self.ligandCapToFull = ligandCapToFull
        self.ligandTorsions = ligandTorsions

        return

    def state(self):
        print "Restraints: %.2f" % self.relativeRestraints
        print "Electrostatics: %.2f" % self.ligandElectrostatics
        print "PME: %.2f" % self.ligandPMEFull
        print "Attraction: %.2f" % self.ligandAttraction
        print "Repulsion: %.2f" % self.ligandRepulsion
        print "Cap: %.2f" % self.ligandCapToFull
        print "Torsion: %.2f" % self.ligandTorsions
        return


#=============================================================================================
# AbsoluteAlchemicalFactory
#=============================================================================================

class AbsoluteAlchemicalFactory(object):
    """
    Factory for generating OpenMM System objects that have been alchemically perturbed for absolute binding free energy calculation.

    EXAMPLES
    
    Create alchemical intermediates for 'denihilating' one water in a water box.
    
    >>> # Create a reference system.
    >>> from simtk.pyopenmm.extras import testsystems
    >>> [reference_system, coordinates] = testsystems.WaterBox()
    >>> # Create a factory to produce alchemical intermediates.
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
    >>> # Get the default protocol for 'denihilating' in solvent.
    >>> protocol = factory.defaultSolventProtocolExplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.

    >>> # Create a reference system.
    >>> from simtk.pyopenmm.extras import testsystems
    >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
    >>> # Create a factory to produce alchemical intermediates.
    >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
    >>> ligand_atoms = range(2603,2621) # p-xylene
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
    >>> protocol = factory.defaultComplexProtocolImplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    """    

    # Factory initialization.
    def __init__(self, reference_system, ligand_atoms=[]):
        """
        Initialize absolute alchemical intermediate factory with reference system.

        ARGUMENTS

        reference_system (System) - reference system containing receptor and ligand
        ligand_atoms (list) - list of atoms to be designated 'ligand' -- everything else in system is considered the 'environment'
        
        """

        # Store serialized form of reference system.
        self.reference_system = copy.deepcopy(reference_system)

        # Store copy of atom sets.
        self.ligand_atoms = copy.deepcopy(ligand_atoms)
        
        # Store atom sets
        self.ligand_atomset = Set(self.ligand_atoms)

        return

    @classmethod
    def defaultComplexProtocolImplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """
        import pdb
        pdb.set_trace()
        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.75, 1.00, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.50, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.25, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.95, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.90, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.80, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.70, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.60, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.40, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.30, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.20, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.10, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultComplexProtocolExplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """

        alchemical_states = list()
        #                                        Rest  Elec  Attr  Repu  Cap   Tor PME
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=1.00)) # fully interacting, Changing EPA
        alchemical_states.append(AlchemicalState(0.00, 0.66, 0.66, 1.00, 1.00, 1., ligandPMEFull=0.66)) 
        alchemical_states.append(AlchemicalState(0.00, 0.33, 0.33, 1.00, 1.00, 1., ligandPMEFull=0.33)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 1.00, 1., ligandPMEFull=0.00)) # EPA off, capping next
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 0.00, 1., ligandPMEFull=0.00)) # discharged, uncapped
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 0.00, 1., ligandPMEFull=0.00)) #Attractive Off, R capped
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.90, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.80, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.70, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.60, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.50, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.40, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.30, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.20, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.10, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.00, 0.00, 1., ligandPMEFull=0.00)) #Charge off, LJ fully decoupled
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=0.00)) # ACR to solve for A
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=1.00)) # PME only
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=0.50)) # Different PME to isolate the solvent-solvent vs solvent-solute interactions
        #alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 0.00, 1., ligandPMEFull=0.00)) # discharged, repulsive basis only
        
        return alchemical_states

    @classmethod
    def defaultSolventProtocolImplicit(cls):
        """
        Return the default protocol for ligand in solvent.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """

        alchemical_states = list()

        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.)) # discharged
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.)) # discharged, LJ annihilated
        
        return alchemical_states

    @classmethod
    def defaultVacuumProtocol(cls):
        """
        Return the default protocol for ligand in solvent.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """

        alchemical_states = list()

        #                                        Rest  Elec  Attr  Repu  Cap   Tor PME
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=1.00)) # fully interacting, Changing EPA
        alchemical_states.append(AlchemicalState(0.00, 0.50, 0.50, 1.00, 1.00, 1., ligandPMEFull=0.50)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 1.00, 1., ligandPMEFull=0.00)) # EPA off, capping next
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 0.00, 1., ligandPMEFull=0.00)) # discharged, uncapped
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 1.00, 0.00, 1., ligandPMEFull=0.00)) #Attractive Off, R capped
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.50, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.20, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.10, 0.00, 1., ligandPMEFull=0.00))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.00, 0.00, 1., ligandPMEFull=0.00)) #Charge off, LJ fully decoupled
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=0.00)) # ACR to solve for A
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=1.00)) # PME only
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1., ligandPMEFull=0.50)) # Different PME to isolate the solvent-solvent vs solvent-solute interactions
        
        return alchemical_states

    @classmethod
    def defaultSolventProtocolExplicit(cls):
        """
        Return the default protocol for 'denihilating' a ligand in complex with a protein.

        RETURNS

        alchemical_states (list of AlchemicalState) - states
        
        NOTES

        The unrestrained, fully interacting system is always listed first.

        """

        alchemical_states = list()

        #                                        Rest  Elec  Attr  Repu  Cap   Tor PME
        alchemical_states.append(AlchemicalState(0.00, 1.00, 1.00, 1.00, 1.00, 1.)) # fully interacting
        alchemical_states.append(AlchemicalState(0.00, 0.00, 1.00, 1.00, 1.00, 1.)) # discharged
        #alchemical_states.append(AlchemicalState(0.00, 0.00, 0.95, 0.95, 1.00,  1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.90, 0.90, 1.00, 1.)) 
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.80, 0.80, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.70, 0.70, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.60, 0.60, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.50, 0.50, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.40, 0.40, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.30, 0.30, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.20, 0.20, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.10, 0.10, 1.00, 1.))
        alchemical_states.append(AlchemicalState(0.00, 0.00, 0.00, 0.00, 1.00, 1.)) # discharged, LJ annihilated
       
        return alchemical_states

    @classmethod
    def _alchemicallyModifyLennardJones(cls, system, nonbonded_force, alchemical_atom_indices, alchemical_state, alpha=0.50, a=1, b=1, c=1, mm=None, nb_method="SC"): #Original line, changed the alpha, a, b, and c. to refelct 1-1-48
        """
        Create a softcore version of the given reference force that controls only interactions between alchemically modified atoms and
        the rest of the system.

        ARGUMENTS

        system (simtk.openmm.System) - system to modify
        nonbonded_force (implements NonbondedForce API) - the NonbondedForce to modify
        alchemical_atom_indices (list of int) - atom indices to be alchemically modified
        alchemical_state (AlchemicalState)

        OPTIONAL ARGUMENTS

        annihilate (boolean) - if True, will annihilate alchemically-modified self-interactions; if False, will decouple
        alpha (float) - softcore parameter
        a, b, c (float) - parameters describing softcore force        
        mm (simtk.openmm implementation) - OpenMM implementation to use
        nb_method (string) - Choice of which alchemical transformation to use. Added by Levi Naden.
        
        """

        if mm is None:
            mm = openmm

        # Create CustomNonbondedForce to handle softcore interactions between alchemically-modified system and rest of system.
        # This is Levi N.'s custom variable so I only have to change it once to flip between systems instead of finding all my comments
        ####OPTION NOW SET IN createPerturbedSystem####
        #Valid Options:
        #"SC" = soft core 1-1-6
        #"SC48" = soft core 1-1-48
        #"LinA" = My linear basis function
        #"PureLin" = Purly linear
        #"Lin4" = Linear scaling with Lambda^4 instead of just Lambda
        #"Buelens" = Method by Bulens and Grubmuller, doi:10.1002/jcc.21938, Single Lambda
        if nb_method is "SC":
            #===============================================================
            # Soft Core 1-1-6
            #===============================================================
            #Single Lambda
            #energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
            #energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
            #Multi lambda
            energy_expression = "4*epsilon*(((lambda_rep^a)*x^2)-((lambda_att^a)*y));"
            energy_expression += "x = (1.0/(alpha*(1.0-lambda_rep)^b + (r/sigma)^c))^(6/c);" 
            energy_expression += "y = (1.0/(alpha*(1.0-lambda_att)^b + (r/sigma)^c))^(6/c);" 
            #Common
            energy_expression += "alpha = %f;" % alpha
            energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
        elif nb_method is "SC48":
            #===============================================================
            # Soft Core 1-1-48
            #===============================================================
            #Single Lambda
            #energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
            #energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
            #Multi lambda
            energy_expression = "4*epsilon*(((lambda_rep^a)*x^2)-((lambda_att^a)*y));"
            energy_expression += "x = (1.0/(alpha*(1.0-lambda_rep)^b + (r/sigma)^c))^(6/c);" 
            energy_expression += "y = (1.0/(alpha*(1.0-lambda_att)^b + (r/sigma)^c))^(6/c);" 
            #Common
            alpha = 0.0025
            c = 48
            energy_expression += "alpha = %f;" % alpha
            energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
        elif nb_method is "LinA":
            #===============================================================
            # Linear Basis Functions A - Naden's Original
            #===============================================================
            ###NOTE: the E has been extracted but the 4 is still embeded!###########
            energy_expression = "epsilon*((h*(Rnear+Rtrans+Rfar))+(lambda_att*(Anear+Afar)));" #Primary WCA breakdown
            energy_expression += "h = (X^lambda_rep - 1)/(X-1);" #Repulsive Term
            energy_expression += "X = %f;" % (35) #Setting the min variance vlaue
            energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
            energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
            energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
            energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
            energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
            #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
            energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
            energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
            energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
            energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
            energy_expression += "d=0;"
            energy_expression += "e=deltaC;"
            energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
            energy_expression += "idt=1.0/dt;"
            energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
            energy_expression += "deltaC = Ucap1-Ucap2;"
            energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
            energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
            energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
            energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
            energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
            energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
        elif nb_method is "LinB":
            #===============================================================
            # Linear Basis Functions B - Shirts' Original
            #===============================================================
            ###NOTE: the E has been extracted but the 4 is still embeded!###########
            energy_expression = "epsilon*((h*(Rnear+Rtrans+Rfar))+(lambda_att*(Anear+Afar)));" #Primary WCA breakdown
            energy_expression += "h = (prefactor*lambda_rep) + (lambda_rep)*(1-prefactor)*expfactor;" #Repulsive h function
            energy_expression += "expfactor = exp(-(((1-lambda_rep)/varfactor)^2));" #expfactor
            energy_expression += "prefactor = %f; varfactor = %f;" % (0.22, 0.284) #Setting the min variance values, taken from "wca_3_split.py" on Oxygen, for Ethane/Water
            energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
            energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
            energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
            energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
            energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
            #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
            energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
            energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
            energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
            energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
            energy_expression += "d=0;"
            energy_expression += "e=deltaC;"
            energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
            energy_expression += "idt=1.0/dt;"
            energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
            energy_expression += "deltaC = Ucap1-Ucap2;"
            energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
            energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
            energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
            energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
            energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
            energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
        elif nb_method is "126LinO":
            #===============================================================
            # 12-6 basis function with capping potential
            #===============================================================
            energy_expression =  "epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis);"
            ###NOTE: the epsilon has been extracted but the 4 is still embeded!###########
            energy_expression += "CappingSwitchBasis = lambda_cap*CappingBasis;"
            energy_expression += "CappingBasis = repUncap - RepCappedBasis;"
            energy_expression += "RepSwitchCappedBasis = repSwitch*RepCappedBasis;"
            energy_expression += "RepCappedBasis = Rnear + Rtrans + Rfar;"
            energy_expression += "AttSwitchBasis = attSwitch*attBasis;"
            energy_expression += "repSwitch = pa*(lambda_rep^4) + pb*(lambda_rep^3) + pc*(lambda_rep^2) + (1-pa-pb-pc)*lambda_rep;" #Repulsive Term
            energy_expression += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
            energy_expression += "attSwitch = lambda_att;"
            energy_expression += "Rnear = Ucap1*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
            energy_expression += "Rtrans = (T + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));" #cap1<r<cap2
            energy_expression += "Rfar = repUncap*(1-step(1-r/(cap2*sigma)));" #r>cap2
            #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
            energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
            energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
            energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
            energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
            energy_expression += "d=0;"
            energy_expression += "e=deltaC;"
            energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
            energy_expression += "idt=1.0/dt;"
            energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
            energy_expression += "deltaC = Ucap1-Ucap2;"
            energy_expression += "repUncap = 4 * (sigma/r)^12;" #Uncapped Repulsive basis
            energy_expression += "attBasis = -4 * (sigma/r)^6;" #Attractive basis, no capping is put on this
            energy_expression += "Ucap1 = 4*(1.0/cap1)^12;" #Potential at cap 1 = 0.8sigma
            energy_expression += "Ucap2 = 4*(1.0/cap2)^12;" #Potential at cap 2 = 0.9sigma
            energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma));"#Derivative at cap 2 = .9 sigma
            energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2));"# Second Derivative at cap 2 = .9 sigma
            energy_expression += "cap1 = %f; cap2 = %f;" % (0.82, 0.92) #Determiened by finding the average cap value near 6-7kT
        elif nb_method is "Buelens":
            #===============================================================
            # Buelens and Grubmuller Linear Transformation/Capped potential
            #===============================================================
            #Add-in constants (Vmax, rswitch) added as a global parameter below
            energy_expression = "lambda_rep * (Vnear+Vmid+Vclose);"
            energy_expression += "Vnear = LJ * (1-step(1-t));" #Long range
            energy_expression += "Vmid = T * (1-step(-t))*step(1-t);" #Transition Potential
            energy_expression += "Vclose = (Vmax + Vswitch)*step(-t);" #Maxed out potential, active only when t <= 0
            energy_expression += "T = a*t^3+b*t^2+c*t+d;"
            energy_expression += "a = LJc-b-d;" #Can work this out by hand under the pretense t goes from 0 to 1
            energy_expression += "b = -dLJc*idt+3*LJc-3*d;"
            energy_expression += "c = 0;"
            energy_expression += "d = Vmax + Vswitch;"
            energy_expression += "idt = 1/dt;"
            energy_expression += "dt = 1/(rcap-rstop);"
            energy_expression += "t = (r-rstop)/(rcap-rstop);" #Base my transition variable around 0 to 1
            energy_expression += "LJ = 4*epsilon*(x^12 - x^6);"
            energy_expression += "LJc = 4*epsilon*(y^12 - y^6);"
            energy_expression += "dLJc = 4*epsilon*(-12*sigma^12/rcap^13 + 6*sigma^6/rcap^8);"
            energy_expression += "rstop = rcap - rswitch;"
            energy_expression += "x = sigma/r;"
            energy_expression += "y = sigma/rcap;"
            energy_expression += "rcap = sigma*exp((1/6)*log(risU));" #My solution to fractional exponents being disallowed
            energy_expression += "risU = (2*epsilon/Vmax)*(sqrt(1+(Vmax/epsilon))-1);"
            #import simtk.unit as unit #OpenMM base units: kJ/mol, nm. Custom expressions do not support explicit units.
            energy_expression += "Vmax = %.3f;" %(40*units.kilocalorie_per_mole/units.kilojoule_per_mole)
            energy_expression += "rswitch = %.3f;" %(0.4 *units.angstrom/units.nanometer)
            energy_expression += "Vswitch = %.3f;" %(40*units.kilocalorie_per_mole/units.kilojoule_per_mole)
        elif nb_method is "PureLin":
            #===============================================================
            # Pure Linear Transformation
            #===============================================================
            energy_expression = "4*epsilon*(lambda_rep*(x^12) - lambda_att*(x^6));"
            energy_expression += "x = sigma/r;"
        elif nb_method is "Lin4":
            #===============================================================
            # Lambda^4 Linear Transformation
            #===============================================================
            energy_expression = "4*epsilon*((lambda_rep^4)*(x^12) - (lambda_att^4)*(x^6));"
            energy_expression += "x = sigma/r;"
        elif nb_method is "LinOptimized":
            #===============================================================
            # Linear Basis Functions Optimized - General polynomial - Found by Optimizing from LinA, with capping
            #===============================================================
            energy_expression =  "epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis);"
            ###NOTE: the epsilon has been extracted but the 4 is still embeded!###########
            energy_expression += "CappingSwitchBasis = lambda_cap*CappingBasis;"
            energy_expression += "CappingBasis = repUncap - RepCappedBasis;"
            energy_expression += "RepSwitchCappedBasis = repSwitch*RepCappedBasis;"
            energy_expression += "RepCappedBasis = Rnear + Rtrans + Rfar;"
            energy_expression += "AttSwitchBasis = attSwitch*attBasis;"
            energy_expression += "repSwitch = pa*(lambda_rep^4) + pb*(lambda_rep^3) + pc*(lambda_rep^2) + (1-pa-pb-pc)*lambda_rep;" #Repulsive Term
            energy_expression += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
            energy_expression += "attSwitch = lambda_att;"
            energy_expression += "attBasis = Anear+Afar;"
            energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
            energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
            energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
            energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
            energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
            #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
            energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
            energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
            energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
            energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
            energy_expression += "d=0;"
            energy_expression += "e=deltaC;"
            energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
            energy_expression += "idt=1.0/dt;"
            energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
            energy_expression += "deltaC = Ucap1-Ucap2;"
            energy_expression += "repUncap = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma));" #Uncapped Repulsive Basis
            energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
            energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
            energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
            energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
            energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
            energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
        #===============================================================
        # Common Terms
        #===============================================================
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma

        if alchemical_state.annihilateLennardJones:
            # Annihilate interactions within alchemically-modified subsystem and between subsystem and environment.
            energy_expression += "lambda_rep = repulsive_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);" 
            energy_expression += "lambda_att = attractive_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);" 
            energy_expression += "lambda_cap = capping_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);" 
        else:
            # Decouple interactions between alchemically-modified subsystem and environment only.
            energy_expression += "lambda_rep = repulsive_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1)) + alchemical1*alchemical2;"
            energy_expression += "lambda_att = attractive_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1)) + alchemical1*alchemical2;"
            energy_expression += "lambda_cap = capping_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1)) + alchemical1*alchemical2;"
        custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)            
        custom_nonbonded_force.addGlobalParameter("repulsive_lambda", alchemical_state.ligandRepulsion);
        custom_nonbonded_force.addGlobalParameter("attractive_lambda", alchemical_state.ligandAttraction);
        custom_nonbonded_force.addGlobalParameter("capping_lambda", alchemical_state.ligandCapToFull);
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
        custom_nonbonded_force.addPerParticleParameter("alchemical") # alchemical flag: 1 if this particle is alchemically modified, 0 otherwise
        #if nb_method is "Buelens":
            #===============================================================
            # Buelens and Grubmuller Add-ins
            #===============================================================
            #import simtk.unit as unit #OpenMM base units: kJ/mol, nm. Custom expressions do not support explicit units.
            #Custom expressions do not support explicit units (SimtK site, OpenMM march 2012 workshop, custom expression video)
            #Default units shown in OpenMMUserGuide.pdf
            #custom_nonbonded_force.addGlobalParameter("Vmax", 40*units.kilocalorie_per_mole/units.kilojoule_per_mole)
            #custom_nonbonded_force.addGlobalParameter("rswitch", 0.4 * units.angstrom/units.nanometer)
            #custom_nonbonded_force.addGlobalParameter("Vswitch", 40*units.kilocalories_per_mole/units.kilojoule_per_mole) #40 for LJ, 20 for Electro.

        system.addForce(custom_nonbonded_force)

        # Create CustomBondedForce to handle softcore exceptions if alchemically annihilating ligand.

        if alchemical_state.annihilateLennardJones:
            if nb_method is "SC":
                #===============================================================
                # Soft Core 1-1-6
                #===============================================================
                #Single Lambda
                #energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
                #energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
                #Multi Lambda
                energy_expression = "4*epsilon*(((lambda_rep^a)*x^2)-((lambda_att^a)*y));"
                energy_expression += "x = (1.0/(alpha*(1.0-lambda_rep)^b + (r/sigma)^c))^(6/c);" 
                energy_expression += "y = (1.0/(alpha*(1.0-lambda_att)^b + (r/sigma)^c))^(6/c);" 
                #Common
                energy_expression += "alpha = %f;" % alpha
                energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
            elif nb_method is "SC48":
                #===============================================================
                # Soft Core 1-1-48
                #===============================================================
                #Single Lambda
                #energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
                #energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);" 
                #Multi Lambda
                energy_expression = "4*epsilon*(((lambda_rep^a)*x^2)-((lambda_att^a)*y));"
                energy_expression += "x = (1.0/(alpha*(1.0-lambda_rep)^b + (r/sigma)^c))^(6/c);" 
                energy_expression += "y = (1.0/(alpha*(1.0-lambda_att)^b + (r/sigma)^c))^(6/c);" 
                #Common
                alpha = 0.0025
                c = 48
                energy_expression += "alpha = %f;" % alpha
                energy_expression += "a = %f; b = %f; c = %f;" % (a,b,c)    
            elif nb_method is "LinA":
                #===============================================================
                # Linear Basis Functions A
                #===============================================================
                ###NOTE: the E has been extracted but the 4 is still embeded!###########
                energy_expression = "epsilon*((h*(Rnear+Rtrans+Rfar))+(lambda_att*(Anear+Afar)));" #Primary WCA breakdown
                energy_expression += "h = (X^lambda_rep - 1)/(X-1);" #Repulsive Term
                energy_expression += "X = %f;" % (35) #Setting the min variance vlaue
                energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
                energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
                energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
                energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
                energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
                #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
                energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
                energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
                energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
                energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
                energy_expression += "d=0;"
                energy_expression += "e=deltaC;"
                energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
                energy_expression += "idt=1/dt;"
                energy_expression += "dt = 1/(cap2*sigma-cap1*sigma);"
                energy_expression += "deltaC = Ucap1-Ucap2;"
                energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
                energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
                energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
                energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
                energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
                energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
            elif nb_method is "LinB":
                #===============================================================
                # Linear Basis Functions B - Shirts' Original
                #===============================================================
                ###NOTE: the E has been extracted but the 4 is still embeded!###########
                energy_expression = "epsilon*((h*(Rnear+Rtrans+Rfar))+(lambda_att*(Anear+Afar)));" #Primary WCA breakdown
                energy_expression += "h = (prefactor*lambda_rep) + (lambda_rep)*(1-prefactor)*expfactor;" #Repulsive h function
                energy_expression += "expfactor = exp(-(((1-lambda_rep)/varfactor)^2));" #expfactor
                energy_expression += "prefactor = %f; varfactor = %f;" % (0.22, 0.284) #Setting the min variance values, taken from "wca_3_split.py" on Oxygen, for Ethane/Water
                energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
                energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
                energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
                energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
                energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
                #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
                energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
                energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
                energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
                energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
                energy_expression += "d=0;"
                energy_expression += "e=deltaC;"
                energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
                energy_expression += "idt=1.0/dt;"
                energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
                energy_expression += "deltaC = Ucap1-Ucap2;"
                energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
                energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
                energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
                energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
                energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
                energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
            elif nb_method is "126LinO":
                #===============================================================
                # 12-6 basis function with capping potential
                #===============================================================
                energy_expression =  "epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis);"
                ###NOTE: the epsilon has been extracted but the 4 is still embeded!###########
                energy_expression += "CappingSwitchBasis = lambda_cap*CappingBasis;"
                energy_expression += "CappingBasis = repUncap - RepCappedBasis;"
                energy_expression += "RepSwitchCappedBasis = repSwitch*RepCappedBasis;"
                energy_expression += "RepCappedBasis = Rnear + Rtrans + Rfar;"
                energy_expression += "AttSwitchBasis = attSwitch*(attBasis);"
                energy_expression += "repSwitch = pa*(lambda_rep^4) + pb*(lambda_rep^3) + pc*(lambda_rep^2) + (1-pa-pb-pc)*lambda_rep;" #Repulsive Term
                energy_expression += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
                energy_expression += "attSwitch = lambda_att;"
                energy_expression += "Rnear = Ucap1*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
                energy_expression += "Rtrans = (T + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));" #cap1<r<cap2
                energy_expression += "Rfar = repUncap*(1-step(1-r/(cap2*sigma)));" #r>cap2
                #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
                energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
                energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
                energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
                energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
                energy_expression += "d=0;"
                energy_expression += "e=deltaC;"
                energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
                energy_expression += "idt=1.0/dt;"
                energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
                energy_expression += "deltaC = Ucap1-Ucap2;"
                energy_expression += "repUncap = 4 * (sigma/r)^12;" #Uncapped Repulsive basis
                energy_expression += "attBasis = -4 * (sigma/r)^6;" #Attractive basis, no capping is put on this
                energy_expression += "Ucap1 = 4*(1.0/cap1)^12;" #Potential at cap 1 = 0.8sigma
                energy_expression += "Ucap2 = 4*(1.0/cap2)^12;" #Potential at cap 2 = 0.9sigma
                energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma));"#Derivative at cap 2 = .9 sigma
                energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2));"# Second Derivative at cap 2 = .9 sigma
                energy_expression += "cap1 = %f; cap2 = %f;" % (0.82, 0.92) #Determiened by finding the average cap value near 6-7kT
            elif nb_method is "Buelens":
                #===============================================================
                # Buelens and Grubmuller Linear Transformation/Capped potential
                #===============================================================
                #Add-in constants (Vmax, rswitch) added as a global parameter below
                energy_expression = "lambda_rep * (Vnear+Vmid+Vclose);"
                energy_expression += "Vnear = LJ * (1-step(1-t));" #Long range
                energy_expression += "Vmid = T * (1-step(-t))*step(1-t);" #Transition Potential
                energy_expression += "Vclose = (Vmax + Vswitch)*step(-t);" #Maxed out potential, active only when t <= 0
                energy_expression += "T = a*t^3+b*t^2+c*t+d;"
                energy_expression += "a = LJc-b-d;" #Can work this out by hand under the pretense t goes from 0 to 1
                energy_expression += "b = -dLJc*idt+3*LJc-3*d;"
                energy_expression += "c = 0;"
                energy_expression += "d = Vmax + Vswitch;"
                energy_expression += "idt = 1/dt;"
                energy_expression += "dt = 1/(rcap-rstop);"
                energy_expression += "t = (r-rstop)/(rcap-rstop);" #Base my transition variable around 0 to 1
                energy_expression += "LJ = 4*epsilon*(x^12 - x^6);"
                energy_expression += "LJc = 4*epsilon*(y^12 - y^6);"
                energy_expression += "dLJc = 4*epsilon*(-12*sigma^12/rcap^13 + 6*sigma^6/rcap^8);"
                energy_expression += "rstop = rcap - rswitch;"
                energy_expression += "x = sigma/r;"
                energy_expression += "y = sigma/rcap;"
                energy_expression += "rcap = sigma*exp((1/6)*log(risU));" #My solution to fractional exponents being disallowed
                energy_expression += "risU = (2*epsilon/Vmax)*(sqrt(1+(Vmax/epsilon))-1);"
                energy_expression += "Vmax = %.3f;" %(40*units.kilocalorie_per_mole/units.kilojoule_per_mole)
                energy_expression += "rswitch = %.3f;" %(0.4 *units.angstrom/units.nanometer)
                energy_expression += "Vswitch = %.3f;" %(40*units.kilocalorie_per_mole/units.kilojoule_per_mole)
            elif nb_method is "PureLin":
                #===============================================================
                # Pure Linear Transformation
                #===============================================================
                energy_expression = "4*epsilon*(lambda_rep*(x^12) - lambda_att*(x^6));"
                energy_expression += "x = sigma/r;"
            elif nb_method is "Lin4":
                #===============================================================
                # Lambda^4 Linear Transformation
                #===============================================================
                energy_expression = "4*epsilon*((lambda_rep^4)*(x^12) - (lambda_att^4)*(x^6));"
                energy_expression += "x = sigma/r;"
            elif nb_method is "LinOptimized":
                #===============================================================
                # Linear Basis Functions Optimized - General polynomial - Found by Optimizing from LinA
                #===============================================================
                ###NOTE: the E has been extracted but the 4 is still embeded!###########
                energy_expression = "epsilon*((h*(Rnear+Rtrans+Rfar))+(lambda_att*(Anear+Afar)));" #Primary WCA breakdown
                energy_expression += "h = pa*(lambda_rep^4) + pb*(lambda_rep^3) + pc*(lambda_rep^2) + (1-pa-pb-pc)*lambda_rep;" #Repulsive Term
                energy_expression += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
                energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
                energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
                energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
                energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
                energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
                #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
                energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
                energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
                energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
                energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
                energy_expression += "d=0;"
                energy_expression += "e=deltaC;"
                energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
                energy_expression += "idt=1.0/dt;"
                energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
                energy_expression += "deltaC = Ucap1-Ucap2;"
                energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
                energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
                energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
                energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
                energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
                energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
            #===============================================================
            # Common Terms
            #===============================================================
            custom_bond_force = mm.CustomBondForce(energy_expression)            
            custom_bond_force.addGlobalParameter("repulsive_lambda", alchemical_state.ligandRepulsion);
            custom_bond_force.addGlobalParameter("attractive_lambda", alchemical_state.ligandAttraction);
            custom_nonbonded_force.addGlobalParameter("capping_lambda", alchemical_state.ligandCapToFull);
            custom_bond_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
            custom_bond_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
            #if nb_method is "Buelens":
                #===============================================================
                # Buelens and Grubmuller Add-ins
                #===============================================================
                #Units import handled from above, do not want conflicting memory locations
                #Custom expressions do not support explicit units (SimtK site, OpenMM march 2012 workshop, custom expression video)
                #Default units shown in OpenMMUserGuide.pdf
                #custom_nonbonded_force.addGlobalParameter("Vmax", 40*units.kilocalorie_per_mole/units.kilojoule_per_mole)
                #custom_nonbonded_force.addGlobalParameter("rswitch", 0.4 * units.angstrom/units.nanometer)
                #custom_nonbonded_force.addGlobalParameter("Vswitch", 40*units.kilocalories_per_mole/units.kilojoule_per_mole) #40 for LJ, 20 for Electro.
            #system.addForce(custom_bond_force)

        # Copy Lennard-Jones particle parameters.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Add corresponding particle to softcore interactions.
            if particle_index in alchemical_atom_indices:
                # Turn off Lennard-Jones contribution from alchemically-modified particles.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon*0.0) 
                # Add contribution back to custom force.
                custom_nonbonded_force.addParticle([sigma, epsilon, 1])
            else:
                custom_nonbonded_force.addParticle([sigma, epsilon, 0])

        # Create an exclusion for each exception in the reference NonbondedForce, assuming that NonbondedForce will handle them.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            custom_nonbonded_force.addExclusion(iatom, jatom)

            # If annihilating Lennard-Jones, these interactions will be handled by the softcore force.
            if alchemical_state.annihilateLennardJones and (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices):
                # Remove Lennard-Jones exception.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon * 0.0)
                # Add special CustomBondForce term to handle alchemically-modified Lennard-Jones exception.
                custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [mm.NonbondedForce.Ewald, mm.NonbondedForce.PME]:
            # Convert Ewald and PME to CutoffPeriodic.
            custom_nonbonded_force.setNonbondedMethod( mm.CustomNonbondedForce.CutoffPeriodic )
        else:
            custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )
        custom_nonbonded_force.setCutoffDistance( nonbonded_force.getCutoffDistance() )

        return 

    @classmethod
    def _alchemicallyModifyElectrostatics(cls, system, nonbonded_force, alchemical_atom_indices, alchemical_state, mm=None, nbe_basis="1/r"):

        if mm is None:
            mm = openmm
        #Adjust ligandPMEFull if its not set
        if alchemical_state.ligandPMEFull is None:
            alchemical_state.ligandPMEFull = alchemical_state.ligandEletrostatics
        #Remove the direct space calculations from the PME term
        from math import pi
        #Rebuild cofactor. NOTE: Manual is wrong and missing this term, had to look at openmm code to find out
        #electron_charge = 1.60218E-19 * units.coulombs
        #dielec_vac = 8.85419E-12 * (units.coulombs)**2 / (units.joules * units.meter * units.AVOGADRO_CONSTANT_NA) #Source: Prausnitz et al, Molecular Thermodynamics of Fluid-Phase Equilibria, 3rd ed.
        #enot = dielec_vac / electron_charge**2
        #dimles_enot = enot * units.kilojoules_per_mole * units.nanometer
        #ONE_4PI_EPS0 = 1.0/(4*pi*dimles_enot) #Returns 138.936030, OpenMM uses 138.935456
        ONE_4PI_EPS0 = 138.935456 #From OpenMM's OpenCL kernel
        
        err_tol = nonbonded_force.getEwaldErrorTolerance()
        rcutoff = nonbonded_force.getCutoffDistance() / units.nanometer #getCutoffDistance Returns a unit object, convert to OpenMM default (nm)
        alpha = numpy.sqrt(-numpy.log(2*err_tol))/rcutoff #NOTE: OpenMM manual is wrong here
        energy_expression = "alchEdirect-(Edirect * eitheralch);" #Correction for the alchemical1=0 alchemical2=0 case)
        energy_expression += "Edirect = (%f * (switchPME*alchemical1 + 1 -alchemical1) * charge1 * (switchPME*alchemical2 + 1 -alchemical2) * charge2 * erfc(%f * r)/r);" % (ONE_4PI_EPS0, alpha) #The extra bits correct for alchemical1 and alchemical2 being on
        energy_expression += "alchEdirect = switchE * %f * charge1 * charge2 * erfc(%f * r)/r;" % (ONE_4PI_EPS0, alpha)
        energy_expression += "switchPME = lambda_PME;"
        energy_expression += "switchE = lambda_E;"
        energy_expression += "eitheralch = (alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);"
        if alchemical_state.annihilateElectrostatics:
            # Annihilate interactions within alchemically-modified subsystem and between subsystem and environment.
            energy_expression += "lambda_E = electrostatic_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);" 
            energy_expression += "lambda_PME = particlemeshewald_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1) + alchemical1*alchemical2);" 
        else:
            # Decouple interactions between alchemically-modified subsystem and environment only.
            energy_expression += "lambda_E = electrostatic_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1)) + alchemical1*alchemical2;"
            energy_expression += "lambda_PME = particlemeshewald_lambda*(alchemical1*(1-alchemical2) + alchemical2*(1-alchemical1)) + alchemical1*alchemical2;"
        custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addGlobalParameter("electrostatic_lambda", alchemical_state.ligandElectrostatics)
        custom_nonbonded_force.addGlobalParameter("particlemeshewald_lambda", alchemical_state.ligandPMEFull)
        custom_nonbonded_force.addPerParticleParameter("charge")
        custom_nonbonded_force.addPerParticleParameter("alchemical")
        system.addForce(custom_nonbonded_force)
        
        if alchemical_state.annihilateElectrostatics:
            energy_expression = "-Edirect + alchEdirect;"
            energy_expression += "Edirect = switchPME^2 * %f * chargeprod * erfc(%f * r)/r;" % (ONE_4PI_EPS0, alpha)
            energy_expression += "alchEdirect = switchE * %f * chargeprod * erfc(%f * r)/r;" % (ONE_4PI_EPS0, alpha)
            energy_expression += "switchPME = lambda_PME;"
            energy_expression += "switchE = lambda_E;"
            custom_bond_force = mm.CustomBondForce(energy_expression)
            custom_bond_force.addGlobalParameter("lambda_E", alchemical_state.ligandElectrostatics)
            custom_bond_force.addGlobalParameter("lambda_PME", alchemical_state.ligandPMEFull)
            custom_bond_force.addPerBondParameter("chargeprod")
            system.addForce(custom_bond_force)

        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Alchemically modify charges.
            if particle_index in alchemical_atom_indices:
                custom_nonbonded_force.addParticle([charge, 1]) #List is in sequence of PerParticleParameters
                charge *= alchemical_state.ligandPMEFull #CHARGE MODIFICATION MUST COME AFTER CUSTOM_NONBONDED ASSIGNMENT! Otherwise you dobule up on caunting for the ligandPME term and removing the direct space calculation is wrong
            else: # Nonalchemical atom
                custom_nonbonded_force.addParticle([charge, 0])
            # Set modified particle parameters.
            nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            custom_nonbonded_force.addExclusion(iatom, jatom)
            # Alchemically modify chargeprod.
            if (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices) and alchemical_state.annihilateElectrostatics:
                custom_bond_force.addBond(iatom, jatom, [chargeprod])
                chargeprod *= alchemical_state.ligandPMEFull**2
            # Set modified exception parameters.
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon) #Copy parameters in

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [mm.NonbondedForce.Ewald, mm.NonbondedForce.PME]:
            # Convert Ewald and PME to CutoffPeriodic.
            custom_nonbonded_force.setNonbondedMethod( mm.CustomNonbondedForce.CutoffPeriodic )
        else:
            custom_nonbonded_force.setNonbondedMethod( nonbonded_force.getNonbondedMethod() )
        custom_nonbonded_force.setCutoffDistance( nonbonded_force.getCutoffDistance() )


        return 

    @classmethod
    def _createCustomSoftcoreGBOBC(cls, reference_force, particle_lambdas, sasa_model='ACE', mm=None):
        """
        Create a softcore OBC GB force using CustomGBForce.

        ARGUMENTS

        reference_force (simtk.openmm.GBSAOBCForce) - reference force to use for template
        particle_lambdas (list or numpy array) - particle_lambdas[i] is the alchemical lambda for particle i, with 1.0 being fully interacting and 0.0 noninteracting

        OPTIONAL ARGUMENTS

        sasa_model (string) - solvent accessible surface area model (default: 'ACE')
        mm (simtk.openmm API) - (default: simtk.openmm)

        RETURNS

        custom (openmm.CustomGBForce) - custom GB force object
        
        """

        if mm is None:
            mm = openmm    
            
        custom = mm.CustomGBForce()

        # Add per-particle parameters.
        custom.addPerParticleParameter("q");
        custom.addPerParticleParameter("radius");
        custom.addPerParticleParameter("scale");
        custom.addPerParticleParameter("lambda");
        
        # Set nonbonded method.
        custom.setNonbondedMethod(reference_force.getNonbondedMethod())

        # Add global parameters.
        custom.addGlobalParameter("solventDielectric", reference_force.getSolventDielectric())
        custom.addGlobalParameter("soluteDielectric", reference_force.getSoluteDielectric())
        custom.addGlobalParameter("offset", 0.009)

        custom.addComputedValue("I",  "lambda2*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
                                "U=r+sr2;"
                                "L=max(or1, D);"
                                "D=abs(r-sr2);"
                                "sr2 = scale2*or2;"
                                "or1 = radius1-offset; or2 = radius2-offset", mm.CustomGBForce.ParticlePairNoExclusions)

        custom.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", mm.CustomGBForce.SingleParticle)

        custom.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B", mm.CustomGBForce.SingleParticle)
        if sasa_model == 'ACE':
            custom.addEnergyTerm("lambda*28.3919551*(radius+0.14)^2*(radius/B)^6", mm.CustomGBForce.SingleParticle)

        custom.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                             "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", mm.CustomGBForce.ParticlePairNoExclusions);

        # Add particle parameters.
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
            lambda_factor = float(particle_lambdas[particle_index])
            # Set particle parameters.
            # Note that charges must be scaled by lambda_factor in this representation.
            parameters = [charge * lambda_factor, radius, scaling_factor, lambda_factor]
            custom.addParticle(parameters)

        return custom

    def createPerturbedSystem(self, alchemical_state, mm=None, verbose=False):
        """
        Create a perturbed copy of the system given the specified alchemical state.

        ARGUMENTS

        alchemical_state (AlchemicalState) - the alchemical state to create from the reference system

        TODO

        * Start from a deep copy of the system, rather than building copy through Python interface.
        * isinstance(mm.NonbondedForce) and related expressions won't work if reference system was created with a different OpenMM implemnetation.

        EXAMPLES

        Create alchemical intermediates for 'denihilating' one water in a water box.
        
        >>> # Create a reference system.
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.WaterBox()
        >>> # Create a factory to produce alchemical intermediates.
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.00, 1.00, 1.00, 1.)
        >>> # Create the perturbed system.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
        >>> # Compare energies.
        >>> import simtk.openmm as openmm
        >>> import simtk.unit as units
        >>> timestep = 1.0 * units.femtosecond
        >>> reference_integrator = openmm.VerletIntegrator(timestep)
        >>> reference_context = openmm.Context(reference_system, reference_integrator)
        >>> reference_state = reference_context.getState(getEnergy=True)
        >>> reference_potential = reference_state.getPotentialEnergy()
        >>> alchemical_integrator = openmm.VerletIntegrator(timestep)
        >>> alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
        >>> alchemical_state = alchemical_context.getState(getEnergy=True)
        >>> alchemical_potential = alchemical_state.getPotentialEnergy()
        >>> delta = alchemical_potential - reference_potential 
        >>> print delta
        0.0 kJ/mol
        
        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
        >>> # Compute reference potential.
        >>> timestep = 1.0 * units.femtosecond
        >>> reference_integrator = openmm.VerletIntegrator(timestep)
        >>> reference_context = openmm.Context(reference_system, reference_integrator)
        >>> reference_context.setPositions(coordinates)
        >>> reference_state = reference_context.getState(getEnergy=True)
        >>> reference_potential = reference_state.getPotentialEnergy()
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
        >>> # Create an alchemically-perturbed state corresponding to fully-interacting.
        >>> alchemical_state = AlchemicalState(0.00, 1.00, 1.00, 1.)
        >>> # Create the perturbed systems using this protocol.
        >>> alchemical_system = factory.createPerturbedSystem(alchemical_state)
        >>> # Compare energies.        
        >>> alchemical_integrator = openmm.VerletIntegrator(timestep)
        >>> alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
        >>> alchemical_context.setPositions(coordinates)
        >>> alchemical_state = alchemical_context.getState(getEnergy=True)
        >>> alchemical_potential = alchemical_state.getPotentialEnergy()
        >>> delta = alchemical_potential - reference_potential 
        >>> print delta
        0.0 kJ/mol

        NOTES

        If lambda = 1.0 is specified for some force terms, they will not be replaced with modified forms.        

        """

        # Record timing statistics.
        initial_time = time.time()
        if verbose: print "Creating alchemically modified intermediate..."

        reference_system = self.reference_system

        # Create new deep copy reference system to modify.
        system = openmm.System()
        
        # Set periodic box vectors.
        [a,b,c] = reference_system.getDefaultPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(a,b,c)
        
        # Add atoms.
        for atom_index in range(reference_system.getNumParticles()):
            mass = reference_system.getParticleMass(atom_index)
            system.addParticle(mass)

        # Add constraints
        for constraint_index in range(reference_system.getNumConstraints()):
            [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
            system.addConstraint(iatom, jatom, r0)    

        #Set the nonbonded method
        # This is Levi N.'s custom variable so I only have to change it once to flip between systems instead of finding all my comments
        #!!!
        self.nb_method = "LinOptimized"
        #Valid Options:
        #"SC" = soft core
        #"LinA" = My linear basis function
        #"PureLin" = Purly linear
        #"Lin4" = Linear scaling with Lambda^4 instead of just Lambda
        #"Buelens" = Method by Bulens and Grubmuller, doi:10.1002/jcc.21938, Single Lambda

        # Modify forces as appropriate, copying other forces without modification.
        nforces = reference_system.getNumForces()
        for force_index in range(nforces):
            reference_force = reference_system.getForce(force_index)

            if isinstance(reference_force, openmm.PeriodicTorsionForce):
                # PeriodicTorsionForce
                force = openmm.PeriodicTorsionForce()
                for torsion_index in range(reference_force.getNumTorsions()):
                    # Retrieve parmaeters.
                    [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                    # Scale torsion barrier of alchemically-modified system.
                    if set([particle1,particle2,particle3,particle4]).issubset(self.ligand_atomset):
                        k *= alchemical_state.ligandTorsions
                    force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
                system.addForce(force)

            elif isinstance(reference_force, openmm.NonbondedForce):

                # Copy NonbondedForce.
                force = copy.deepcopy(reference_force)
                system.addForce(force)

                # Modify electrostatics.
                if alchemical_state.ligandElectrostatics != 1.0 or (alchemical_state.ligandPMEFull is not None and alchemical_state.ligandPMEFull != 1.0):
                    #Have to make a custom force, so we need to do something similar similar to the Lennard-Jones
                    self._alchemicallyModifyElectrostatics(system, force, self.ligand_atoms, alchemical_state) #Templated off the Lennard-Jones
                    #for particle_index in range(force.getNumParticles()):
                    #    # Retrieve parameters.
                    #    [charge, sigma, epsilon] = force.getParticleParameters(particle_index)
                    #    # Alchemically modify charges.
                    #    if particle_index in self.ligand_atomset:
                    #        charge *= alchemical_state.ligandPMEFull
                    #    # Set modified particle parameters.
                    #    force.setParticleParameters(particle_index, charge, sigma, epsilon)
                    #for exception_index in range(force.getNumExceptions()):
                    #    # Retrieve parameters.
                    #    [iatom, jatom, chargeprod, sigma, epsilon] = force.getExceptionParameters(exception_index)
                    #    # Alchemically modify chargeprod.
                    #    if (iatom in self.ligand_atomset) and (jatom in self.ligand_atomset):
                    #        if alchemical_state.annihilateElectrostatics:
                    #            chargeprod *= alchemical_state.ligandPMEFull**2
                    #    # Set modified exception parameters.
                    #    force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)

                # Modify Lennard-Jones if required.
                #if alchemical_state.ligandRepulsion != 1.0:
                #Modify them if the electrostatics are off
                #if alchemical_state.ligandRepulsion != 1.0 or alchemical_state.ligandAttraction != 1.0:
                #!!!
                #if alchemical_state.ligandElectrostatics == 0.0 and (alchemical_state.ligandPMEFull == 0.0 or alchemical_state.ligandPMEFull is None):
                if alchemical_state.ligandCapToFull != 1.0 or alchemical_state.ligandAttraction != 1.0 or alchemical_state.ligandRepulsion != 1.0:
                    #Warn if electrostatics are on
                    if alchemical_state.ligandElectrostatics != 0.0 or (alchemical_state.ligandPMEFull != 0.0 and alchemical_state.ligandPMEFull is not None):
                        print "Warrning! Lennard Jones changing with electrostatic interactions!"
                    # Create softcore Lennard-Jones interactions by modifying NonbondedForce and adding CustomNonbondedForce.                
                    self._alchemicallyModifyLennardJones(system, force, self.ligand_atoms, alchemical_state, nb_method=self.nb_method)                

            elif isinstance(reference_force, openmm.GBSAOBCForce) and (alchemical_state.ligandElectrostatics != 1.0):

                # Create a CustomNonbondedForce to implement softcore interactions.
                particle_lambdas = numpy.ones([system.getNumParticles()], numpy.float32)
                particle_lambdas[self.ligand_atoms] = alchemical_state.ligandElectrostatics
                custom_force = AbsoluteAlchemicalFactory._createCustomSoftcoreGBOBC(reference_force, particle_lambdas)
                system.addForce(custom_force)
                    
            elif isinstance(reference_force, openmm.CustomExternalForce):

                force = openmm.CustomExternalForce( reference_force.getEnergyFunction() )
                for parameter_index in range(reference_force.getNumGlobalParameters()):
                    name = reference_force.getGlobalParameterName(parameter_index)
                    default_value = reference_force.getGlobalParameterDefaultValue(parameter_index)
                    force.addGlobalParameter(name, default_value)
                for parameter_index in range(reference_force.getNumPerParticleParameters()):
                    name = reference_force.getPerParticleParameterName(parameter_index)
                    force.addPerParticleParameter(name)
                for index in range(reference_force.getNumParticles()):
                    [particle_index, parameters] = reference_force.getParticleParameters(index)
                    force.addParticle(particle_index, parameters)
                system.addForce(force)

            elif isinstance(reference_force, openmm.CustomBondForce):                                

                force = openmm.CustomBondForce( reference_force.getEnergyFunction() )
                for parameter_index in range(reference_force.getNumGlobalParameters()):
                    name = reference_force.getGlobalParameterName(parameter_index)
                    default_value = reference_force.getGlobalParameterDefaultValue(parameter_index)
                    force.addGlobalParameter(name, default_value)
                for parameter_index in range(reference_force.getNumPerBondParameters()):
                    name = reference_force.getPerBondParameterName(parameter_index)
                    force.addPerBondParameter(name)
                for index in range(reference_force.getNumBonds()):
                    [particle1, particle2, parameters] = reference_force.getBondParameters(index)
                    force.addBond(particle1, particle2, parameters)
                system.addForce(force)                    

            else:                

                # Copy force without modification.
                force = copy.deepcopy(reference_force)
                system.addForce(force)

        # Record timing statistics.
        final_time = time.time()
        elapsed_time = final_time - initial_time
        if verbose: print "Elapsed time %.3f s." % (elapsed_time)
        
        return system

    def createPerturbedSystems(self, alchemical_states, verbose=False):
        """
        Create a list of perturbed copies of the system given a specified set of alchemical states.

        ARGUMENTS

        states (list of AlchemicalState) - list of alchemical states to generate
        
        RETURNS
        
        systems (list of simtk.openmm.System) - list of alchemically-modified System objects

        EXAMPLES

        Create alchemical intermediates for 'denihilating' p-xylene in T4 lysozyme L99A in GBSA.
        
        >>> # Create a reference system.
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.LysozymeImplicit()
        >>> # Create a factory to produce alchemical intermediates.
        >>> receptor_atoms = range(0,2603) # T4 lysozyme L99A
        >>> ligand_atoms = range(2603,2621) # p-xylene
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
        >>> # Get the default protocol for 'denihilating' in complex in explicit solvent.
        >>> protocol = factory.defaultComplexProtocolImplicit()
        >>> # Create the perturbed systems using this protocol.
        >>> systems = factory.createPerturbedSystems(protocol)        
        
        """

        systems = list()
        for (state_index, alchemical_state) in enumerate(alchemical_states):            
            if verbose: print "Creating alchemical system %d / %d..." % (state_index, len(alchemical_states))
            system = self.createPerturbedSystem(alchemical_state, verbose=verbose)
            systems.append(system)

        return systems
    
    def _is_restraint(self, valence_atoms):
        """
        Determine whether specified valence term connects the ligand with its environment.

        ARGUMENTS
        
        valence_atoms (list of int) - atom indices involved in valence term (bond, angle or torsion)

        RETURNS

        True if the set of atoms includes at least one ligand atom and at least one non-ligand atom; False otherwise

        EXAMPLES
        
        Various tests.
        
        >>> # Create a reference system.
        >>> from simtk.pyopenmm.extras import testsystems
        >>> [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
        >>> # Create a factory.
        >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
        >>> factory._is_restraint([0,1,2])
        False
        >>> factory._is_restraint([1,2,3])
        True
        >>> factory._is_restraint([3,4])
        False
        >>> factory._is_restraint([2,3,4,5])
        True

        """

        valence_atomset = Set(valence_atoms)
        intersection = Set.intersection(valence_atomset, self.ligand_atomset)
        if (len(intersection) >= 1) and (len(intersection) < len(valence_atomset)):
            return True

        return False        

#=============================================================================================
# MAIN AND UNIT TESTS
#=============================================================================================

def testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms, platform_name='OpenCL'):
    """
    Compare energies of reference system and fully-interacting alchemically modified system.

    ARGUMENTS
    
    reference_system (simtk.openmm.System) - the reference System object to compare with
    coordinates - the coordinates to assess energetics for
    receptor_atoms (list of int) - the list of receptor atoms 
    ligand_atoms (list of int) - the list of ligand atoms to alchemically modify

    """

    import simtk.unit as units
    import simtk.openmm as openmm
    import time

    # Create a factory to produce alchemical intermediates.
    print "Creating alchemical factory..."
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    print "AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time

    # Create an alchemically-perturbed state corresponding to nearly fully-interacting.
    # NOTE: We use a lambda slightly smaller than 1.0 because the AlchemicalFactory does not use Custom*Force softcore versions if lambda = 1.0 identically.
    lambda_value = 1.0 - 1.0e-6
    alchemical_state = AlchemicalState(0.00, lambda_value, lambda_value, lambda_value)

    #platform_name = 'Reference' # DEBUG
    platform = openmm.Platform.getPlatformByName(platform_name)
    
    # Create the perturbed system.
    print "Creating alchemically-modified state..."
    initial_time = time.time()
    alchemical_system = factory.createPerturbedSystem(alchemical_state)    
    final_time = time.time()
    elapsed_time = final_time - initial_time
    # Compare energies.
    timestep = 1.0 * units.femtosecond
    print "Computing reference energies..."
    reference_integrator = openmm.VerletIntegrator(timestep)
    reference_context = openmm.Context(reference_system, reference_integrator, platform)
    reference_context.setPositions(coordinates)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()    
    print "Computing alchemical energies..."
    alchemical_integrator = openmm.VerletIntegrator(timestep)
    alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, platform)
    alchemical_context.setPositions(coordinates)
    alchemical_state = alchemical_context.getState(getEnergy=True)
    alchemical_potential = alchemical_state.getPotentialEnergy()
    delta = alchemical_potential - reference_potential 
    print "reference system       : %24.8f kcal/mol" % (reference_potential / units.kilocalories_per_mole)
    print "alchemically modified  : %24.8f kcal/mol" % (alchemical_potential / units.kilocalories_per_mole)
    print "ERROR                  : %24.8f kcal/mol" % ((alchemical_potential - reference_potential) / units.kilocalories_per_mole)
    print "elapsed alchemical time  %.3f s" % elapsed_time

    return delta

def test_overlap():
    """
    BUGS TO REPORT:
    * Even if epsilon = 0, energy of two overlapping atoms is 'nan'.
    * Periodicity in 'nan' if dr = 0.1 even in nonperiodic system
    """

    # Create a reference system.    
    from simtk.pyopenmm.extras import testsystems

    print "Creating Lennard-Jones cluster system..."
    #[reference_system, coordinates] = testsystems.LennardJonesFluid()
    #receptor_atoms = [0]
    #ligand_atoms = [1]

    [reference_system, coordinates] = testsystems.LysozymeImplicit()
    receptor_atoms = range(0,2603) # T4 lysozyme L99A
    ligand_atoms = range(2603,2621) # p-xylene

    import simtk.unit as units
    unit = coordinates.unit
    coordinates = units.Quantity(numpy.array(coordinates / unit), unit)

    factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
    alchemical_state = AlchemicalState(0.00, 0.00, 0.00, 1.0)

    # Create the perturbed system.
    print "Creating alchemically-modified state..."
    alchemical_system = factory.createPerturbedSystem(alchemical_state)    
    # Compare energies.
    import simtk.unit as units
    import simtk.openmm as openmm
    timestep = 1.0 * units.femtosecond
    print "Computing reference energies..."
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(reference_system, integrator)
    context.setPositions(coordinates)
    state = context.getState(getEnergy=True)
    reference_potential = state.getPotentialEnergy()    
    del state, context, integrator
    print reference_potential
    print "Computing alchemical energies..."
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(alchemical_system, integrator)
    dr = 0.1 * units.angstroms # TODO: Why does 0.1 cause periodic 'nan's?
    a = receptor_atoms[-1]
    b = ligand_atoms[-1]
    delta = coordinates[a,:] - coordinates[b,:]
    for k in range(3):
        coordinates[ligand_atoms,k] += delta[k]
    for i in range(30):
        r = dr * i
        coordinates[ligand_atoms,0] += dr
          
        context.setPositions(coordinates)
        state = context.getState(getEnergy=True)
        alchemical_potential = state.getPotentialEnergy()    
        print "%8.3f A : %f " % (r / units.angstroms, alchemical_potential / units.kilocalories_per_mole)
    del state, context, integrator

    return

if __name__ == "__main__":
    # Run doctests.
    import doctest
    #doctest.testmod()

    # Run overlap tests.
    #test_overlap()
    
    # Run tests on individual systems.
    from simtk.pyopenmm.extras import testsystems

    print "Creating Lennard-Jones fluid system without dispersion correction..."
    [reference_system, coordinates] = testsystems.LennardJonesFluid(dispersion_correction=False)
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(2,3) # second atom
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating Lennard-Jones fluid system with dispersion correction..."
    [reference_system, coordinates] = testsystems.LennardJonesFluid(dispersion_correction=True)
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(2,3) # second atom
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating T4 lysozyme system..."
    [reference_system, coordinates] = testsystems.LysozymeImplicit()
    receptor_atoms = range(0,2603) # T4 lysozyme L99A
    ligand_atoms = range(2603,2621) # p-xylene
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)    
    print ""

    print "Creating Lennard-Jones cluster..."
    [reference_system, coordinates] = testsystems.LennardJonesCluster()
    ligand_atoms = range(0,1) # first atom
    receptor_atoms = range(1,2) # second atom
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating alanine dipeptide implicit system..."
    [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
    ligand_atoms = range(0,4) # methyl group
    receptor_atoms = range(4,22) # rest of system
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating alanine dipeptide explicit system..."
    [reference_system, coordinates] = testsystems.AlanineDipeptideExplicit()
    ligand_atoms = range(0,22) # alanine residue
    receptor_atoms = range(22,25) # one water
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

    print "Creating alanine dipeptide explicit system without dispersion correction..."
    #forces = { reference_system.getForce(index).__class__.__name__ : reference_system.getForce(index)) for index in range(reference_system.getNumForces()) } # requires Python 2.7 features
    import pdb
    pdb.set_trace()
    forces = dict( (reference_system.getForce(index).__class__.__name__,  reference_system.getForce(index)) for index in range(reference_system.getNumForces()) ) # python 2.6 compatible
    forces['NonbondedForce'].setUseDispersionCorrection(False) # turn off dispersion correction
    ligand_atoms = range(0,22) # alanine residue
    receptor_atoms = range(22,25) # one water
    testAlchemicalFactory(reference_system, coordinates, receptor_atoms, ligand_atoms)
    print ""

