"""
BASIS_ANALYSIS.PY

Written by Levi Naden with samples from John Chodera.

This function houses all the scripts nessicary to manipulate ncdata, construct linear basis functions, and analyze basis function variances.

This is a hybrid of the previous ncdata and linfuns modules, and a new constuction for variance analysis.
Requires the timesereies module
"""

import numpy
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import UnivariateSpline as US
from scipy.integrate import simps
import os
import os.path
import sys
import math
import netCDF4 as netcdf # netcdf4-python
from numpy.random import random_integers

from pymbar import MBAR # multistate Bennett acceptance ratio
import timeseries # for statistical inefficiency analysis

import simtk.unit as units

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

"""
Class: LinFunctions
Custom Linear basis functions that I can just import
"""

class LinFunctions:
    #Creates a custom class to wrap up all the basis functions, My original basis function is default
    #Valid methods are:
        #"LinA" = Naden linear basis set 1
        #"LinB" = Shirts Linear Basis set 1
        #"PureLin" = Purely linear scaling
        #"Lin4" = Linear scaling wit lambda^4 instead of just lambda^1
    def _cubic_mono_hermite_spline(self, x,y):
        #Create a sequence of cubic hermite splines
        #Code based on Michalski's Python variant
        n = len(x)
        #Compute Secants
        secants = (y[1:] - y[:-1])/ (x[1:] - x[:-1])
        #Compute initial tangents
        tangents = numpy.zeros(n)
        tangents[0] = secants[0]
        tangents[-1] = secants[-1]
        tangents[1:-1] = (secants[:-1]+secants[1:])/2
        #Solve case where delta = 0
        m_to_change = numpy.compress((secants == 0.0), range(n))
        for i in m_to_change:
            tangents[i] = 0.0
            tangents[i+1] = 0.0
        #Create alpha and beta
        alpha = tangents[:-1] / secants
        beta = tangents[1:] / secants
        distance = alpha**2 + beta**2
        tau = 3.0 / numpy.sqrt(distance)
        #Find where the alpha and beta cannot be transcribed within a guarenteed monotonic circle of radius 3
        over = (distance > 9.0)
        m_to_change = numpy.compress(over, range(n))
        #Find where there is non monotonics
        notmono = numpy.logical_or(alpha < 0, beta < 0)
        m_to_mono = numpy.compress(notmono, range(n))
        #for j in m_to_mono:
        #   tangents[j] = 0
        #   tangents[j+1] = 0
        #Build the monotonics
        for i in m_to_change:
            #check to see if i is in m_to_mono and dont touch it if it is
            #if i in m_to_mono:
            #    continue
            #else:
                tangents[i] = tau[i] * alpha[i] * secants[i]
                tangents[i+1] = tau[i] * beta[i] * secants[i]
        return tangents
    
    #---------------------------------------------------------------------------------------------
    
    def _hermite_spline_point(self, x_in,x_all,y_all,m_all, point_or_der):
        #Build a set of y values to pass back over the hermite spline
        output = numpy.zeros([len(x_in)])
        for i in range(len(x_in)):
            if x_in[i] == x_all[0]:
                ndx_min = 0
            else:
                ndx_min = numpy.where(x_all < x_in[i])[0][-1]
            ndx_max = ndx_min + 1
            x_min = x_all[ndx_min]
            x_max = x_all[ndx_max]
            m_min = m_all[ndx_min]
            m_max = m_all[ndx_max]
            y_min = y_all[ndx_min]
            y_max = y_all[ndx_max]
            h = x_max - x_min
            t = (x_in[i] - x_min)/h
            if point_or_der is 'point':
                h00 = (1+2*t) * (1-t)**2
                h10 = t * (1-t)**2
                h01 = t**2 * (3-2*t)
                h11 = t**2 * (t-1)
                output[i] = y_min*h00 + h*m_min*h10 + y_max*h01 + h*m_max*h11
            elif point_or_der is 'der':
                #dh/dx = dh/dt * dt/dx, dt/dx = 1/h
                dh00 = (6*t**2 - 6*t)/h
                dh10 = (3*t**2 - 4*t + 1)/h
                dh01 = (-6*t**2 + 6*t)/h
                dh11 = (3*t**2 - 2*t)/h
                output[i] = y_min*dh00 + h*m_min*dh10 + y_max*dh01 + h*m_max*dh11
        return output
    def _hermite_spline(self, x_in,y_min,y_max,m_min,m_max):
        #Build a set of y values to pass back over the hermite spline
        x_min = x_in[0]
        x_max = x_in[-1]
        h = x_max - x_min
        t = (x_in - x_min)/h
        h00 = (1+2*t) * (1-t)**2
        h10 = t * (1-t)**2
        h01 = t**2 * (3-2*t)
        h11 = t**2 * (t-1)
        #dh/dx = dh/dt * dt/dx, dt/dx = 1/h
        dh00 = (6*t**2 - 6*t)/h
        dh10 = (3*t**2 - 4*t + 1)/h
        dh01 = (-6*t**2 + 6*t)/h
        dh11 = (3*t**2 - 2*t)/h
        y = y_min*h00 + h*m_min*h10 + y_max*h01 + h*m_max*h11
        dy = y_min*dh00 + h*m_min*dh10 + y_max*dh01 + h*m_max*dh11
        return y,dy
    #---------------------------------------------------------------------------------------------
    def _buildHermite(self, x,y,n_between):
        #Find the tangents (will be needed)
        m = self._cubic_mono_hermite_spline(x,y)
        n = len(x)
        #Create the sequence of intermediate points to fill in the gaps
        x_filled = numpy.empty(0,numpy.float64)
        yr_filled = numpy.empty(0,numpy.float64)
        dyr_filled = numpy.empty(0,numpy.float64)
        for i in range(n-1):
            #Create the spacing
            x_to_hermite = scipy.linspace(x[i],x[i+1],n_between+2)
            (yr_hermite_out,dyr_herm_out) = self._hermite_spline(x_to_hermite,y[i],y[i+1],m[i],m[i+1])
            x_filled = numpy.append(x_filled[:-1], x_to_hermite)
            yr_filled = numpy.append(yr_filled[:-1], yr_hermite_out)
            dyr_filled = numpy.append(dyr_filled[:-1], dyr_herm_out)
        return x_filled,yr_filled,dyr_filled
    #---------------------------------------------------------------------------------------------
    def _unboxLinA(self, hrconstant): #Unwraps the Naden 1 basis set
        self.h_r_const = hrconstant #Constant for Naden's H_R(lambda)
        self.h_r = lambda L: (self.h_r_const**L - 1)/(self.h_r_const - 1)
        self.dh_r = lambda L: (numpy.log(self.h_r_const)*self.h_r_const**L)/(self.h_r_const-1)
        self.d2h_r = lambda L: ((numpy.log(self.h_r_const)**2)*self.h_r_const**L)/(self.h_r_const-1)
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.h_r_inv = lambda h: numpy.log(h*(self.h_r_const-1)+1)/numpy.log(self.h_r_const)
        self.h_a_inv = lambda h: h

    def _unboxLinB(self, prefactor, normvar): #Unwraps the Shirts 1 basis set
        self.prefac = prefactor #Constant for prefactor in Shirts' 1 basis
        self.normvar = normvar #Constant for the variance factor in the normal distro. of Shirts' 1 function
        self.expfactor = lambda L: numpy.exp(-((1-L)/self.normvar)**2)
        self.h_r = lambda L: (self.prefac*L) + (L*(1-self.prefac)*self.expfactor(L))
        self.dh_r = lambda L: (self.prefac) + ((1-self.prefac)*self.expfactor(L)) + (2*(1-L)*L*(1-self.prefac)*self.expfactor(L)/(self.normvar**2))
        self.d2h_r = lambda L: (self.expfactor(L)/self.normvar**2)*(4*L*(1-self.prefac)*(1-L)**2/self.normvar**2 + 4*(1-self.prefac)*(1-L) - 2*L*(1-self.prefac))
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0
        #Construct the invere function
        x = scipy.linspace(0,1,501)
        y = self.h_r(x)
        self.h_r_inv = IUS(y,x)
        self.h_a_inv = lambda h: h

    def _unboxPureLin(self): #Unwraps the Pure Linear Transformation
        self.h_r = lambda L: L
        self.dh_r = lambda L: 1
        self.d2h_r = lambda L: 0
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0

    def _unboxLin4(self): #Unwraps the Lambda^4 transformation
        self.h_r = lambda L: L**4
        self.dh_r = lambda L: 4*L**3
        self.d2h_r = lambda L: 12*L**2
        self.h_a = lambda L: L**4
        self.dh_a = lambda L: 4*L**3
        self.d2h_a = lambda L: 12*L**2
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0

    def _unboxLinGeneral(self,hrconstant,haconstant): #Unwraps the Lambda^4 transformation
        self.h_r_constant = hrconstant
        self.h_a_constant = haconstant
        self.h_r = lambda L: L**self.h_r_constant
        self.dh_r = lambda L: self.h_r_constant*L**(self.h_r_constant-1)
        if self.h_r_constant == 1:
            self.d2h_r = lambda L: 0
        else:
            self.d2h_r = lambda L: (self.h_r_constant)*(self.h_r_constant-1)*L**(self.h_r_constant-2)
        self.h_a = lambda L: L**self.h_a_constant
        self.dh_a = lambda L: self.h_a_constant*L**(self.h_a_constant-1)
        if self.h_a_constant == 1:
            self.d2h_a = lambda L: 0
        else:
            self.d2h_a = lambda L: (self.h_a_constant)*(self.h_a_constant-1)*L**(self.h_a_constant-2)
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0
        self.h_r_inv = lambda h: h**(1.0/self.h_r_constant)
        self.h_a_inv = lambda h: h**(1.0/self.h_a_constant)

    def _unboxLinHybrid(self,hrconstant,haconstant): #Unwraps a hybrid combination
        self.h_r_constant = hrconstant
        self.h_a_constant = haconstant
        self.h_r = lambda L: (self.h_r_constant**L - 1)/(self.h_r_constant - 1)
        self.dh_r = lambda L: (numpy.log(self.h_r_constant)*self.h_r_constant**L)/(self.h_r_constant-1)
        self.d2h_r = lambda L: ((numpy.log(self.h_r_constant)**2)*self.h_r_constant**L)/(self.h_r_constant-1)
        self.h_a = lambda L: L**self.h_a_constant
        self.dh_a = lambda L: self.h_a_constant*L**(self.h_a_constant-1)
        if self.h_a_constant == 1:
            self.d2h_a = lambda L: 0
        else:
            self.d2h_a = lambda L: (self.h_a_constant)*(self.h_a_constant-1)*L**(self.h_a_constant-2)
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0
        self.h_r_inv = lambda h: numpy.log(h*(self.h_r_const-1)+1)/numpy.log(self.h_r_const)
        self.h_a_inv = lambda h: h**(1.0/self.h_a_constant)
    
    def _unboxSin(self):
        self.h_r = lambda L: numpy.sin(L*numpy.pi/2) 
        self.dh_r = lambda L: (numpy.pi/2) * numpy.cos(L*numpy.pi/2)
        self.d2h_r = lambda L: -(numpy.pi/2)**2 * numpy.sin(L*numpy.pi/2)
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0

    def _unboxExpGen(self,C1,C2,prefactor):
        if prefactor:
            self.h_r = lambda L: L*(prefactor + (1-prefactor)*((numpy.exp(L**C1)-1)/(numpy.exp(1)-1))**C2)
            self.dh_r = lambda L: prefactor + L*(1-prefactor)*(C1*C2*((numpy.exp(L**C1)-1)/(numpy.exp(1)-1))**(C2-1)*(L**(C1-1))*numpy.exp(L**C1) + ((numpy.exp(L**C1)-1)/(numpy.exp(1)-1))**C2)
        else:
            self.h_r = lambda L: ((numpy.exp(L**C1)-1)/(numpy.exp(1)-1))**C2
            self.dh_r = lambda L: C1*C2*((numpy.exp(L**C1)-1)/(numpy.exp(1)-1))**(C2-1)*(L**(C1-1))*numpy.exp(L**C1)
        #self.d2h_r = lambda L: L #placeolder for now
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0
        
    def _unboxOptimal(self, C1=1.61995584, C2=-0.8889962, C3=0.02552684):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        #This is a special constructor that was built from an optimization routine, then curve fit
        self.h_r = lambda L: C1*L**4 + C2*L**3 + C3*L**2 + (1-C1-C2-C3)*L 
        self.dh_r = lambda L: 4*C1*L**3 + 3*C2*L**2 + 2*C3*L + (1-C1-C2-C3)
        self.d2h_r = lambda L: 12*C1*L**2 + 6*C2*L + 2*C3
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0
        #Construct the invere function
        x = scipy.linspace(0,1,501)
        y = self.h_r(x)
        self.h_r_inv = IUS(y,x)
        self.h_a_inv = lambda h: h

    def _unboxHermiteOptimal(self):
        #This is a special constructor that was built from an optimization routine
        Npoints=5
        x_herm = numpy.concatenate( (scipy.linspace(0,0.3,Npoints), scipy.linspace(.3,1,Npoints)[1:]) )
        y_herm = numpy.array([0.0, 0.01950203, 0.0351703, 0.04887419, 0.06248457, 0.11138664, 0.21265207, 0.46874771,1.0])
        m_herm = self._cubic_mono_hermite_spline(x_herm,y_herm)
        interm_n = 100
        #Construct inverse
        (xout, yout, dr)=self._buildHermite(x_herm,y_herm,interm_n)
        self.h_r = lambda L: self._hermite_spline_point(L,x_herm,y_herm,m_herm,'point')
        self.dh_r = lambda L: self._hermite_spline_point(L,x_herm,y_herm,m_herm,'der')
        self.h_a = lambda L: L
        self.dh_a = lambda L: 1
        self.d2h_a = lambda L: 0
        self.h_e = lambda L: L
        self.dh_e = lambda L: 1
        self.d2h_e = lambda L: 0
        self.h_r_inv = IUS(yout,xout)
        self.h_a_inv = lambda h: h

    def _unboxHermiteGeneral(self, lam_range, fullg_r, fullg_a, fullg_e):
        #Construct hermite vlaues
        m_herm_r = self._cubic_mono_hermite_spline(lam_range,fullg_r)
        m_herm_a = self._cubic_mono_hermite_spline(lam_range,fullg_a)
        m_herm_e = self._cubic_mono_hermite_spline(lam_range,fullg_e)
        interm_n = 100
        (xout_r, yout_r, dr)=self._buildHermite(lam_range,fullg_r,interm_n)
        (xout_a, yout_a, da)=self._buildHermite(lam_range,fullg_a,interm_n)
        (xout_e, yout_e, de)=self._buildHermite(lam_range,fullg_e,interm_n)
        #Constuct functions, not worying about second derivative for now
        self.h_r = lambda L: self._hermite_spline_point(L,lam_range,fullg_r,m_herm_r,'point')
        self.dh_r = lambda L: self._hermite_spline_point(L,lam_range,fullg_r,m_herm_r,'der')
        self.h_a = lambda L: self._hermite_spline_point(L,lam_range,fullg_a,m_herm_a,'point')
        self.dh_a = lambda L: self._hermite_spline_point(L,lam_range,fullg_a,m_herm_a,'der')
        self.h_e = lambda L: self._hermite_spline_point(L,lam_range,fullg_e,m_herm_e,'point')
        self.dh_e = lambda L: self._hermite_spline_point(L,lam_range,fullg_e,m_herm_e,'der')
        self.h_r_inv = IUS(yout_r,xout_r)
        self.h_a_inv = IUS(yout_a,xout_a)
        self.h_e_inv = IUS(yout_e,xout_e)
        
    def __init__(self, method="LinA", **kwargs):
        self.method = method
        #Create the constructor list
        if self.method is "LinA":
            #Unbox the dictionary
            if 'hrconstant' in kwargs:
                hrconstant = kwargs['hrconstant']
            else:
                hrconstant = 35
            self._unboxLinA(hrconstant=hrconstant)
        elif self.method is "LinB":
            if 'prefactor' in kwargs:
                prefactor = kwargs['prefactor']
            else:
                prefactor = 0.22
            if 'normalizer' in kwargs:
                normvar = kwargs['normalizer']
            else:
                normvar = 0.284
            self._unboxLinB(prefactor, normvar)
        elif self.method is "PureLin":
            self._unboxPureLin()
        elif self.method is "Lin4":
            self._unboxLin4()
        elif self.method is "LinGeneral":
            if 'hrconstant' in kwargs:
                hrconstant = kwargs['hrconstant']
            else:
                hrconstant = 4
            if 'haconstant' in kwargs:
                haconstant = kwargs['haconstant']
            else:
                haconstant = 1
            self._unboxLinGeneral(hrconstant, haconstant)
        elif self.method is "LinHybrid":
            if 'hrconstant' in kwargs:
                hrconstant = kwargs['hrconstant']
            else:
                hrconstant = 10
            if 'haconstant' in kwargs:
                haconstant = kwargs['haconstant']
            else:
                haconstant = 4
            self._unboxLinHybrid(hrconstant, haconstant)
        elif self.method is "Sin":
            #Special test method which should be bad
            self._unboxSin()
        elif self.method is "ExpGen":
            #Exponent method
            if 'C1' in kwargs:
                C1 = kwargs['C1']
            else:
                C1 = 1
            if 'C2' in kwargs:
                C2 = kwargs['C2']
            else:
                C2 = 1
            if not 'prefactor' in kwargs:
                prefactor = None
            else:
                prefactor = kwargs['prefactor']
            self._unboxExpGen(C1,C2,prefactor)
        elif self.method is "Optimal":
            self._unboxOptimal(**kwargs)
        elif self.method is 'HermiteOptimal':
            #Special case that manipulates the hermite splines
            self._unboxHermiteOptimal()
        elif self.method is 'HermiteGeneral':
            if 'lam_range' in kwargs:
                lam_range=kwargs['lam_range']
            else:
                lam_range=scipy.linspace(0,1,11)
            #if 'interm_n' in kwargs:
            #    interm_n=kwargs['interm_n']
            #else:
            #    interm_n=2
            if 'fullg_r' in kwargs:
                fullg_r = kwargs['fullg_r']
            else:
                fullg_r = scipy.linspace(0,1,len(lam_range))
            if 'fullg_a' in kwargs:
                fullg_a = kwargs['fullg_a']
            else:
                fullg_a = scipy.linspace(0,1,len(lam_range))
            if 'fullg_e' in kwargs:
                fullg_e = kwargs['fullg_e']
            else:
                fullg_e = scipy.linspace(0,1,len(lam_range))
            self._unboxHermiteGeneral(lam_range, fullg_r, fullg_a, fullg_e)

        return
##############################################################################################

"""
Class: ncdata
Manipulate the ncdata files analyze results from YANK. Most code from John Chodera's analyze.py script in YANK
"""
class ncdata:

    def _read_pdb(self, filename):
        """
        Read the contents of a PDB file.
    
        ARGUMENTS
    
        filename (string) - name of the file to be read
    
        RETURNS
    
        atoms (list of dict) - atoms[index] is a dict of fields for the ATOM residue
    
        """
        
        # Read the PDB file into memory.
        pdbfile = open(filename, 'r')
    
        # Extract the ATOM entries.
        # Format described here: http://bmerc-www.bu.edu/needle-doc/latest/atom-format.html
        atoms = list()
        for line in pdbfile:
            if line[0:6] == "ATOM  ":
                # Parse line into fields.
                atom = dict()
                atom["serial"] = line[6:11]
                atom["atom"] = line[12:16]
                atom["altLoc"] = line[16:17]
                atom["resName"] = line[17:20]
                atom["chainID"] = line[21:22]
                atom["Seqno"] = line[22:26]
                atom["iCode"] = line[26:27]
                atom["x"] = line[30:38]
                atom["y"] = line[38:46]
                atom["z"] = line[46:54]
                atom["occupancy"] = line[54:60]
                atom["tempFactor"] = line[60:66]
                atoms.append(atom)
                
        # Close PDB file.
        pdbfile.close()
    
        # Return dictionary of present residues.
        return atoms
    
    def _write_pdb(self, atoms, filename, iteration, replica, title, ncfile,trajectory_by_state=True):
        """Write out replica trajectories as multi-model PDB files.
    
        ARGUMENTS
           atoms (list of dict) - parsed PDB file ATOM entries from read_pdb() - WILL BE CHANGED
           filename (string) - name of PDB file to be written
           title (string) - the title to give each PDB file
           ncfile (NetCDF) - NetCDF file object for input file       
        """
    
        # Extract coordinates to be written.
        coordinates = numpy.array(ncfile.variables['positions'][iteration,replica,:,:])
        coordinates *= 10.0 # convert nm to angstroms
    
        # Create file.
        #outfile = open(filename, 'w')
    
        # Write ATOM records.
        for (index, atom) in enumerate(atoms):
            atom["x"] = "%8.3f" % coordinates[index,0]
            atom["y"] = "%8.3f" % coordinates[index,1]
            atom["z"] = "%8.3f" % coordinates[index,2]
            filename.write('ATOM  %(serial)5s %(atom)4s%(altLoc)c%(resName)3s %(chainID)c%(Seqno)5s   %(x)8s%(y)8s%(z)8s\n' % atom)
            
        # Close file.
        #outfile.close()
    
        return
    
    def _write_crd(self, filename, iteration, replica, title, ncfile):
        """
        Write out AMBER format CRD file.
    
        """
        # Extract coordinates to be written.
        coordinates = numpy.array(ncfile.variables['positions'][iteration,replica,:,:])
        coordinates *= 10.0 # convert nm to angstroms
    
        # Create file.
        outfile = open(filename, 'w')
    
        # Write title.
        outfile.write(title + '\n')
    
        # Write number of atoms.
        natoms = ncfile.variables['positions'].shape[2]
        outfile.write('%6d\n' % natoms)
    
        # Write coordinates.
        for index in range(natoms):
            outfile.write('%12.7f%12.7f%12.7f' % (coordinates[index,0], coordinates[index,1], coordinates[index,2]))
            if ((index+1) % 2 == 0): outfile.write('\n')
            
        # Close file.
        outfile.close()
        
    def _write_gro(self, atoms, filename, iteration, replica, title, trajectory_by_state=True):
        """Write out replica trajectories as multi-model GRO files.
    
        ARGUMENTS
           atoms (list of dict) - parsed PDB file ATOM entries from read_pdb() - WILL BE CHANGED
           filename (string) - name of PDB file to be written
           title (string) - the title to give each PDB file
        """
    
        # Extract coordinates to be written (comes out in nm)
        coordinates = numpy.array(self.ncfile.variables['positions'][iteration,replica,:,:])
    
        # Create file.
        #outfile = open(filename, 'w')
    
        # Write ATOM records.
        for (index, atom) in enumerate(atoms):
            #atom["x"] = "%8.3f" % coordinates[index,0]
            #atom["y"] = "%8.3f" % coordinates[index,1]
            #atom["z"] = "%8.3f" % coordinates[index,2]
            #Increasing precision
            atom["x"] = "%8f" % coordinates[index,0]
            atom["y"] = "%8f" % coordinates[index,1]
            atom["z"] = "%8f" % coordinates[index,2]
            #               ResNumber ResName     AtomName AtomNumber  X-pos  Y-pos  Z-pos
            filename.write('%(Seqno)5s%(resName)5s%(atom)5s%(serial)5s %(x)8s %(y)8s %(z)8s\n' % atom)
            
        # Close file.
        #outfile.close()
    
        return

    def write_gro_replica_trajectories(self, directory, prefix, title, trajectory_by_state=True, fraction_to_write=None, equilibrated_data = False, uncorrelated_data = False):
        """Write out replica trajectories as multi-model GRO files.
    
        ARGUMENTS
           directory (string) - the directory to write files to
           prefix (string) - prefix for replica trajectory files
           title (string) - the title to give each PDB file
           trajectory_by_state (boolean) - If true, write trajectories by alchemical state, not by replica
           fraction_to_write (float, [0,1]) - Leading fraction of iterations to write out, used to make smaller files
           equilibrated_data (boolean) - Only use the data after the equilibrated region
           uncorrelated_data (boolean) - Only use the uncorrelated, sub-sampled data; implies equilibrated_data
        """
        atom_list=self._read_pdb(self.reference_pdb_filename)
        if (len(atom_list) != self.natoms):
            print ("Number of atoms in trajectory (%d) differs from number of atoms in reference PDB (%d)." % (self.natoms, len(atom_list)))
            raise Exception

        #Determine which pool we are sampling from
        output_indices = numpy.array(range(self.niterations))
        if uncorrelated_data:
           #Truncate the opening sequence, then retain only the entries which match with the indicies of the subsampled set 
           output_indices = output_indices[self.nequil:][self.retained_indices]
        elif equilibrated_data:
           output_indices = output_indices[self.nequil:]
        #Set up number of samples to go throguh
        if fraction_to_write > 1 or fraction_to_write is None:
            fraction_to_write = 1
        max_samples=int(len(output_indices)*fraction_to_write)

        if trajectory_by_state:
            for state_index in range(0,self.nstates):
                print "Working on state %d / %d" % (state_index,self.nstates)  
            	file_name= "%s-%03d.gro" % (prefix,state_index)
    		full_filename=directory+'/'+file_name
    		outfile = open(full_filename, 'w')
    		for iteration in output_indices[:max_samples]: #Only go through the retained indicies
                    state_indices = self.ncfile.variables['states'][iteration,:]
                    replica_index = list(state_indices).index(state_index)
                    outfile.write('%s phase data at iteration %4d\n' % (self.phase, iteration)) #Header
                    outfile.write('%d\n' % self.natoms) #Atom count header
                    self._write_gro(atom_list,outfile,iteration,replica_index,title,trajectory_by_state=True)
                    box_x = self.ncfile.variables['box_vectors'][iteration,replica_index,0,0]
                    box_y = self.ncfile.variables['box_vectors'][iteration,replica_index,1,1]
                    box_z = self.ncfile.variables['box_vectors'][iteration,replica_index,2,2]
                    #outfile.write('    %.4f    %.4f    %.4f\n' % (box_x, box_y, box_z)) #Box vectors output
                    outfile.write('    %8f    %8f    %8f\n' % (box_x, box_y, box_z)) #Box vectors output
    		
    		outfile.close()	
    
        else:
            for replica_index in range(nstates):
                print "Working on replica %d / %d" % (replica_index,nstates)
                file_name="R-%s-%03d.gro" % (prefix,replica_index)
                full_filename=directory+'/'+file_name
                outfile = open(full_filename, 'w')
                for iteration in output_indices[:max_samples]: #Only go through the retained indicies
                    outfile.write('%s of uncorrelated data at iteration %4d\n' % (self.phase, iteration)) #Header
                    outfile.write('%d\n' % self.natoms) #Atom count header
                    self._write_gro(atom_list,outfile,iteration,replica_index,title,trajectory_by_state=True)
                    box_x = self.ncfile.variables['box_vectors'][iteration,replica_index,0,0]
                    box_y = self.ncfile.variables['box_vectors'][iteration,replica_index,1,1]
                    box_z = self.ncfile.variables['box_vectors'][iteration,replica_index,2,2]
                    outfile.write('    %.4f    %.4f    %.4f\n' % (box_x, box_y, box_z)) #Box vectors output
                outfile.close()
    		
        return

    def _show_mixing_statistics(self, ncfile, cutoff=0.05, nequil=0):
        """
        Print summary of mixing statistics.
    
        ARGUMENTS
    
        ncfile (netCDF4.Dataset) - NetCDF file
        
        OPTIONAL ARGUMENTS
    
        cutoff (float) - only transition probabilities above 'cutoff' will be printed (default: 0.05)
        nequil (int) - if specified, only samples nequil:end will be used in analysis (default: 0)
        
        """
        
        # Get dimensions.
        niterations = ncfile.variables['states'].shape[0]
        nstates = ncfile.variables['states'].shape[1]
    
        # Compute statistics of transitions.
        Nij = numpy.zeros([nstates,nstates], numpy.float64)
        for iteration in range(nequil, niterations-1):
            for ireplica in range(nstates):
                istate = ncfile.variables['states'][iteration,ireplica]
                jstate = ncfile.variables['states'][iteration+1,ireplica]
                Nij[istate,jstate] += 0.5
                Nij[jstate,istate] += 0.5
        Tij = numpy.zeros([nstates,nstates], numpy.float64)
        for istate in range(nstates):
            Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()
    
        # Print observed transition probabilities.
        print "Cumulative symmetrized state mixing transition matrix:"
        print "%6s" % "",
        for jstate in range(nstates):
            print "%6d" % jstate,
        print ""
        for istate in range(nstates):
            print "%-6d" % istate,
            for jstate in range(nstates):
                P = Tij[istate,jstate]
                if (P >= cutoff):
                    print "%6.3f" % P,
                else:
                    print "%6s" % "",
            print ""
    
        # Estimate second eigenvalue and equilibration time.
        mu = numpy.linalg.eigvals(Tij)
        mu = -numpy.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            print "Perron eigenvalue is unity; Markov chain is decomposable."
        else:
            print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))
            
        return
    
    def _analyze_acceptance_probabilities(self, ncfile, cutoff = 0.4):
        """Analyze acceptance probabilities.
    
        ARGUMENTS
           ncfile (NetCDF) - NetCDF file to be analyzed.
    
        OPTIONAL ARGUMENTS
           cutoff (float) - cutoff for showing acceptance probabilities as blank (default: 0.4)
        """
    
        # Get current dimensions.
        niterations = ncfile.variables['mixing'].shape[0]
        nstates = ncfile.variables['mixing'].shape[1]
    
        # Compute mean.
        mixing = ncfile.variables['mixing'][:,:,:]
        Pij = mean(mixing, 0)
    
        # Write title.
        print "Average state-to-state acceptance probabilities"
        print "(Probabilities less than %(cutoff)f shown as blank.)" % vars()
        print ""
    
        # Write header.
        print "%4s" % "",
        for j in range(nstates):
            print "%6d" % j,
        print ""
    
        # Write rows.
        for i in range(nstates):
            print "%4d" % i, 
            for j in range(nstates):
                if Pij[i,j] > cutoff:
                    print "%6.3f" % Pij[i,j],
                else:
                    print "%6s" % "",
                
            print ""
    
        return
    
    def _check_energies(self, ncfile, atoms, verbose=False):
        """
        Examine energy history for signs of instability (nans).
    
        ARGUMENTS
           ncfile (NetCDF) - input YANK netcdf file
        """
    
        # Get current dimensions.
        niterations = ncfile.variables['energies'].shape[0]
        nstates = ncfile.variables['energies'].shape[1]
    
        # Extract energies.
        if verbose: print "Reading energies..."
        energies = ncfile.variables['energies']
        u_kln_replica = numpy.zeros([nstates, nstates, niterations], numpy.float64)
        for n in range(niterations):
            u_kln_replica[:,:,n] = energies[n,:,:]
        if verbose: print "Done."
    
        # Deconvolute replicas
        if verbose: print "Deconvoluting replicas..."
        u_kln = numpy.zeros([nstates, nstates, niterations], numpy.float64)
        for iteration in range(niterations):
            state_indices = ncfile.variables['states'][iteration,:]
            u_kln[state_indices,:,iteration] = energies[iteration,:,:]
        if verbose: print "Done."
    
        if verbose:
            # Show all self-energies
            show_self_energies = False
            if (show_self_energies):
                print 'all self-energies for all replicas'
                for iteration in range(niterations):
                    for replica in range(nstates):
                        state = int(ncfile.variables['states'][iteration,replica])
                        print '%12.1f' % energies[iteration, replica, state],
                    print ''
    
        # If no energies are 'nan', we're clean.
        if not numpy.any(numpy.isnan(energies[:,:,:])):
            return
    
        # There are some energies that are 'nan', so check if the first iteration has nans in their *own* energies:
        u_k = numpy.diag(energies[0,:,:])
        if numpy.any(numpy.isnan(u_k)):
            print "First iteration has exploded replicas.  Check to make sure structures are minimized before dynamics"
            print "Energies for all replicas after equilibration:"
            print u_k
            sys.exit(1)
    
        # There are some energies that are 'nan' past the first iteration.  Find the first instances for each replica and write PDB files.
        first_nan_k = numpy.zeros([nstates], numpy.int32)
        for iteration in range(niterations):
            for k in range(nstates):
                if numpy.isnan(energies[iteration,k,k]) and first_nan_k[k]==0:
                    first_nan_k[k] = iteration
        if not all(first_nan_k == 0):
            print "Some replicas exploded during the simulation."
            print "Iterations where explosions were detected for each replica:"
            print first_nan_k
            print "Writing PDB files immediately before explosions were detected..."
            for replica in range(nstates):            
                if (first_nan_k[replica] > 0):
                    state = ncfile.variables['states'][iteration,replica]
                    iteration = first_nan_k[replica] - 1
                    filename = 'replica-%d-before-explosion.pdb' % replica
                    title = 'replica %d state %d iteration %d' % (replica, state, iteration)
                    write_pdb(atoms, filename, iteration, replica, title, ncfile)
                    filename = 'replica-%d-before-explosion.crd' % replica                
                    write_crd(filename, iteration, replica, title, ncfile)
            sys.exit(1)
    
        # There are some energies that are 'nan', but these are energies at foreign lambdas.  We'll just have to be careful with MBAR.
        # Raise a warning.
        print "WARNING: Some energies at foreign lambdas are 'nan'.  This is recoverable."
            
        return
    
    def _check_positions(self, ncfile):
        """Make sure no positions have gone 'nan'.
    
        ARGUMENTS
           ncfile (NetCDF) - NetCDF file object for input file
        """
    
        # Get current dimensions.
        niterations = ncfile.variables['positions'].shape[0]
        nstates = ncfile.variables['positions'].shape[1]
        natoms = ncfile.variables['positions'].shape[2]
    
        # Compute torsion angles for each replica
        for iteration in range(niterations):
            for replica in range(nstates):
                # Extract positions
                positions = numpy.array(ncfile.variables['positions'][iteration,replica,:,:])
                # Check for nan
                if numpy.any(numpy.isnan(positions)):
                    # Nan found -- raise error
                    print "Iteration %d, state %d - nan found in positions." % (iteration, replica)
                    # Report coordinates
                    for atom_index in range(natoms):
                        print "%16.3f %16.3f %16.3f" % (positions[atom_index,0], positions[atom_index,1], positions[atom_index,2])
                        if numpy.any(numpy.isnan(positions[atom_index,:])):
                            raise "nan detected in positions"
    
        return
    
    def _extract_u_n(self, ncfile, verbose=False):
        """
        Extract timeseries of u_n = - log q(x_n)
    
        """
    
        # Get current dimensions.
        niterations = ncfile.variables['energies'].shape[0]
        nstates = ncfile.variables['energies'].shape[1]
        natoms = ncfile.variables['energies'].shape[2]
    
        # Extract energies.
        if verbose: print "Reading energies..."
        energies = ncfile.variables['energies']
        u_kln_replica = numpy.zeros([nstates, nstates, niterations], numpy.float64)
        for n in range(niterations):
            u_kln_replica[:,:,n] = energies[n,:,:]
        if verbose: print "Done."
    
        # Deconvolute replicas
        if verbose: print "Deconvoluting replicas..."
        u_kln = numpy.zeros([nstates, nstates, niterations], numpy.float64)
        for iteration in range(niterations):
            state_indices = ncfile.variables['states'][iteration,:]
            u_kln[state_indices,:,iteration] = energies[iteration,:,:]
        if verbose: print "Done."
    
        # Compute total negative log probability over all iterations.
        u_n = numpy.zeros([niterations], numpy.float64)
        for iteration in range(niterations):
            u_n[iteration] = numpy.sum(numpy.diagonal(u_kln[:,:,iteration]))
    
        return u_n
    
    def _detect_equilibration(self, A_t):
        """
        Automatically detect equilibrated region.
    
        ARGUMENTS
    
        A_t (numpy.array) - timeseries
    
        RETURNS
    
        t (int) - start of equilibrated data
        g (float) - statistical inefficiency of equilibrated data
        Neff_max (float) - number of uncorrelated samples   
        
        """
        T = A_t.size
    
        # Special case if timeseries is constant.
        if A_t.std() == 0.0:
            return (0, 1, T)
        
        g_t = numpy.ones([T-1], numpy.float32)
        Neff_t = numpy.ones([T-1], numpy.float32)
        for t in range(T-1):
            g_t[t] = timeseries.statisticalInefficiency(A_t[t:T])
            Neff_t[t] = (T-t+1) / g_t[t]
        
        Neff_max = Neff_t.max()
        t = Neff_t.argmax()
        g = g_t[t]
        
        return (t, g, Neff_max)

    def _build_u_kln(self, nuse = None):
        ndiscard = self.nequil
        # Extract energies.
        if self.verbose: 
            print "Building initial u_kln matrix..."
            print "Reading energies..."
        energies = self.ncfile.variables['energies']
        u_kln_replica = numpy.zeros([self.nstates, self.nstates, self.niterations], numpy.float64)
        for n in range(self.niterations):
            u_kln_replica[:,:,n] = energies[n,:,:]
        if self.verbose: print "Done."

        # Deconvolute replicas
        if self.verbose: print "Deconvoluting replicas..."
        u_kln = numpy.zeros([self.nstates, self.nstates, self.niterations], numpy.float64)
        for iteration in range(self.niterations):
            state_indices = self.ncfile.variables['states'][iteration,:]
            u_kln[state_indices,:,iteration] = energies[iteration,:,:]
        if self.verbose: print "Done."

        # Compute total negative log probability over all iterations.
        u_n = numpy.zeros([self.niterations], numpy.float64)
        for iteration in range(self.niterations):
            u_n[iteration] = numpy.sum(numpy.diagonal(u_kln[:,:,iteration]))
        self.u_kln = u_kln[:,:,ndiscard:]
        self.u_n = u_n[ndiscard:]

        # Truncate to number of specified conforamtions to use
        if (nuse):
            u_kln_replica = u_kln_replica[:,:,0:nuse]
            self.u_kln = self.u_kln[:,:,0:nuse]
            self.u_n = self.u_n[0:nuse]

        # Subsample data to obtain uncorrelated samples
        self.N_k = numpy.zeros(self.nstates, numpy.int32)    
        indices = timeseries.subsampleCorrelatedData(self.u_n) # indices of uncorrelated samples
        N = len(indices) # number of uncorrelated samples
        self.N_k[:] = N      
        #Original Line
        #self.u_kln[:,:,0:N] = self.u_kln[:,:,indices]
        #Modified line to discard corelated data, reduces memory and should not change result since the data past N is ignored by mbar
        self.u_kln = self.u_kln[:,:,indices]
        if self.verbose:
            print "number of uncorrelated samples:"
            print self.N_k
            print ""
        self.retained_iters = self.u_kln.shape[2]
        self.retained_indices = indices
        #!!! Temporary fix
        self.u_kln_raw = u_kln[:,:,indices]
        return

    def _AutoAlchemyStates(self, phase, real_R_states=None, real_A_states=None, real_E_states=None, alchemy_source=None):
        #Generate the real alchemical states automatically.
        if alchemy_source: #Load alchemy from an external source
            import imp
            if alchemy_source[-3:] != '.py': #Check if the file or the folder was provided
                alchemy_source = os.path.join(alchemy_source, 'alchemy.py')
            alchemy = imp.load_source('alchemy', alchemy_source)
            AAF = alchemy.AbsoluteAlchemicalFactory
        else: #Standard load
            from alchemy import AbsoluteAlchemicalFactory as AAF
        if phase is 'vacuum':
            protocol = AAF.defaultVacuumProtocol()
        elif phase is 'complex':
            protocol = AAF.defaultComplexProtocolExplicit()
        #Determine which phases need crunched
        if real_R_states is None:
            real_R_states = list()
            crunchR = True
        else:
            crunchR = False
        if real_A_states is None:
            real_A_states = list()
            crunchA = True
        else:
            crunchA = False
        if real_E_states is None:
            real_E_states = list()
            crunchE = True
        else:
            crunchE = False
        #Import from the alchemy file if need be
        for state in protocol: #Go through each state
            if crunchE:
                real_E_states.append(state.ligandElectrostatics)
            if crunchR:
                real_R_states.append(state.ligandRepulsion)
            if crunchA:
                real_A_states.append(state.ligandAttraction)
        #Determine cutoffs
        self.real_E_states = numpy.array(real_E_states)
        self.real_R_states = numpy.array(real_R_states)
        self.real_A_states = numpy.array(real_A_states)
        indicies = numpy.array(range(len(real_E_states)))
        #Set the indicies, trap TypeError (logical_and false everywhere) as None (i.e. state not found in alchemy)
        try:
            self.real_EAR = int(indicies[ numpy.logical_and(self.real_E_states == 1, numpy.logical_and(self.real_R_states == 1, self.real_A_states == 1)) ])
        except TypeError:
            self.real_EAR = None
        try:
            self.real_AR = int(indicies[ numpy.logical_and(self.real_E_states == 0, numpy.logical_and(self.real_R_states == 1, self.real_A_states == 1)) ])
        except TypeError:
            self.real_AR = None
        try:
            self.real_R = int(indicies[ numpy.logical_and(self.real_E_states == 0, numpy.logical_and(self.real_R_states == 1, self.real_A_states == 0)) ])
        except TypeError:
            self.real_R = None
        try:
            self.real_alloff = int(indicies[ numpy.logical_and(self.real_E_states == 0, numpy.logical_and(self.real_R_states == 0, self.real_A_states == 0)) ])
        except:
            self.real_alloff = None

        return

    def compute_mbar(self):
        self.mbar = MBAR(self.u_kln, self.N_k, verbose = self.verbose, method = 'adaptive')
        self.mbar_ready = True

    def __init__(self, phase, source_directory, verbose=False, real_R_states = None, real_A_states = None, real_E_states = None, compute_mbar = False, alchemy_source = None):
        self.phase = phase
        self.verbose = verbose
        self._AutoAlchemyStates(self.phase, real_R_states=real_R_states, real_A_states=real_A_states, real_E_states=real_E_states, alchemy_source=alchemy_source)
        if real_R_states:
            self.real_R_states = numpy.array(real_R_states)
        self.real_R_count = len(self.real_R_states)
        if real_A_states:
            self.real_A_states = numpy.array(real_A_states)
        self.real_A_count = len(self.real_A_states)
        if real_E_states:
            self.real_E_states = numpy.array(real_E_states)
        self.real_E_count = len(self.real_E_states)

        #Import file and grab the constants
        fullpath = os.path.join(source_directory, phase + '.nc')
        if (not os.path.exists(fullpath)): #Check for path
            print phase + ' file does not exsist!'
            sys.exit(1)
        
        if self.verbose: print "Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars()
        self.ncfile = netcdf.Dataset(fullpath, 'r')

        if self.verbose:
            print "dimensions:"
            for dimension_name in self.ncfile.dimensions.keys():
                print "%16s %8d" % (dimension_name, len(self.ncfile.dimensions[dimension_name]))
    
        # Read dimensions.
        self.niterations = self.ncfile.variables['positions'].shape[0]
        self.nstates = self.ncfile.variables['positions'].shape[1]
        self.natoms = self.ncfile.variables['positions'].shape[2]

        # Read reference PDB file.
        if self.phase in ['vacuum', 'solvent']:
            self.reference_pdb_filename = os.path.join(source_directory, "ligand.pdb")
        else:
            self.reference_pdb_filename = os.path.join(source_directory, "complex.pdb")
        self.atoms = self._read_pdb(self.reference_pdb_filename)

        # Check to make sure no self-energies go nan.
        self._check_energies(self.ncfile, self.atoms, verbose=self.verbose)

        # Check to make sure no positions are nan
        self._check_positions(self.ncfile)

        # Get Temperature
        dimtemp = self.ncfile.groups['thermodynamic_states'].variables['temperatures']
        self.temperature = dimtemp[0] * units.kelvin
        self.kT = kB * self.temperature
        self.kcalmolsq = (self.kT / units.kilocalories_per_mole)**2
        self.kcalmol = (self.kT / units.kilocalories_per_mole)

        # Choose number of samples to discard to equilibration
        u_n = self._extract_u_n(self.ncfile, verbose=self.verbose)
        [self.nequil, g_t, Neff_max] = self._detect_equilibration(u_n)
        if self.verbose: print [self.nequil, Neff_max]
        self._build_u_kln()
        if compute_mbar:
            self.compute_mbar()
        else:
            self.mbar_ready = False
        self.expected_done = False
        
        return

##############################################################################################
"""
Class: BasisVariance
Functions and classes needed to analyze basis function variances
"""
class BasisVariance:

    def sequence(self, nc, extra_R, extra_A, lam_out_r = None, lam_out_a = None):
        if lam_out_r is None:
            lam_out_r = self.lam_range
        if lam_out_a is None:
            lam_out_a = self.lam_range
        #Resequence the order to be 0 -> 1, based on work in the plot_multi_lam.py file
        extra_count = len(extra_R)
        #Check the ordering of the extra states
        sign = extra_R[0]-extra_R[-1]
        if sign == 0 or sign < 0: #1D array or increasing
           order='up'
        else:
           order='down'
        #Check ordering on extra (assumes they were build in a monotonic order)
        real_E_indicies = range(nc.real_EAR,nc.real_AR)
        real_R_indicies = range(nc.real_R,nc.real_alloff)
        real_A_indicies = range(nc.real_AR,nc.real_R)
        #Determine if it was done with E-AR or EA-R based on the AR and the R states
        if nc.real_R == nc.nstates-1 or True: #Moded this since this file is depreciated and will never use this logic
            alch_path = 'E-AR'
        else:
            alch_path = 'EA-R' 
        extra_RA_indicies = range(nc.nstates,nc.nstates+extra_count)
    
        if order is 'up': #Checks if they are in increasing order
            extra_RA_indicies = extra_RA_indicies[::-1] #reverse to decreasing order for uniform math
        if lam_out_r[0] < lam_out_r[-1]: #make the lam_out sequence decending too
            lam_out_r = lam_out_r[::-1]
            lam_out_a = lam_out_a[::-1]
    
        sim_RA_looper = 0
        extra_RA_looper = 0
        all_ndx_sorted = numpy.zeros(len(lam_out_r), numpy.int32)
        for i in range(len(lam_out_r)):
            container = numpy.array([numpy.allclose([t,u],[lam_out_r[i],lam_out_a[i]]) for t,u in zip(nc.real_R_states,nc.real_A_states)])
            if not numpy.any(container):
                #If entry not part of the simualted states, grab it from the extra ones
                all_ndx_sorted[i] = extra_RA_indicies[extra_RA_looper]
                extra_RA_looper += 1
            else: #pull in entry from the real ones
                if alch_path is 'E-AR':
                    all_ndx_sorted[i] = int(numpy.array(range(nc.nstates))[numpy.logical_and(container,nc.real_E_states==0)])
                else: #EA-R
                    all_ndx_sorted[i] = int(numpy.array(range(nc.nstates))[container])
        #Reverse sequence to return a 0 -> 1 order
        return all_ndx_sorted[::-1]
    #---------------------------------------------------------------------------------------------

    def buildPerturbedExpected(self, nc, extra_R, extra_A, verbose=None):
        #Build the expected values by the perturbed expectation, do not re-create the mbar object this time
        #This is an incoplete method and needs some work
        if not nc.mbar_ready:
            nc.compute_mbar()
        extra_count = len(extra_R)
        """
        Compute the expectations for the sampled states
        Flesh out the expectations for the sampled states
        For each new state
        Construct its energy matrix
        """
        if not nc.expected_done: #Make it so I only have to calculate this once
            #Build the ua_kln and ur_kln matrix
            ua_kln = numpy.zeros(nc.u_kln.shape, numpy.float64) #Making 2 more expectations to find for the sake of testing optimal path creation
            ur_kln = numpy.zeros(nc.u_kln.shape, numpy.float64)
            nc.const_E_matrix = nc.u_kln[:,nc.real_EAR,:] - nc.u_kln[:,nc.real_AR,:]
            nc.const_R_matrix = nc.u_kln[:,nc.real_R,:] - nc.u_kln[:,nc.real_alloff,:]
            nc.const_A_matrix = nc.u_kln[:,nc.real_AR,:] - nc.u_kln[:,nc.real_R,:]
            for i in range(nc.nstates):
                ur_kln[:,i,:] = nc.const_R_matrix
                ua_kln[:,i,:] = nc.const_A_matrix
            (nc.Eur, nc.dEur) = nc.mbar.computeExpectations(ur_kln)
            (nc.Eua, nc.dEua) = nc.mbar.computeExpectations(ua_kln)
            (nc.Eur2, nc.dEur2) = nc.mbar.computeExpectations(ur_kln**2)
            (nc.Eua2, nc.dEua2) = nc.mbar.computeExpectations(ua_kln**2)
            (nc.Eura, nc.dEura) = nc.mbar.computeExpectations(ur_kln * ua_kln)
            nc.var_ur = nc.Eur2 - nc.Eur**2
            nc.var_ua = nc.Eua2 - nc.Eua**2
            nc.expected_done = True
        #Construct array over which to iterate u_kn_new and A_kn new
        Eur, dEur, Eua, dEua, Eur2, dEur2, Eua2, dEua2, Eura, dEura = (numpy.zeros(nc.nstates+extra_count) for x in range(10))
        Eur[:nc.nstates] = nc.Eur
        Eua[:nc.nstates] = nc.Eua
        Eur2[:nc.nstates] = nc.Eur2
        Eua2[:nc.nstates] = nc.Eua2
        Eura[:nc.nstates] = nc.Eura
        dEur[:nc.nstates] = nc.dEur
        dEua[:nc.nstates] = nc.dEua
        dEur2[:nc.nstates] = nc.dEur2
        dEua2[:nc.nstates] = nc.dEua2
        dEura[:nc.nstates] = nc.dEura
        for i in range(extra_RA_count):
            lamR = extra_R[i]
            lamA = extra_A[i]
            u_kn_i = self.basis.h_r(lamR)*nc.const_R_matrix + self.basis.h_a(lamA)*nc.const_A_matrix + nc.u_kln[:,nc.real_alloff,:]
            (Eur[nc.nstates+i], dEur[nc.nstates+i]) = nc.mbar.computePerturbedExpectation(u_kn_i, nc.const_R_matrix)
            (Eua[nc.nstates+i], dEua[nc.nstates+i]) = nc.mbar.computePerturbedExpectation(u_kn_i, nc.const_A_matrix)
            (Eur2[nc.nstates+i], dEur2[nc.nstates+i]) = nc.mbar.computePerturbedExpectation(u_kn_i, nc.const_R_matrix**2)
            (Eua2[nc.nstates+i], dEua2[nc.nstates+i]) = nc.mbar.computePerturbedExpectation(u_kn_i, nc.const_A_matrix**2)
            (Eura[nc.nstates+i], dEura[nc.nstates+i]) = nc.mbar.computePerturbedExpectation(u_kn_i, nc.const_R_matrix*nc.const_A_matrix)
        var_ur = Eur2 - Eur**2
        var_ua = Eua2 - Eua**2
        dvar_ur = numpy.sqrt(dEur2**2 + 2*(Eur*dEur)**2)
        dvar_ua = numpy.sqrt(dEua2**2 + 2*(Eua*dEua)**2)
        expected = {'Eur':Eur, 'dEur':dEur, 'Eua':Eua, 'dEua':dEua, 'Eur2':Eur2, 'dEur2':dEur2, 'Eua2':Eua2, 'dEua2':dEua2, 'Eura':Eura, 'dEura':dEura, 'var_ur':var_ur, 'dvar_ur':dvar_ur, 'var_ua':var_ua, 'dvar_ua':dvar_ua}
        return expected
    #---------------------------------------------------------------------------------------------

    def buildExpected(self, nc, extra_R, extra_A, verbose=None, return_second=False, bootstrap=False):
        if verbose is None:
            verbose = self.default_verbosity
        #Build the expected values based on the u_kln data and the extra_lambda you want filled in, this is based on my work in analyze-expected.py
        extra_count = len(extra_R)
        #Build the new u_kln and N_k
        u_kln_new = numpy.zeros([nc.nstates + extra_count, nc.nstates + extra_count, nc.retained_iters], numpy.float64)
        N_k_new = numpy.zeros(nc.nstates + extra_count, numpy.int32)
        ua_kln = numpy.zeros(u_kln_new.shape, numpy.float64) #Making 2 more expectations to find for the sake of testing optimal path creation
        ur_kln = numpy.zeros(u_kln_new.shape, numpy.float64)
        #Copy over the original data
        u_kln_new[:nc.nstates,:nc.nstates,:nc.retained_iters] = nc.u_kln
        N_k_new[:nc.nstates] = nc.N_k
        N_samples = u_kln_new.shape[2]
        #Solve for the constant vectors
        const_E_matrix = nc.u_kln[:,nc.real_EAR,:] - nc.u_kln[:,nc.real_AR,:]
        const_R_matrix = nc.u_kln[:,nc.real_R,:] - nc.u_kln[:,nc.real_alloff,:]
        const_A_matrix = nc.u_kln[:,nc.real_AR,:] - nc.u_kln[:,nc.real_R,:]
        #Create the new data
        for i in range(extra_count):
            lamR = extra_R[i]
            lamA = extra_A[i]
            u_kln_new[:nc.nstates,i+nc.nstates,:] = self.basis.h_r(lamR)*const_R_matrix + self.basis.h_a(lamA)*const_A_matrix + nc.u_kln[:,nc.real_alloff,:]
        for i in range(extra_count+nc.nstates):
            ur_kln[:nc.nstates,i,:] = const_R_matrix
            ua_kln[:nc.nstates,i,:] = const_A_matrix
        #Shuffle all the states if bootstrap is on
        if bootstrap:
            u_kln_boot = numpy.zeros(u_kln_new.shape)
            ua_kln_boot = numpy.zeros(ua_kln.shape, numpy.float64) #Making 2 more expectations to find for the sake of testing optimal path creation
            ur_kln_boot = numpy.zeros(ur_kln.shape, numpy.float64)
            for state in range(u_kln_boot.shape[0]):
                samplepool = random_integers(0,N_samples-1,N_samples) #Pull the indicies for the sample space, N number of times
                for i in xrange(len(samplepool)): #Had to put this in its own loop as u_kln_new[state,:,samplepool] was returning a NxK matrix instead of a KxN
                    u_kln_boot[state,:,i] = u_kln_new[state,:,samplepool[i]]
                    ur_kln_boot[state,:,i] = ur_kln[state,:,samplepool[i]]
                    ua_kln_boot[state,:,i] = ua_kln[state,:,samplepool[i]]
            #Copy over shuffled data
            u_kln_new = u_kln_boot
            ur_kln = ur_kln_boot
            ua_kln = ua_kln_boot
        #Run MBAR
        mbar = MBAR(u_kln_new, N_k_new, verbose = verbose, method = 'adaptive')
        (Eur, dEur) = mbar.computeExpectations(ur_kln)
        (Eua, dEua) = mbar.computeExpectations(ua_kln)
        (Eur2, dEur2) = mbar.computeExpectations(ur_kln**2)
        (Eua2, dEua2) = mbar.computeExpectations(ua_kln**2)
        (Eura, dEura) = mbar.computeExpectations(ur_kln * ua_kln)
        var_ur = Eur2 - Eur**2
        dvar_ur = numpy.sqrt(dEur2**2 + 2*(Eur*dEur)**2)
        var_ua = Eua2 - Eua**2
        dvar_ua = numpy.sqrt(dEua2**2 + 2*(Eua*dEua)**2)
        #Construct dictionary
        expected_values = {'Eur':Eur, 'Eua':Eua, 'Eura':Eura, 'var_ur':var_ur, 'var_ua':var_ua, 'dEur':dEur, 'dEua':dEua, 'dEura':dEura, 'dvar_ur':dvar_ur, 'dvar_ua':dvar_ua}
        if return_second:
            (EU, dEU) = mbar.computeExpectations(ur_kln + ua_kln)
            expected_values['EU'] = EU
            expected_values['dEU'] = dEU
        return expected_values
    #---------------------------------------------------------------------------------------------

    def calcdhdl(self, expected, basis, lam_r, lam_a, return_error=False):
        #calculate the variance directly
        dr = basis.dh_r(lam_r)
        da = basis.dh_a(lam_a)
        if numpy.all(lam_a == 0): #Check for if lam_a is changing at all
            da = numpy.zeros(len(dr))
    
        #Calculate dhdl = <dH/dL> = h' \dot <u>
        dhdl_calc = dr*expected['Eur'] + da*expected['Eua']
        dhdl_err  = dr*expected['dEur'] + da*expected['dEua']
        dhdl_dict = {'natural':dhdl_calc}
        if return_error:
            dhdl_dict['plus'] = dhdl_calc + dhdl_err
            dhdl_dict['minus'] = dhdl_calc - dhdl_err
        return dhdl_dict
        
    def calcsecond(self, EU, dEU, lam_range, return_error=False):
        #Special case with a singe lam schedule and pure linear alchemical switch, the math works here
        del_space=lam_range[1]-lam_range[0]
        if return_error:
            to_integrate = {'plus':EU+dEU, 'minus':EU-dEU, 'natural':EU}
        else:
            to_integrate = {'natural':EU}
        integrand = {}
        variance = {}
        for key in to_integrate:
            integrand[key] = numpy.zeros(len(lam_range))
            for i in range(len(lam_range)):
                if i <=2: #Forward fintie differences, 2nd order
                    integrand[key][i] = - ( -3*to_integrate[key][i] + 4*to_integrate[key][i+1] - to_integrate[key][i+2] )/(2*del_space)
                elif i >= len(lam_range)-3: #Backwards finite differences, 2nd order
                    integrand[key][i]= - ( 3*to_integrate[key][i] - 4*to_integrate[key][i-1] + to_integrate[key][i-2] )/(2*del_space)
                else: #Central finite differences, 3rd order
                    integrand[key][i] = - ( to_integrate[key][i-2] - 8*to_integrate[key][i-1] + 8*to_integrate[key][i+1] - to_integrate[key][i+2] )/(12*del_space)
            variance[key] = simps(integrand[key],lam_range)
        return integrand,variance

    def calcvar(self, expected, basis, lam_r, lam_a, return_error=False):
        #calculate the variance directly
        dr = basis.dh_r(lam_r)
        da = basis.dh_a(lam_a)
        if numpy.all(lam_a == 0): #Check for if lam_a is changing at all
            da = numpy.zeros(len(dr))
    
        if return_error:
            to_integrate = {'plus': {
                'var_ur':expected['var_ur'] + expected['dvar_ur'], 
                'var_ua':expected['var_ua'] + expected['dvar_ua'],
                'Eura':  expected['Eura']   + expected['dEura'],
                'Eur':   expected['Eur']    + expected['dEur'],
                'Eua':   expected['Eua']    + expected['dEua']},
                          'minus':{
                'var_ur':expected['var_ur'] - expected['dvar_ur'], 
                'var_ua':expected['var_ua'] - expected['dvar_ua'],
                'Eura':  expected['Eura']   - expected['dEura'],
                'Eur':   expected['Eur']    - expected['dEur'],
                'Eua':   expected['Eua']    - expected['dEua']},
                          'natural':{
                'var_ur':expected['var_ur'], 
                'var_ua':expected['var_ua'],
                'Eura':  expected['Eura'],
                'Eur':   expected['Eur'],
                'Eua':   expected['Eua']}
            }
        else:
            to_integrate = {'natural':expected}
        #Calculate the variance
        integrand = {}
        variance = {}
        for key in to_integrate:
            r_only_term = dr**2 * to_integrate[key]['var_ur']
            a_only_term = da**2 * to_integrate[key]['var_ua']
            cross_term = 2* dr*da * (to_integrate[key]['Eura'] - to_integrate[key]['Eur']*to_integrate[key]['Eua'])
            integrand[key] = r_only_term + a_only_term + cross_term
            #This will need modified to work in other alchemical schedules
            variance[key] = simps(integrand[key],lam_r)
        return integrand,variance
    #---------------------------------------------------------------------------------------------
    def vargenerate(self, verbose=None, lam_in_r=None, lam_in_a=None, calculate_var=True, calculatedhdl=False, expect_method = 'complete', return_error=False, bootstrap_error=False, bootstrap_count=200):
        if verbose is None:
            verbose=self.default_verbosity
        if calculatedhdl and not calculate_var:
            print "Warning: dHdL calculation requires variance calculation"
            print "Turning on variance calculation"
            calculate_var=True
        if bootstrap_error and return_error:
            print "Warning: Normal error and bootstrap error are incompatable, using bootstrap only"
            return_error=False
        if lam_in_r is None:
            xr = self.lam_range
        else:
            xr = lam_in_r
        if lam_in_a is None:
            xa = self.lam_range
        else:
            xa = lam_in_a
        #Calculate the variance of the original basis set
        if not xr.shape == xa.shape:
            print 'Lambda R and Lambda A list not the same size!'
            sys.exit(1)
        x = numpy.array([xr,xa])
        extra_R_list = numpy.empty(0)
        extra_A_list = numpy.empty(0)
        for i,j in zip(xr,xa):
            if not numpy.any([numpy.allclose([t,u],[i,j]) for t,u in zip(self.complex.real_R_states,self.complex.real_A_states)]) :
                extra_R_list = numpy.append(extra_R_list,i)
                extra_A_list = numpy.append(extra_A_list,j)
        extra_RA_states = numpy.array([extra_R_list,extra_A_list])
        #Disabling this for now
        #if expect_method is 'perturbed':
        #    (Eur,Eua,Eura,var_ur,var_ua) = buildPerturbedExpected(comp, basis, extra_R_list, extra_A_list, verbose)
        #elif expect_method is 'weighted':
        #    (Eur,Eua,Eura,var_ur,var_ua) = buildWeightedExpected(comp, basis, extra_R_list, extra_A_list, verbose)
        if expect_method is 'second':
            expectations = self.buildExpected(self.complex, extra_R_list, extra_A_list, verbose=verbose, return_second=True)
        else:
            #Fully calculate the expectation matrix
            expectations = self.buildExpected(self.complex, extra_R_list, extra_A_list, verbose=verbose)
        sorted_ndx = self.sequence(self.complex, extra_R_list, extra_A_list, lam_out_r=xr,lam_out_a=xa)
        #Filter out only the RA values I need
        for key in expectations.keys():
            expectations[key] = expectations[key][sorted_ndx]
        lam_range = x.copy()
    
        if calculate_var: #The following statment must be true (for now): lam_range[0,:] == lam_range[1,:]
            if expect_method is 'second':
                integrand,variance = self.calcsecond(expectations['EU'], expectations['dEU'], xr, return_error=return_error)
            else:
                integrand,variance = self.calcvar(expectations, self.basis, xr, xa, return_error=return_error)
            if calculatedhdl:
                dhdl = self.calcdhdl(expectations, self.basis, xr, xa, return_error=return_error)
            #If bootstrap is on, run it
            if bootstrap_error:
                #Deterimine shape of output matrix [lr,bootstrap_count]
                bootstrap_integrands = numpy.zeros([len(xr),bootstrap_count])
                bootstrap_dhdl = numpy.zeros([len(xr),bootstrap_count])
                bootstrap_error = numpy.zeros([len(xr)])
                for i in xrange(bootstrap_count):
                    print "Bootstrap pass: %d / %d" % (i+1,bootstrap_count)
                    if expect_method is 'second':
                        #Special case for pure linear transformation
                        boot_expect = self.buildExpected(self.complex, extra_R_list, extra_A_list, verbose=verbose, bootstrap=True, return_second=True)
                    else:
                        #Normal case
                        boot_expect = self.buildExpected(self.complex, extra_R_list, extra_A_list, verbose=verbose, bootstrap=True)

                    for key in boot_expect.keys():
                        boot_expect[key] = boot_expect[key][sorted_ndx]

                    if expect_method is 'second':
                        boot_integrand_holder, boot_variance_junk = self.calcsecond(boot_expect['EU'], boot_expect['dEU'], xr, return_error=False)
                    else:
                        boot_integrand_holder, boot_variance_junk = self.calcvar(boot_expect, self.basis, xr, xa, return_error=False)
                    if calculatedhdl:
                        boot_dhdl_holder = self.calcdhdl(boot_expect, self.basis, xr, xa, return_error=False)
                        bootstrap_dhdl[:,i] = boot_dhdl_holder['natural']
                    bootstrap_integrands[:,i] = boot_integrand_holder['natural']
                #Calculate variance of the collection
                bootstrap_error[:] = numpy.sqrt(numpy.var(bootstrap_integrands[:,:], axis=1))
                integrand['plus'] = integrand['natural'] + bootstrap_error
                integrand['minus'] = integrand['natural'] - bootstrap_error
                integrand['bootstrap_error'] = bootstrap_error
                if calculatedhdl:
                    bootstrap_dhdl_error = numpy.sqrt(numpy.var(bootstrap_integrands[:,:], axis=1))
                    dhdl['plus'] = dhdl['natural'] + bootstrap_dhdl_error
                    dhdl['minus'] = dhdl['natural'] - bootstrap_dhdl_error
            if calculatedhdl:
                return integrand,variance,dhdl
            else:
                return integrand,variance
        else:
            return expectations
    #---------------------------------------------------------------------------------------------

    def inv_var(self, being_predicted_basis, predicted_lam_r, predicted_lam_a, return_error=False, calculatedhdl=False, verbose=None, bootstrap_error=False, bootstrap_count=200):
        if verbose is None:
            verbose=self.default_verbosity
        #Reconstruct the lambda_predicted to make a new set to pass to the source data and see if we can predict variance
        #For notation, the prediction will be "g" and the source will be "h"
        #Generally, dg/dl_g <u>_{l_g} \neq dh/dl_h <u>_{l_h}... but, since g and h explore the same domain of [0,1], <u>_g(g) = <u>_h(h=g)
        #We can then write <u>_{l_g}(l_g) = <u>_{l_h}(l_h = h_inv(g(l_g)))
      
        #Generate the lam_h to sample from
        lam_source_effective_r = self.basis.h_r_inv(being_predicted_basis.h_r(predicted_lam_r))
        lam_source_effective_a = self.basis.h_a_inv(being_predicted_basis.h_a(predicted_lam_a))
        expectations = self.vargenerate(verbose=verbose, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, calculate_var=False, return_error=return_error)
        pre_integrand, pre_variance = self.calcvar(expectations, being_predicted_basis, predicted_lam_r, predicted_lam_a, return_error=return_error)
        if bootstrap_error:
            boot_integrands, boot_var = self.vargenerate(verbose=verbose, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, calculate_var=True, return_error=False, bootstrap_error=True, bootstrap_count=bootstrap_count)
            pre_integrand['plus'] = pre_integrand['natural'] + boot_integrands['bootstrap_error']
            pre_integrand['minus'] = pre_integrand['natural'] - boot_integrands['bootstrap_error']
        if calculatedhdl:
            pre_dhdl = self.calcdhdl(expectations, being_predicted_basis, predicted_lam_r, predicted_lam_a, return_error=return_error)
            return pre_integrand, pre_variance, pre_dhdl
        else:
            return pre_integrand,pre_variance

    #---------------------------------------------------------------------------------------------

    def _unequalFinite(self, order, u_kln, points):
        num_points = len(points)
        if num_points <= order:
            print "Order of derivative must be less than points available"
            sys.exit(1)
        
    def varSC(self, verbose=None, bootstrap_error=False, bootstrap_count=200,return_sampled_points=False):
        #Custom function to numerically estiamte Variance of SC potentials, cannot do arbitrary construction though, this method has high error
        if verbose is None:
            verbose=self.default_verbosity
        #Build the MBAR object
        u_kln = self.complex.u_kln
        mbar = MBAR(u_kln, self.complex.N_k, verbose = verbose, method = 'adaptive')
        #Calculate the dudl and the du2dl2
        dudl_kln = numpy.zeros(u_kln.shape)
        du2dl2_kln = numpy.zeros(u_kln.shape)
        E_count = abs(self.complex.real_EAR - self.complex.real_AR) + 1 #Fuly coupled state-> E off, + 1 to account for number of states
        RA_count = abs(self.complex.real_AR - self.complex.real_alloff) + 1 #Only AR -> fully decouled, + 1 to account for number of states
        RA_lam = self.complex.real_R_states[self.complex.real_AR:self.complex.real_alloff+1]
        #self.real_R_states Electrostatics first, Assume linear transformation (can be corrected later)
        for i in range(E_count):
            dudl_kln[:,i,:] = u_kln[:,i,:]
            #du2dl2_kln[:,i,:] = numpy.zeros(du2dl2_kln[:,i,:].shape) #Since assumed linear, and already zeros, then this is already set
        #Repulsive attractive derivtavies
        import pdb
        for k in range(RA_count):
            print "state: %i" % k
            for n in range(self.complex.retained_iters):
                splined = US(RA_lam[::-1], u_kln[k,self.complex.real_AR:self.complex.real_alloff+1,n][::-1])
                if numpy.any(numpy.isnan(splined(RA_lam,1))):
                    pdb.set_trace()
                dudl_kln[k,self.complex.real_AR:self.complex.real_alloff+1,n] = splined(RA_lam,1)
                du2dl2_kln[k,self.complex.real_AR:self.complex.real_alloff+1,n] = splined(RA_lam,2)
        #Unfortunatley, the expectations for the R only state do not make sense
        #Compute expectations
        (Edu, dEdu) = mbar.computeExpectations(dudl_kln)
        (Edu2, dEdu2) = mbar.computeExpectations(du2dl2_kln)
        expectations = {'Edu':Edu, 'dEdu':dEdu, 'Edu2':Edu2, 'dEdu2':dEdu2}
        ordered_ndx = self.sequence(self.complex, RA_lam, RA_lam, lam_out_r = RA_lam, lam_out_a = RA_lam)
        for key in expectations.keys():
            expectations[key] = expectations[key][ordered_ndx]
        varspline = US(RA_lam[::-1], expectations['Edu'])
        varpoints={}
        varpoints['natural'] = self.complex.kcalmol*(expectations['Edu2']-varspline(RA_lam[::-1],1))
        variance = simps(varpoints['natural'], RA_lam[::-1])
        if return_sampled_points:
            varpoints['lam_range'] = RA_lam[::-1]
        return varpoints, variance

    #---------------------------------------------------------------------------------------------
    def grid_search(self,verbose=False,stepsize=0.001):
        #Incomplete function
        #This is a general grid based search algorithm, it will construct a path through your phase space to generate an optimal sampling point
        #A small perturbation in lambda is chosen and advanced. 
        if verbose is None:
            verbose=self.default_verbosity
        #Create the empty spline starting at 0,0
        optilam = numpy.zeros([2,1],dtype=float64)
        #Start loop
        while not numpy.all(optilam[:,-1] == numpy.ones([2,1],dtype=float)):
            print "Current Point: LR=%.3f, LA=%.3f" % (optilam[0,-1], optilam[1,-1])
            #Build direction list
            Rstep = optilam[0,-1]
            Astep = optilam[1,-1]
            #R,A,RA
            extraR = numpy.array([Rstep,optilam[0,-1],Rstep])
            extraA = numpy.array([optilam[1,-1],Astep,Astep])
            #Build the expectations
            expectations = self.buildPerturbedExpected(self.complex, extraR, extraA, verbose=verbose)
            #Calculate the variance in all directions, assume linear slope

    #---------------------------------------------------------------------------------------------

    def __init__(self,source_basis,source_comp,source_vac,verbose=False,SC=False):
        #Check if the source information are the correct classes
        incorrect_input = False
        if not isinstance(source_basis,LinFunctions) and not SC:
            print "Basis function selection not a class of 'LinFunctions'"
            incorrect_input = True
        if not isinstance(source_comp,ncdata):
            print "Complex Data selection not a class of 'ncdata'"
            incorrect_input = True
        if not isinstance(source_vac,ncdata):
            print "Vacuum Data selection not a class of 'ncdata'"
            incorrect_input = True
        if incorrect_input:
            print "Incorrect BasisVariance Input!"
            sys.exit(1)
        #If its all correct, assign variables
        if not SC:
            self.basis = source_basis
        self.complex = source_comp
        self.vacuum = source_vac
        #Set default lam_range
        self.lam_range = scipy.linspace(0.0,1.0,101)
        self.default_verbosity = verbose

        return

if __name__ == '__main__':
    print "Eyyyup"
