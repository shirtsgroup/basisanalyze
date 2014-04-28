"""
Import all of the basis analyze tools

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

from basisanalyze.linfunctions import *
from basisanalyze.ncdata import *
from basisanalyze.basisvariance import *
