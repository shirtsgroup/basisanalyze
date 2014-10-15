"""
Class: BasisVariance
Functions and classes needed to analyze basis function variances

TODO:
-Create a way to handle any number of basis functions.
"""

import scipy
import numpy
from scipy.integrate import simps
import sys
from numpy.random import random_integers
from pymbar import MBAR # multistate Bennett acceptance ratio

class statelist():
    def reverse_order(self):
        self.E_states = self.E_states[::-1]
        self.R_states = self.R_states[::-1]
        self.A_states = self.A_states[::-1]
        if self.order == 'up':
            self.order ='down'
        else:
            self.order ='up'

    def __init__(self,Elist,Rlist,Alist):
        self.E_states=Elist
        self.R_states=Rlist
        self.A_states=Alist       
        self.nstates = self.E_states.size
        self.states_list = [self.E_states, self.R_states, self.A_states]
        if len(Elist) == 0:
            self.order = 'down' #this makes no sense, but i just needed a void
        else:
            sign = Elist[0]-Elist[-1]
            if sign ==0 or sign < 0:
                self.order='up'
            else:
                self.order='down'

class BasisVariance:
        
    def sequence_master(self, nc, extra_states, lam_out=None):
        if lam_out is None:
            lam_out = statelist(self.lam_range, self.lam_range, self.lam_range)
        extra_count = extra_states.nstates
        extra_indicies = range(nc.nstates,nc.nstates+extra_count)
        if extra_states.order is 'up': #reverse order if need be
            extra_indicies = extra_indicies[::-1]
        if lam_out.order is 'up': #make the lam_out sequence decending too
            lam_out.reverse_order()
            revert_lam_out = True #If lam_out was altered, switch it back to the original before sending it back
        else:
            revert_lam_out = False
        sim_looper = 0
        extra_looper = 0
        all_ndx_sorted = numpy.zeros(lam_out.nstates, numpy.int32)
        #Correct for inversion of charge
        if nc.Inversion:
            sampledE = self.basis.h_e_inv(nc.real_E_states)
        else:
            sampledE = nc.real_E_states
        sampledR = nc.real_R_states
        sampledA = nc.real_A_states
        for i in range(lam_out.nstates):
            #Pick out lambdas
            lamE = lam_out.E_states[i]
            lamR = lam_out.R_states[i]
            lamA = lam_out.A_states[i]
            #Generate logic container to determine of the state described by i was already sampled, the allclose is to adjust for small differences between t,lamE; u,lamR; v,lamA.
            container = numpy.array([numpy.allclose([t,u,v],[lamE,lamR,lamA]) for t,u,v in zip(sampledE,sampledR,sampledA)])
            if not numpy.any(container):
                #If entry not part of the simualted states, grab it from the extra ones
                all_ndx_sorted[i] = extra_indicies[extra_looper]
                extra_looper += 1
            else: #pull in entry from the real ones, based on where the logic container says it is AND where PMEFull=E, prevents getting the PMEsolve states
                all_ndx_sorted[i] = int(numpy.array(range(nc.nstates))[numpy.logical_and(container,nc.real_PMEFull_states==nc.real_E_states)])
        #Reverse sequence to return a 0 -> 1 order
        if revert_lam_out:
            lam_out.reverse_order()
        return all_ndx_sorted[::-1]

    def Ugen_EP_C_AR(self, nc, raw=False):
        if raw:
            u_kln = nc.u_kln_raw
        else:
            u_kln = nc.u_kln
        const_R_matrix = u_kln[:,nc.real_R,:] - u_kln[:,nc.real_alloff,:]
        const_A_matrix = u_kln[:,nc.real_AR,:] - u_kln[:,nc.real_R,:]
        const_C_matrix = u_kln[:,nc.real_CAR,:] - u_kln[:,nc.real_AR,:]
        const_E_matrix = u_kln[:,nc.real_EPCAR,:] - u_kln[:,nc.real_PCAR,:]
        return const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix
    def Ugen_EPA_C_R(self, nc, raw=False):
        if raw:
            u_kln = nc.u_kln_raw
        else:
            u_kln = nc.u_kln
        const_R_matrix = u_kln[:,nc.real_R,:] - u_kln[:,nc.real_alloff,:]
        const_C_matrix = u_kln[:,nc.real_CR,:] - u_kln[:,nc.real_R,:]
        const_A_matrix = u_kln[:,nc.real_CAR,:] - u_kln[:,nc.real_CR,:]
        const_E_matrix = u_kln[:,nc.real_EPCAR,:] - u_kln[:,nc.real_PCAR,:]
        return const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix
    def Ugen_EP_A_C_R(self, nc, raw=False):
        if raw:
            u_kln = nc.u_kln_raw
        else:
            u_kln = nc.u_kln
        const_R_matrix = u_kln[:,nc.real_R,:] - u_kln[:,nc.real_alloff,:]
        const_C_matrix = u_kln[:,nc.real_CR,:] - u_kln[:,nc.real_R,:]
        const_A_matrix = u_kln[:,nc.real_CAR,:] - u_kln[:,nc.real_CR,:]
        const_E_matrix = u_kln[:,nc.real_EPCAR,:] - u_kln[:,nc.real_PCAR,:]
        return const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix
    def Ugen_A_EP_C_R(self, nc, raw=False):
        if raw:
            u_kln = nc.u_kln_raw
        else:
            u_kln = nc.u_kln
        const_R_matrix = u_kln[:,nc.real_R,:] - u_kln[:,nc.real_alloff,:]
        const_C_matrix = u_kln[:,nc.real_CR,:] - u_kln[:,nc.real_R,:]
        const_E_matrix = u_kln[:,nc.real_EPCR,:] - u_kln[:,nc.real_PCR,:]
        const_A_matrix = u_kln[:,nc.real_EPCAR,:] - u_kln[:,nc.real_EPCR,:]
        return const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix

    def checkSequence(self, seq=None):
        valid_seqs = [ ['EP', 'C', 'AR'], ['EPA', 'C', 'R'], ['EP', 'A', 'C', 'R'], ['A', 'EP', 'C', 'R'] ]
        if seq is None:
            return valid_seqs
        else:
            if seq not in valid_seqs:
                print "Only list of valid sequnces are:"
                print valid_seqs
                sys.exit(1)
            else:
                return

    def genConsts(self, nc, sequence, derU=False, raw=False):
        #Given a sequence, automatically pass in the correct const matricies
        #This is a quality of life function.
        self.checkSequence(seq=sequence)
        valid_seqs = self.checkSequence()
        #Compute PME
        if raw:
            u_kln=nc.u_kln_raw
        else:
            u_kln=nc.u_kln
        if nc.real_PCAR is not None: #Trap when A is decoupled first
            noEstate = nc.real_CAR
            Pstate = nc.real_PCAR
            Psolve = nc.real_Psolve
        else:
            noEstate = nc.real_CR
            Pstate = nc.real_PCR
            Psolve = nc.real_Psolve2
        PMEFull = u_kln[:,Pstate,:] - u_kln[:,noEstate,:]
        PMELess = u_kln[:,Psolve,:] - u_kln[:,noEstate,:]
        LamAtFull = nc.real_PMEFull_states[Pstate]
        LamAtLess = nc.real_PMEFull_states[Psolve]
        hless = self.basis.h_e(LamAtLess)
        hfull = self.basis.h_e(LamAtFull)
        const_Psq_matrix = (PMELess/hless - PMEFull/hfull) / (hless-hfull)
        const_P_matrix = PMEFull/hfull - hfull*const_Psq_matrix
        #Compute the rest of the basis based on sequence
        if derU is True:
            #Set the functions of the derivatives if the derivatives are requested
            f_e = lambda lam: self.basis.dh_e(lam)
            f_psq = lambda lam: 2*self.basis.h_e(lam)*self.basis.dh_e(lam)
            f_r = lambda lam: self.basis.dh_r(lam)
            f_a = lambda lam: self.basis.dh_a(lam)
            f_c = lambda lam: 1
            un_factor = 0
        else:
            #Use normal calculations
            f_e = lambda lam: self.basis.h_e(lam)
            f_psq = lambda lam: self.basis.h_e(lam)**2
            f_r = lambda lam: self.basis.h_r(lam)
            f_a = lambda lam: self.basis.h_a(lam)
            f_c = lambda lam: lam
            un_factor = 1
        if sequence == valid_seqs[0]: #ep c ar
            const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EP_C_AR(nc, raw=raw)
            #Assign energy evaluation stages
            Ustage = [
                lambda lam: f_e(lam)*const_E_matrix + \
                            f_psq(lam)*const_Psq_matrix + \
                            f_e(lam)*const_P_matrix + \
                            un_factor*u_kln[:,nc.real_CAR,:],
                lambda lam: f_c(lam) * const_C_matrix + \
                            un_factor*u_kln[:,nc.real_AR,:],
                lambda lam: f_r(lam)*const_R_matrix + \
                            f_a(lam)*const_A_matrix + \
                            un_factor*u_kln[:,nc.real_alloff,:]
                     ]
        elif sequence == valid_seqs[1]: #epa, c, r
            const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EPA_C_R(nc, raw=raw)
            Ustage = [
                lambda lam: f_e(lam)*const_E_matrix + \
                            f_psq(lam)*const_Psq_matrix + \
                            f_e(lam)*const_P_matrix + \
                            f_a(lam)*const_A_matrix + \
                            un_factor*u_kln[:,nc.real_CR,:],
                lambda lam: f_c(lam) * const_C_matrix + \
                            un_factor * u_kln[:,nc.real_R,:],
                lambda lam: f_r(lam)*const_R_matrix + \
                            un_factor*u_kln[:,nc.real_alloff,:]
                     ]
        elif sequence == valid_seqs[2]: #ep, a, c, r
            const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EP_A_C_R(nc, raw=raw)
            Ustage = [
                lambda lam: f_e(lam)*const_E_matrix + \
                            f_psq(lam)*const_Psq_matrix + \
                            f_e(lam)*const_P_matrix + \
                            un_factor*u_kln[:,nc.real_CAR,:],
                lambda lam: f_a(lam)*const_A_matrix + \
                            un_factor*u_kln[:,nc.real_CR,:],
                lambda lam: f_c(lam) * const_C_matrix + \
                            un_factor*u_kln[:,nc.real_R,:],
                lambda lam: f_r(lam)*const_R_matrix + \
                            un_factor*u_kln[:,nc.real_alloff,:]
                     ]
        elif sequence == valid_seqs[3]: #a, ep, c, r
            const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_A_EP_C_R(nc, raw=raw)
            Ustage = [
                lambda lam: f_a(lam)*const_A_matrix + \
                            un_factor*u_kln[:,nc.real_EPCR,:],
                lambda lam: f_e(lam)*const_E_matrix + \
                            f_psq(lam)*const_Psq_matrix + \
                            f_e(lam)*const_P_matrix + \
                            un_factor*u_kln[:,nc.real_CR,:],
                lambda lam: f_c(lam) * const_C_matrix + \
                            un_factor*u_kln[:,nc.real_R,:],
                lambda lam: f_r(lam)*const_R_matrix + \
                            un_factor*u_kln[:,nc.real_alloff,:]
                     ]
        else: #This sould not trip, added as safety
            print "How did you make it here?"
            sys.exit(1)
        return const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix, const_P_matrix, const_Psq_matrix, Ustage

    def buildExpected_master(self, nc, extra_lam, sequence, verbose=None, bootstrap=False, basis_derivatives=None, single_stage=None, fragment_kln=False, stage_ranges=None):
        """
        This particular bit of coding will be reworked in this branch to make sure that all of the possible scheudles can be handled in a general maner with minimal user input.
        Unfortunatley, this will require reading in both the alchemy file and user input information.
        I have removed the perturbed expected from this since it is no longer used
        """
        if verbose is None:
            verbose=self.default_verbosity
        if basis_derivatives is None:
            basis_derivatives = self.basis
        #Validate the sequence
        valid_seqs = self.checkSequence()
        nstage = len(sequence)
        if len(extra_lam) != nstage:
            print "Incorrect number of extra_lam arguments. There must be at least one item per stage, even if its an empty object"
            sys.exit(1)
        #Compute the constant matricies
        if verbose: print "Generating constant matricies..."
        const_R_matrix, const_A_matrix, const_C_matrix, const_E_matrix, const_P_matrix, const_Psq_matrix, Ustage = self.genConsts(nc, sequence)
 
        #Compute the potential energies
        u_klns = {}
        N_ks = {}
        individualU_klns = {}
        expected_values = {}
        #Check if we are generating a single stage (Useful for single analysis)
        if single_stage is not None:
            stage_generator = [ single_stage ]
        else:
            stage_generator = xrange(nstage)
        for stage_index in stage_generator:
            stage_name = sequence[stage_index]
            xl_stage = extra_lam[stage_name]
            nxl = len(xl_stage)
            if verbose: print "Generating Expectations for stage " + stage_name
            if fragment_kln:
                nstate_analyze = len(stage_ranges[stage_name])
                active_states = stage_ranges[stage_name]
            else:
                nstate_analyze = nc.nstates
                active_states = numpy.array(range(nc.nstates))
            individualU_klns[stage_name] = {}
            #Preallocate all the energy memory
            u_klns[stage_name] = numpy.zeros([nstate_analyze + nxl, nstate_analyze + nxl, nc.retained_iters], numpy.float64)
            N_samples = nc.retained_iters
            N_ks[stage_name] = numpy.zeros(nstate_analyze + nxl, numpy.int32)
            #Copy over original data
            if fragment_kln:
                #Loop over each state individually, inefficent, but multislicing does not work correctly
                for k in xrange(nstate_analyze):
                    for l in xrange(nstate_analyze):
                        u_klns[stage_name][k,l,:nc.retained_iters] = nc.u_kln[active_states[k],active_states[l],:]
            else:
                #Original, quick way
                u_klns[stage_name][:nstate_analyze,:nstate_analyze,:nc.retained_iters] = nc.u_kln
            N_ks[stage_name][:nstate_analyze] = nc.N_k[active_states]
            #Determine the set of changing basis functions
            nchanging = len(sequence[stage_index])
            #Generate basis function containers, run through each one
            for label in sequence[stage_index]:
                if label == "P":
                    individualU_klns[stage_name]["P"] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
                    individualU_klns[stage_name]["Psq"] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
                else:
                    individualU_klns[stage_name][label] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
            #Compute energies
            for i in xrange(nxl):
                lam = xl_stage[i]
                U = Ustage[stage_index](lam)
                if fragment_kln:
                    for k in xrange(len(active_states)):
                        u_klns[stage_name][k,i+nstate_analyze,:] = U[active_states[k],:]
                else:
                    u_klns[stage_name][:nstate_analyze,i+nstate_analyze,:] = U
            for i in xrange(nstate_analyze + nxl):
                for label in sequence[stage_index]:
                    if fragment_kln:
                        for k in xrange(nstate_analyze):
                            if   label == "E":
                                individualU_klns[stage_name][label][k,i,:] = const_E_matrix[k,:]
                            elif label == "P":
                                individualU_klns[stage_name][label][k,i,:] = const_P_matrix[k,:]
                                individualU_klns[stage_name]["Psq"][k,i,:] = const_Psq_matrix[k,:]
                            elif label == "C":
                                individualU_klns[stage_name][label][k,i,:] = const_C_matrix[k,:]
                            elif label == "A":
                                individualU_klns[stage_name][label][k,i,:] = const_A_matrix[k,:]
                            elif label == "R":
                                individualU_klns[stage_name][label][k,i,:] = const_R_matrix[k,:]
                            else:
                                print "Seriously, how did you make it here?"
                                sys.exit(1)
                    else:
                        if   label == "E":
                            individualU_klns[stage_name][label][:nstate_analyze,i,:] = const_E_matrix
                        elif label == "P":
                            individualU_klns[stage_name][label][:nstate_analyze,i,:] = const_P_matrix
                            individualU_klns[stage_name]["Psq"][:nstate_analyze,i,:] = const_Psq_matrix
                        elif label == "C":
                            individualU_klns[stage_name][label][:nstate_analyze,i,:] = const_C_matrix
                        elif label == "A":
                            individualU_klns[stage_name][label][:nstate_analyze,i,:] = const_A_matrix
                        elif label == "R":
                            individualU_klns[stage_name][label][:nstate_analyze,i,:] = const_R_matrix
                        else:
                            print "Seriously, how did you make it here?"
                            sys.exit(1)
            if bootstrap:
                u_kln_boot = numpy.zeros(u_klns[stage_name].shape)
                individualU_klns_boot = {}
                for label in individualU_klns[stage_name].keys():
                    individualU_klns_boot[label] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
                for state in xrange(u_kln_boot.shape[0]):
                    samplepool = random_integers(0,N_samples-1,N_samples) #Pull the indicies for the sample space, N number of times
                    for i in xrange(len(samplepool)): #Had to put this in its own loop as u_klns[stage_name][state,:,samplepool] was returning a NxK matrix instead of a KxN
                        u_kln_boot[state,:,i] = u_klns[stage_name][state,:,samplepool[i]]
                        for label in individualU_klns[stage_name].keys():
                            individualU_klns_boot[label][state,:,i] = individualU_klns[stage_name][label][state,:,samplepool[i]]
                #Copy over shuffled data
                u_klns[stage_name] = u_kln_boot
                for label in individualU_klns[stage_name].keys():
                    individualU_klns[stage_name][label] = individualU_klns_boot[label]
            basis_labels = individualU_klns[stage_name].keys()
            Nbasis = len(basis_labels)
            #Prep MBAR
            if nc.mbar_ready:
                mbar = MBAR(u_klns[stage_name], N_ks[stage_name], verbose = False, method = 'adaptive', initial_f_k=numpy.concatenate((nc.mbar.f_k[active_states],numpy.zeros(nxl))))
            else:
                mbar = MBAR(u_klns[stage_name], N_ks[stage_name], verbose = verbose, method = 'adaptive')
            expected_values[stage_name] = {'labels':basis_labels, 'Nbasis':Nbasis}
            for label in basis_labels:
                if   label == "E":
                   expected_values[stage_name]["dswitch" + label] = basis_derivatives.dh_e
                elif label == "P":
                    expected_values[stage_name]["dswitch" + label] = basis_derivatives.dh_e
                elif label == "Psq":
                    expected_values[stage_name]["dswitch" + label] = lambda X: 2*basis_derivatives.h_e(X)*basis_derivatives.dh_e(X)
                elif label == "C":
                    expected_values[stage_name]["dswitch" + label] = lambda X: X
                elif label == "A":
                    expected_values[stage_name]["dswitch" + label] = basis_derivatives.dh_a
                elif label == "R":
                    expected_values[stage_name]["dswitch" + label] = basis_derivatives.dh_r
            exclude_from_sorting = expected_values[stage_name].keys() 
            exclude_from_sorting.append('sorting_items')
            #Generate Expectations
            for i in xrange(Nbasis):
                label = basis_labels[i]
                (Eui, dEui) = mbar.computeExpectations(individualU_klns[stage_name][label], real_space=True)
                (Eui2, dEui2) = mbar.computeExpectations(individualU_klns[stage_name][label]**2)
                expected_values[stage_name]['var_u'+label] = Eui2 - Eui**2
                dvar_ui = numpy.sqrt(dEui2**2 + 2*(Eui*dEui)**2)
                expected_values[stage_name]['dvar_u'+label] = dvar_ui
                expected_values[stage_name]['Eu'+label] = Eui
                expected_values[stage_name]['dEu'+label] = dEui
                expected_values[stage_name]['Eu'+label + '2'] = Eui2
                expected_values[stage_name]['dEu'+label + '2'] = dEui2
                for j in range(i+1,Nbasis): #Compute the cross terms, no need to run i=j since that was handled above
                    crosslabel = basis_labels[j]
                    (Eu_ij, dEu_ij) = mbar.computeExpectations(individualU_klns[stage_name][label] * individualU_klns[stage_name][crosslabel])
                    expected_values[stage_name]['Eu' + label + '_' + crosslabel] = Eu_ij
                    expected_values[stage_name]['dEu' + label + '_' + crosslabel] = dEu_ij
                expected_values[stage_name]['sorting_items'] = [i for i in expected_values[stage_name].keys() if i not in exclude_from_sorting]
        return expected_values
             
    #---------------------------------------------------------------------------------------------

    def calcdhdl_master(self, expected, lam_master, return_error=False):
        #calculate the dhdl directly
        dhdl_calc = numpy.zeros(len(lam_master))
        dhdl_err = numpy.zeros(len(lam_master))
        for label in expected['labels']:
            dhdl_calc += expected['Eu' + label] * expected['dswitch' + label](lam_master)
            dhdl_err += expected['dEu' + label] * expected['dswitch' + label](lam_master)
        dhdl_dict = {'natural':dhdl_calc}
        if return_error:
            dhdl_dict['plus'] = dhdl_calc + dhdl_err
            dhdl_dict['minus'] = dhdl_calc - dhdl_err
        return dhdl_dict
        

    def calcvar_master(self, expected, lam_master, return_error=False):
        #calculate the variance directly
        integrand = {}
        variance = {}
        #Allocate variables and error directions (+/-/not)
        error_dir = ['natural']
        if return_error:
            error_dir += ['plus','minus']
        for offset in error_dir:
            to_integrate = numpy.zeros(len(lam_master))
            for i in range(expected['Nbasis']): #Go through each basis function
                label = expected['labels'][i] #Assigh the i-th basis
                #Assign the error direciton
                if offset is 'plus':
                    varoffset = expected['dvar_u' + label]
                elif offset is 'minus':
                    varoffset = - expected['dvar_u' + label]
                else:
                    varoffset = 0
                #Variance of the i-th basis
                to_integrate += expected['dswitch' + label](lam_master)**2 * (expected['var_u' + label] + varoffset)
                for j in range(i+1,expected['Nbasis']): #Go through all j != i basis, preventing repeats by having 2x since Cov(x,y)=Cov(y,x)
                    crosslabel = expected['labels'][j] #Assign the j-th basis
                    #Assign error directions
                    if offset is 'plus':
                        firsttermoff = expected['dEu' + label]
                        secondtermoff = expected['dE_u' + crosslabel]
                        crosstermoff = expected['dEu' + label + '_' + crosslabel]
                    elif offset is 'minus':
                        firsttermoff = -expected['dEu' + label]
                        secondtermoff = -expected['dE_u' + crosslabel]
                        crosstermoff = -expected['dEu' + label + '_' + crosslabel]
                    else:
                        firsttermoff  = 0
                        secondtermoff = 0
                        crosstermoff = 0
                    #2 h_i' h_j' cov(i,j)
                    to_integrate += 2 * expected['dswitch'+label](lam_master) * expected['dswitch'+crosslabel](lam_master) * ((expected['Eu' + label + '_' + crosslabel]+crosstermoff) - (expected['Eu' + label]+firsttermoff)*(expected['Eu' + crosslabel]+secondtermoff))
            integrand[offset] = numpy.copy(to_integrate)
            variance[offset] = simps(integrand[offset],lam_master)
        return integrand,variance

    #---------------------------------------------------------------------------------------------
    def seqGen(self, sampled, lams = None):
        extra_lam = numpy.empty(0)
        if lams is None:
            lams = self.lam_range
        for i in lams:
            if not numpy.any([numpy.allclose([t],[i]) for t in sampled]) :
                extra_lam = numpy.append(extra_lam,i)
        return extra_lam

    def seqSolv(self, stage_range, sampled, extras, outlam=None, nstates=None):
        #determine the sequence to pull from the data
        if outlam is None:
            outlam = self.lam_range
        #Set the nstates if u_kln is fragemnted
        if nstates is None:
            nstates=self.complex.nstates
        nout = len(outlam)
        all_ndx_sorted = numpy.zeros(nout, numpy.int32)
        extracount = 0
        for i in xrange(nout):
            #Determine if our lambda is somewhere in the sampled
            container = numpy.array([numpy.allclose(t,outlam[i]) for t in sampled])
            if not numpy.any(container):
                #If entry not part of simulated states, grab from extra
                all_ndx_sorted[i] = nstates + extracount
                extracount += 1
            else: #Pull from the real state
                all_ndx_sorted[i] = int(stage_range[container][0]) #There should be only one True here if set up stage_range and sampled correctly
                #!!! The [0] is a hotfix for my EPA-C-R simulation
        return all_ndx_sorted #This is already a 0->1 order

    #---------------------------------------------------------------------------------------------
    def vargenerate_master(self, sequence=None, lam_in=None, verbose=None, calculate_var=True, calculatedhdl=False, return_error=False, bootstrap_error=False, bootstrap_count=200, basis_derivatives=None, single_stage=None, fragment_kln=False): 
        #In progress function to calculate the total variance along one transformation, independent of path

        #Set all flags
        if verbose is None:
            verbose=self.default_verbosity
        if calculatedhdl and not calculate_var:
            print "Warning: dHdL calculation requires variance calculation"
            print "Turning on variance calculation"
            calculate_var=True
        if bootstrap_error and return_error:
            print "Warning: Normal error and bootstrap error are incompatable, using bootstrap only"
            return_error=False
        if bootstrap_error:
            if int(bootstrap_count) < 2:
                print "Must have more than 1 bootstrap sample. Disabling bootstrap sampling!"
                bootstrap_error = False
        if lam_in is None:
            lam_in = self.lam_range
        if sequence is None:
            sequence = self.checkSequence()[0]
        self.checkSequence(sequence)
        valid_seqs = self.checkSequence()
        nstage = len(sequence)
        states = numpy.array(range(self.complex.nstates))
        #Figure out sequencing, need to determine actual state
        #Need the stage_ranges and the sampled for index sorting later
        nc = self.complex
        if sequence == valid_seqs[0]: #ep c ar
            stage_ranges = { 'EP':states[nc.real_EPCAR:nc.real_CAR+1], 'C':states[nc.real_CAR:nc.real_AR+1], 'AR':states[nc.real_AR:nc.real_alloff+1] }
            sampled = { 'EP':nc.real_E_states[stage_ranges['EP']], 'C':nc.real_C_states[stage_ranges['C']], 'AR':nc.real_R_states[stage_ranges['AR']] }
            extra0 = self.seqGen(sampled['EP'], lams=lam_in) #Stage 0: EPCAR -> CAR
            extra1 = self.seqGen(sampled['C'], lams=lam_in) #Stage 1: CAR -> AR
            extra2 = self.seqGen(sampled['AR'], lams=lam_in) #Stage 2: AR -> off
            extra_lam = {'EP':extra0, 'C':extra1, 'AR':extra2}
        elif sequence == valid_seqs[1]: #epa c r
            stage_ranges = { 'EPA':states[nc.real_EPCAR:nc.real_CR+1], 'C':states[nc.real_CR:nc.real_R+1], 'R':states[nc.real_R:nc.real_alloff+1] }
            sampled = { 'EPA':nc.real_E_states[stage_ranges['EPA']], 'C':nc.real_C_states[stage_ranges['C']], 'R':nc.real_R_states[stage_ranges['R']] }
            extra0 = self.seqGen(sampled['EPA'], lams=lam_in) #Stage 0: EPCAR -> CR
            extra1 = self.seqGen(sampled['C'], lams=lam_in) #Stage 1: CR -> R
            extra2 = self.seqGen(sampled['R'], lams=lam_in) #Stage 2: R -> off
            extra_lam = {'EPA':extra0, 'C':extra1, 'R':extra2}
        elif sequence == valid_seqs[2]: #ep a c r
            stage_ranges = { 'EP':states[nc.real_EPCAR:nc.real_CAR+1], 'A':states[nc.real_CAR:nc.real_CR+1], 'C':states[nc.real_CR:nc.real_R+1], 'R':states[nc.real_R:nc.real_alloff+1] }
            sampled = { 'EP':nc.real_E_states[stage_ranges['EP']], 'A':nc.real_A_states[stage_ranges['A']], 'C':nc.real_C_states[stage_ranges['C']], 'R':nc.real_R_states[stage_ranges['R']] }
            extra0 = self.seqGen(sampled['EP'], lams=lam_in) #Stage 0: EPCAR -> CAR
            extra1 = self.seqGen(sampled['A'], lams=lam_in) #Stage 1: CAR -> CR
            extra2 = self.seqGen(sampled['C'], lams=lam_in) #Stage 2: CR -> R
            extra3 = self.seqGen(sampled['R'], lams=lam_in) #Stage 3: R -> off
            extra_lam = {'EP':extra0, 'A':extra1, 'C':extra2, 'R':extra3}
        elif sequence == valid_seqs[3]: #a ep c r
            stage_ranges = { 'A':states[nc.real_EPCAR:nc.real_EPCR+1], 'EP':states[nc.real_EPCR:nc.real_CR+1], 'C':states[nc.real_CR:nc.real_R+1], 'R':states[nc.real_R:nc.real_alloff+1] }
            sampled = { 'A':nc.real_A_states[stage_ranges['A']], 'EP':nc.real_E_states[stage_ranges['EP']], 'C':nc.real_C_states[stage_ranges['C']], 'R':nc.real_R_states[stage_ranges['R']] }
            extra0 = self.seqGen(sampled['A'], lams=lam_in) #Stage 0: EPCAR -> EPCR
            extra1 = self.seqGen(sampled['EP'], lams=lam_in) #Stage 1: EPCR -> CR
            extra2 = self.seqGen(sampled['C'], lams=lam_in) #Stage 2: CR -> R
            extra3 = self.seqGen(sampled['R'], lams=lam_in) #Stage 3: R -> off
            extra_lam = {'A':extra0, 'EP':extra1, 'C':extra2, 'R':extra3}
        else:
            print "Sequence not identified by logic, failing"
            sys.exit(1)
        #Find the expectations
        expectations = self.buildExpected_master(self.complex, extra_lam, sequence, verbose=verbose, basis_derivatives=None, single_stage=single_stage, fragment_kln=fragment_kln, stage_ranges=stage_ranges)
        sorted_ndxs  = {}
        #If we are using only 1 stage, overwrite sequence to be just the stage
        if single_stage is not None:
            sequence = [ sequence[single_stage] ] #Wrap this in a list otherwise it will pull just the chars from the stage name
        #Generate the sorting algroithm
        for stage in sequence:
            if fragment_kln:
                sorted_ndxs[stage] = self.seqSolv(numpy.array(range(len(stage_ranges[stage]))), sampled[stage], extra_lam[stage], outlam=lam_in, nstates=len(stage_ranges[stage]))
            else:
                sorted_ndxs[stage] = self.seqSolv(stage_ranges[stage], sampled[stage], extra_lam[stage], outlam=lam_in)
            for key in expectations[stage]['sorting_items']:
                try:
                    expectations[stage][key] = expectations[stage][key][sorted_ndxs[stage]]
                except:
                    print stage, key
                    import pdb
                    pdb.set_trace()
                    pass
        #Perform remaining calculations
        if calculate_var:
            integrand = {}
            variance = {}
            dhdl = {}
            for stage in sequence:
                integrand[stage],variance[stage] = self.calcvar_master(expectations[stage], self.lam_range, return_error=return_error)
                if calculatedhdl:
                    dhdl[stage] = self.calcdhdl_master(expectations[stage], self.lam_range, return_error=return_error)
            #If bootstrap is on, run it
            if bootstrap_error:
                Nlam = len(self.lam_range)
                bootstrap_integrands = {}
                bootstrap_dhdl = {}
                bootstrap_error = {}
                bootstrap_dhdl_error = {}
                for stage in sequence:
                    #Deterimine shape of output matrix [le,bootstrap_count]
                    bootstrap_integrands[stage] = numpy.zeros([Nlam,bootstrap_count])
                    bootstrap_dhdl[stage] = numpy.zeros([Nlam,bootstrap_count])
                    bootstrap_error[stage] = numpy.zeros([Nlam])
                #Generate bootstraped data
                for i in xrange(bootstrap_count):
                    print "Bootstrap Pass: %d / %d" % (i+1,bootstrap_count) 
                    boot_expect = self.buildExpected_master(self.complex, extra_lam, sequence, verbose=verbose, bootstrap=True, fragment_kln=fragment_kln, stage_ranges=stage_ranges) 
                    for stage in sequence:
                        for key in boot_expect[stage]['sorting_items']:
                            boot_expect[stage][key] = boot_expect[stage][key][sorted_ndxs[stage]]
                        boot_integrand_holder, boot_variance_junk = self.calcvar_master(boot_expect[stage], self.lam_range, return_error=False)
                        if calculatedhdl:
                            boot_dhdl_holder = self.calcdhdl_master(boot_expect[stage], self.lam_range, return_error=False)
                            bootstrap_dhdl[stage][:,i] = boot_dhdl_holder['natural']
                        bootstrap_integrands[stage][:,i] = boot_integrand_holder['natural']
                #Determine bootstrap error
                for stage in sequence:
                    bootstrap_error[stage][:] = numpy.sqrt(numpy.var(bootstrap_integrands[stage][:,:],axis=1))
                    integrand[stage]['plus'] = integrand[stage]['natural'] + bootstrap_error[stage]
                    integrand[stage]['minus'] = integrand[stage]['natural'] - bootstrap_error[stage]
                    integrand[stage]['bootstrap_error'] = bootstrap_error[stage]
                    if calculatedhdl:
                        bootstrap_dhdl_error[stage] = numpy.sqrt(numpy.var(bootstrap_dhdl[stage][:,:], axis=1))
                        dhdl[stage]['plus'] = dhdl[stage]['natural'] + bootstrap_dhdl_error[stage]
                        dhdl[stage]['minus'] = dhdl[stage]['natural'] - bootstrap_dhdl_error[stage]
                        dhdl[stage]['bootstrap_error'] = bootstrap_dhdl_error[stage]
            if calculatedhdl:
                return integrand,variance,dhdl
            else:
                return integrand,variance
        else:
            return expectations

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

    def inv_var_e(self, being_predicted_basis, predicted_lam_e, sequence=None, return_error=False, calculatedhdl=False, verbose=None, bootstrap_error=False, bootstrap_count=200, single_stage=None):
        if verbose is None:
            verbose=self.default_verbosity
        if bootstrap_error and return_error:
            print "Warning: Normal error and bootstrap error are incompatable, using bootstrap only"
            return_error=False
        #Reconstruct the lambda_predicted to make a new set to pass to the source data and see if we can predict variance
        #For notation, the prediction will be "g" and the source will be "h"
        #Generally, dg/dl_g <u>_{l_g} \neq dh/dl_h <u>_{l_h}... but, since g and h explore the same domain of [0,1], <u>_g(g) = <u>_h(h=g)
        #We can then write <u>_{l_g}(l_g) = <u>_{l_h}(l_h = h_inv(g(l_g)))
      
        #Generate the lam_h to sample from
        lam_source_effective_e = self.basis.h_e_inv(being_predicted_basis.h_e(predicted_lam_e))
        expectations = self.vargenerate_master(sequence=sequence, lam_in=lam_source_effective_e, verbose=verbose, calculate_var=False, return_error=return_error, basis_derivatives=being_predicted_basis, single_stage=single_stage)
        if single_stage is not None:
            sequence = [ sequence[single_stage] ]
        pre_integrand = {}
        pre_variance = {}
        for stage in sequence:
            pre_integrand[stage], pre_variance[stage] = self.calcvar_master(expectations[stage], predicted_lam_e, return_error=return_error)
        #if bootstrap_error:
        #    if calculatedhdl:
        #        boot_integrands, boot_var, boot_dhdl = self.vargenerate_electrostatics(lam_in_e=lam_source_effective_e, verbose=verbose, calculate_var=True, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam_e)
        #    else:
        #        boot_integrands, boot_var = self.vargenerate_electrostatics(lam_in_e=lam_source_effective_e, verbose=verbose, calculate_var=False, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam_e)
        #    pre_integrand['plus'] = pre_integrand['natural'] + boot_integrands['bootstrap_error']
        #    pre_integrand['minus'] = pre_integrand['natural'] - boot_integrands['bootstrap_error']
        if calculatedhdl:
            pre_dhdl = {}
            for stage in sequence:
                pre_dhdl[stage] = self.calcdhdl_master(expectations[stage], predicted_lam_e, return_error=return_error)
                #if bootstrap_error:
                #    pre_dhdl['plus'] = pre_dhdl['natural'] + boot_dhdl['bootstrap_error']
                #    pre_dhdl['minus'] = pre_dhdl['natural'] - boot_dhdl['bootstrap_error']
                return pre_integrand, pre_variance, pre_dhdl
        else:
            return pre_integrand,pre_variance
    #---------------------------------------------------------------------------------------------
    def inv_var_master(self, being_predicted_basis, predicted_lam, return_error=False, calculatedhdl=False, verbose=None, bootstrap_error=False, bootstrap_count=200):
        if verbose is None:
            verbose=self.default_verbosity
        if bootstrap_error and return_error:
            print "Warning: Normal error and bootstrap error are incompatable, using bootstrap only"
            return_error=False
        #Reconstruct the lambda_predicted to make a new set to pass to the source data and see if we can predict variance
        #For notation, the prediction will be "g" and the source will be "h"
        #Generally, dg/dl_g <u>_{l_g} \neq dh/dl_h <u>_{l_h}... but, since g and h explore the same domain of [0,1], <u>_g(g) = <u>_h(h=g)
        #We can then write <u>_{l_g}(l_g) = <u>_{l_h}(l_h = h_inv(g(l_g)))
        predicted_lams = statelist(predicted_lam, predicted_lam, predicted_lam)
        #Generate the lam_h to sample from
        lam_source_effective_e = self.basis.h_e_inv(being_predicted_basis.h_e(predicted_lams.E_states))
        lam_source_effective_r = self.basis.h_r_inv(being_predicted_basis.h_r(predicted_lams.R_states))
        lam_source_effective_a = self.basis.h_a_inv(being_predicted_basis.h_a(predicted_lams.A_states))
        expectations = self.vargenerate_master(lam_in_e=lam_source_effective_e, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, verbose=verbose, calculate_var=False, return_error=return_error)
        pre_integrand, pre_variance = self.calcvar_master(expectations, being_predicted_basis, predicted_lam, return_error=return_error)
        if bootstrap_error:
            if calculatedhdl:
                boot_integrands, boot_var, boot_dhdl = self.vargenerate_master(lam_in_e=lam_source_effective_e, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, verbose=verbose, calculate_var=True, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam)
            else:
                boot_integrands, boot_var = self.vargenerate_electrostatics(lam_in_e=lam_source_effective_e, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, verbose=verbose, calculate_var=False, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam)
            pre_integrand['plus'] = pre_integrand['natural'] + boot_integrands['bootstrap_error']
            pre_integrand['minus'] = pre_integrand['natural'] - boot_integrands['bootstrap_error']
        if calculatedhdl:
            pre_dhdl = self.calcdhdl_master(expectations, being_predicted_basis, predicted_lam, return_error=return_error)
            if bootstrap_error:
                pre_dhdl['plus'] = pre_dhdl['natural'] + boot_dhdl['bootstrap_error']
                pre_dhdl['minus'] = pre_dhdl['natural'] - boot_dhdl['bootstrap_error']
            return pre_integrand, pre_variance, pre_dhdl
        else:
            return pre_integrand,pre_variance

    #---------------------------------------------------------------------------------------------
    def inv_var_xform(self, const_matricies, being_predicted_basis, predicted_lam, return_error=False, calculatedhdl=False, verbose=None, bootstrap_error=False, bootstrap_count=200):
        if verbose is None:
            verbose=self.default_verbosity
        if bootstrap_error and return_error:
            print "Warning: Normal error and bootstrap error are incompatable, using bootstrap only"
            return_error=False
        #Reconstruct the lambda_predicted to make a new set to pass to the source data and see if we can predict variance
        #For notation, the prediction will be "g" and the source will be "h"
        #Generally, dg/dl_g <u>_{l_g} \neq dh/dl_h <u>_{l_h}... but, since g and h explore the same domain of [0,1], <u>_g(g) = <u>_h(h=g)
        #We can then write <u>_{l_g}(l_g) = <u>_{l_h}(l_h = h_inv(g(l_g)))
        predicted_lams = statelist(predicted_lam, predicted_lam, predicted_lam)
        #Generate the lam_h to sample from
        lam_source_effective_e = self.basis.h_e_inv(being_predicted_basis.h_e(predicted_lams.E_states))
        lam_source_effective_r = self.basis.h_r_inv(being_predicted_basis.h_r(predicted_lams.R_states))
        lam_source_effective_a = self.basis.h_a_inv(being_predicted_basis.h_a(predicted_lams.A_states))
        expectations = self.vargenerate_xform(const_matricies,lam_in_e=lam_source_effective_e, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, verbose=verbose, calculate_var=False, return_error=return_error)
        pre_integrand, pre_variance = self.calcvar_master(expectations, being_predicted_basis, predicted_lam, return_error=return_error)
        if bootstrap_error:
            if calculatedhdl:
                boot_integrands, boot_var, boot_dhdl = self.vargenerate_xform(const_matricies,lam_in_e=lam_source_effective_e, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, verbose=verbose, calculate_var=True, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam)
            else:
                boot_integrands, boot_var = self.vargenerate_xform(const_matricies,lam_in_e=lam_source_effective_e, lam_in_r=lam_source_effective_r, lam_in_a=lam_source_effective_a, verbose=verbose, calculate_var=False, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam)
            pre_integrand['plus'] = pre_integrand['natural'] + boot_integrands['bootstrap_error']
            pre_integrand['minus'] = pre_integrand['natural'] - boot_integrands['bootstrap_error']
        if calculatedhdl:
            pre_dhdl = self.calcdhdl_master(expectations, being_predicted_basis, predicted_lam, return_error=return_error)
            if bootstrap_error:
                pre_dhdl['plus'] = pre_dhdl['natural'] + boot_dhdl['bootstrap_error']
                pre_dhdl['minus'] = pre_dhdl['natural'] - boot_dhdl['bootstrap_error']
            return pre_integrand, pre_variance, pre_dhdl
        else:
            return pre_integrand,pre_variance
    #---------------------------------------------------------------------------------------------

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
        for k in range(RA_count):
            print "state: %i" % k
            for n in range(self.complex.retained_iters):
                splined = US(RA_lam[::-1], u_kln[k,self.complex.real_AR:self.complex.real_alloff+1,n][::-1])
                if numpy.any(numpy.isnan(splined(RA_lam,1))):
                    import pdb
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
    def free_energy(self, verbose=None, startstate=None, endstate=None):
        #Compute the free energy difference between states
        #Startstate is the the i state you want to compare against
        #Endstate is the j state you want to go to
        if verbose is None:
            verbose = self.default_verbosity
        for nc in [self.vacuum, self.complex]:
            if not nc.mbar_ready: nc.compute_mbar()
            (nc.DeltaF_ij, nc.dDeltaF_ij) = nc.mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
            if verbose or True: #Making this forced
                print "%s DeltaF_ij:" % nc.phase
                for i in range(nc.nstates):
                    for j in range(nc.nstates):
                        print "%8.3f" % nc.DeltaF_ij[i,j],
                    print ""
                print "%s dDeltaF_ij:" % nc.phase
                for i in range(nc.nstates):
                    for j in range(nc.nstates):
                        print "%8.3f" % nc.dDeltaF_ij[i,j],
                    print ""
            if startstate is None or startstate is 'EPCAR':
                nc.free_energy_start = nc.real_EPCAR
            elif startstate is 'PCAR':
                nc.free_energy_start = nc.real_PCAR
            elif startstate is 'CAR':
                nc.free_energy_start = nc.real_CAR
            elif startstate is 'AR':
                nc.free_energy_start = nc.real_AR
            elif startstate is 'R':
                nc.free_energy_start = nc.real_R
            elif startstate is 'Alloff':
                nc.free_energy_start = nc.real_alloff
            else:
                nc.free_energy_start = startstate
            if endstate is None or endstate is 'EPCAR':
                nc.free_energy_end = nc.real_EPCAR
            elif endstate is 'PCAR':
                nc.free_energy_end = nc.real_PCAR
            elif endstate is 'CAR':
                nc.free_energy_end = nc.real_CAR
            elif endstate is 'AR':
                nc.free_energy_end = nc.real_AR
            elif endstate is 'R':
                nc.free_energy_end = nc.real_R
            elif endstate is 'Alloff':
                nc.free_energy_end = nc.real_alloff
            else:
                nc.free_energy_end = endstate

        self.vacuum.DeltaF = self.vacuum.DeltaF_ij[self.vacuum.free_energy_start, self.vacuum.free_energy_end]
        self.complex.DeltaF = self.complex.DeltaF_ij[self.complex.free_energy_start, self.complex.free_energy_end]
        self.vacuum.dDeltaF = self.vacuum.dDeltaF_ij[self.vacuum.free_energy_start, self.vacuum.free_energy_end]
        self.complex.dDeltaF = self.complex.dDeltaF_ij[self.complex.free_energy_start, self.complex.free_energy_end]
        #Reversed the sign to make the math be correct
        self.DeltaF = -(self.vacuum.DeltaF - self.complex.DeltaF)
        self.dDeltaF = numpy.sqrt(self.vacuum.dDeltaF**2 + self.complex.dDeltaF**2)
        print "Binding free energy : %16.3f +- %.3f kT (%16.3f +- %.3f kcal/mol)" % (self.DeltaF, self.dDeltaF, self.DeltaF * self.complex.kcalmol, self.dDeltaF * self.complex.kcalmol)
        print ""
        print "DeltaG vacuum       : %16.3f +- %.3f kT (%16.3f +- %.3f kcal/mol)" % (self.vacuum.DeltaF, self.vacuum.dDeltaF, self.vacuum.DeltaF * self.vacuum.kcalmol, self.vacuum.dDeltaF * self.vacuum.kcalmol)
        print "DeltaG complex      : %16.3f +- %.3f kT (%16.3f +- %.5f kcal/mol)" % (self.complex.DeltaF, self.complex.dDeltaF, self.complex.DeltaF * self.complex.kcalmol, self.complex.dDeltaF * self.complex.kcalmol)
    #---------------------------------------------------------------------------------------------

    def __init__(self,source_basis,source_comp,source_vac,verbose=False,SC=False,lam_range = None):
        #Check if the source information are the correct classes
        incorrect_input = False
        from basisanalyze.linfunctions import LinFunctions
        from basisanalyze.ncdata import ncdata
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
        if lam_range is None:
            self.lam_range = scipy.linspace(0.0,1.0,101)
        else:
            self.lam_range = lam_range
        self.default_verbosity = verbose
        #Clean up imports
        del LinFunctions, ncdata

        return

if __name__ == "__main__":
    print "Syntax is good boss"
