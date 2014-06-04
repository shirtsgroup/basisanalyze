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
        
    def sequence_e(self, nc, extra_E, lam_out_e = None):
        if lam_out_e is None:
            lam_out_e = self.lam_range
        extra_count = len(extra_E)
        #Check the ordering of the extra states
        sign = extra_E[0]-extra_E[-1]
        if sign == 0 or sign < 0: #1D array or increasing
           order='up'
        else:
           order='down'
        #Check ordering on extra (assumes they were build in a monotonic order)
        if nc.Inversion: #This block is wrong, but unused
            real_E_indicies = range(nc.real_EAR,nc.real_inverse)
        else:
            real_E_indicies = range(nc.real_EAR,nc.real_AR)
        #Determine if it was done with E-AR or EA-R based on the AR and the R states
        extra_E_indicies = range(nc.nstates,nc.nstates+extra_count)
        if order is 'up': #Checks if they are in increasing order
            extra_E_indicies = extra_E_indicies[::-1] #reverse to decreasing order for uniform math
        if lam_out_e[0] < lam_out_e[-1]: #make the lam_out sequence decending too
            lam_out_e = lam_out_e[::-1]
        sim_E_looper = 0
        extra_E_looper = 0
        all_ndx_sorted = numpy.zeros(len(lam_out_e), numpy.int32)
        for i in range(len(lam_out_e)):
            if nc.Inversion: #Eventually fix this
                container = numpy.array([numpy.allclose(t,lam_out_e[i]) for t in self.basis.h_e_inv(nc.real_E_states)])
            else:
                container = numpy.array([numpy.allclose(t,lam_out_e[i]) for t in nc.real_E_states])
            if not numpy.any(container):
                #If entry not part of the simualted states, grab it from the extra ones
                all_ndx_sorted[i] = extra_E_indicies[extra_E_looper]
                extra_E_looper += 1
            else: #pull in entry from the real ones
                all_ndx_sorted[i] = int(numpy.array(range(nc.nstates))[numpy.logical_and(container,nc.real_PMEFull_states==nc.real_E_states)])
        #Reverse sequence to return a 0 -> 1 order
        return all_ndx_sorted[::-1]

    def sequence_cap(self, nc, extra_states, lam_out=None, single_basis=None, sampledR=numpy.array([1.0,0.75,0.50,0.25,0.00])):
        if lam_out is None:
            lam_out = statelist(self.lam_range, self.lam_range, self.lam_range)
        extra_count = extra_states.nstates
        if single_basis:
            ##Fill with junk
            sampledR = numpy.array([-1]*nc.nstates)
            #Set fully coupled and state of interst
            sampledR[0] = 1.0
            sampledR[single_basis]=0
            nstates= nc.nstates
            #sampledR=[1.0,0.0]
            #nstates = 2
        else:
            nstates = nc.nstates

        extra_indicies = range(nstates,nstates+extra_count)
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
        for i in range(lam_out.nstates):
            #Pick out lambdas
            lamR = lam_out.R_states[i]
            #Generate logic container to determine of the state described by i was already sampled, the allclose is to adjust for small differences between t,lamE; u,lamR; v,lamA.
            container = numpy.array([numpy.allclose([u],[lamR]) for u in sampledR])
            if not numpy.any(container):
                #If entry not part of the simualted states, grab it from the extra ones
                all_ndx_sorted[i] = extra_indicies[extra_looper]
                extra_looper += 1
            else: #pull in entry from the real ones, based on where the logic container says it is AND where PMEFull=E, prevents getting the PMEsolve states
                all_ndx_sorted[i] = int(numpy.array(range(nstates))[container])
        #Reverse sequence to return a 0 -> 1 order
        if revert_lam_out:
            lam_out.reverse_order()
        return all_ndx_sorted[::-1]

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
        real_E_indicies = range(nc.real_EAR,nc.real_AR) #This statment is wrong, but unused
        real_R_indicies = nc.real_R_states
        real_A_indicies = nc.real_A_states
        #Determine if it was done with E-AR or EA-R based on the AR and the R states
        if nc.real_R == nc.nstates-1:
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

    def Ugen_EP_C_AR(self, nc):
        const_R_matrix = nc.u_kln[:,nc.real_R,:] - nc.u_kln[:,nc.real_alloff,:]
        const_A_matrix = nc.u_kln[:,nc.real_AR,:] - nc.u_kln[:,nc.real_R,:]
        const_C_matrix = nc.u_kln[:,nc.real_CAR,:] - nc.u_kln[:,nc.real_AR,:]
        const_E_matrix = nc.u_kln[:,nc.real_EPCAR,:] - nc.u_kln[:,nc.real_PCAR,:]
        return const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix
    def Ugen_EPA_C_R(self, nc):
        const_R_matrix = nc.u_kln[:,nc.real_R,:] - nc.u_kln[:,nc.real_alloff,:]
        const_C_matrix = nc.u_kln[:,nc.real_CR,:] - nc.u_kln[:,nc.real_R,:]
        const_A_matrix = nc.u_kln[:,nc.real_CAR,:] - nc.u_kln[:,nc.real_CR,:]
        const_E_matrix = nc.u_kln[:,nc.real_EPCAR,:] - nc.u_kln[:,nc.real_PCAR,:]
        return const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix
    def Ugen_EP_A_C_R(self, nc):
        const_R_matrix = nc.u_kln[:,nc.real_R,:] - nc.u_kln[:,nc.real_alloff,:]
        const_C_matrix = nc.u_kln[:,nc.real_CR,:] - nc.u_kln[:,nc.real_R,:]
        const_A_matrix = nc.u_kln[:,nc.real_CAR,:] - nc.u_kln[:,nc.real_CR,:]
        const_E_matrix = nc.u_kln[:,nc.real_EPCAR,:] - nc.u_kln[:,nc.real_PCAR,:]
        return const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix
    def Ugen_A_EP_C_R(self, nc):
        const_R_matrix = nc.u_kln[:,nc.real_R,:] - nc.u_kln[:,nc.real_alloff,:]
        const_C_matrix = nc.u_kln[:,nc.real_CR,:] - nc.u_kln[:,nc.real_R,:]
        const_E_matrix = nc.u_kln[:,nc.real_EPCR,:] - nc.u_kln[:,nc.real_PCR,:]
        const_A_matrix = nc.u_kln[:,nc.real_EPCAR,:] - nc.u_kln[:,nc.real_EPCR,:]
        return const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix

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

    def buildExpected_master(self, nc, extra_lam, sequence, verbose=None, bootstrap=False):#, basislabels=['E','P','PS','C','R','A']):
        """
        This particular bit of coding will be reworked in this branch to make sure that all of the possible scheudles can be handled in a general maner with minimal user input.
        Unfortunatley, this will require reading in both the alchemy file and user input information.
        I have removed the perturbed expected from this since it is no longer used
        """
        if verbose is None:
            verbose=self.default_verbosity
        #Validate the sequence
        self.checkSequence(seq=sequence)
        valid_seqs = self.checkSequence()
        nstage = len(sequence)
        if len(extra_lam) != nstage:
            print "Incorrect number of extra_lam arguments. There must be at least one item per stage, even if its an empty object"
            sys.exit(1)
        #Compute the constant matricies
        #Start with alloff
        const_U_matrix = nc.u_kln[:,nc.real_alloff,:]
        #Compute PME
        PMEFull = nc.u_kln[:,nc.real_PCAR,:] - nc.u_kln[:,nc.real_CAR,:]
        PMELess = nc.u_kln[:,nc.real_Psolve,:] - nc.u_kln[:,nc.real_CAR,:]
        LamAtFull = nc.real_PMEFull_states[nc.real_PCAR]
        LamAtLess = nc.real_PMEFull_states[nc.real_Psolve]
        hless = self.basis.h_e(LamAtLess)
        hfull = self.basis.h_e(LamAtFull)
        const_P2_matrix = (PMELess/hless - PMEFull/hfull) / (hless-hfull)
        const_P_matrix = PMEFull/hfull - hfull*const_PMEsquare_matrix
        #Compute the rest of the basis based on sequence
        if sequence is valid_seqs[0]: #ep c ar
            const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EP_C_AR(nc)
            #Assign energy evaluation stages
            Ustage = [
                lambda lam: self.basis.h_e(lam)*const_E_matrix + \
                            (self.basis.h_e(lam)**2)*const_P2_matrix + \
                            self.basis.h_e(lam)*const_P_matrix + \
                            nc.u_kln[:,nc.real_CAR,:],
                lambda lam: lam * const_C_matrix + \
                            nc.u_kln[:,nc.real_AR,:],
                lambda lam: self.basis.h_r(lam)*const_R_matrix + \
                            self.basis.h_a(lam)*const_A_matrix + \
                            nc.u_kln[:,nc.real_alloff,:]
                     ]
        elif sequence is valid_seqs[1]: #epa, c, r
            const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EPA_C_R(nc)
            Ustage = [
                lambda lam: self.basis.h_e(lam)*const_E_matrix + \
                            (self.basis.h_e(lam)**2)*const_P2_matrix + \
                            self.basis.h_e(lam)*const_P_matrix + \
                            self.basis.h_a(lam)*const_A_matrix + \
                            nc.u_kln[:,nc.real_CR,:],
                lambda lam: lam * const_C_matrix + \
                            nc.u_kln[:,nc.real_R,:],
                lambda lam: self.basis.h_r(lam)*const_R_matrix + \
                            nc.u_kln[:,nc.real_alloff,:]
                     ]
        elif sequence is valid_seqs[2]: #ep, a, c, r
            const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EP_A_C_R(nc)
            Ustage = [
                lambda lam: self.basis.h_e(lam)*const_E_matrix + \
                            (self.basis.h_e(lam)**2)*const_P2_matrix + \
                            self.basis.h_e(lam)*const_P_matrix + \
                            nc.u_kln[:,nc.real_CAR,:],
                lambda lam: self.basis.h_a(lam)*const_A_matrix + \
                            nc.u_kln[:,nc.real_CR,:],
                lambda lam: lam * const_C_matrix + \
                            nc.u_kln[:,nc.real_R,:],
                lambda lam: self.basis.h_r(lam)*const_R_matrix + \
                            nc.u_kln[:,nc.real_alloff,:]
                     ]
        elif sequence is valid_seqs[3]: #a, ep, c, r
            const_R_matix, const_A_matrix, const_C_matrix, const_E_matrix = self.Ugen_EP_A_C_R(nc)
            Ustage = [
                lambda lam: self.basis.h_a(lam)*const_A_matrix + \
                            nc.u_kln[:,nc.real_EPCR,:],
                lambda lam: self.basis.h_e(lam)*const_E_matrix + \
                            (self.basis.h_e(lam)**2)*const_P2_matrix + \
                            self.basis.h_e(lam)*const_P_matrix + \
                            nc.u_kln[:,nc.real_CR,:],
                lambda lam: lam * const_C_matrix + \
                            nc.u_kln[:,nc.real_R,:],
                lambda lam: self.basis.h_r(lam)*const_R_matrix + \
                            nc.u_kln[:,nc.real_alloff,:]
                     ]
        else: #This sould not trip, added as safety
            print "How did you make it here?"
            sys.exit(1) 
        #Compute the potential energies
        u_klns = {}
        N_ks = {}
        individualU_klns = {}
        expected_values = {}
        for stage_index in xrange(nstage):
            xl_stage = extra_lam[stage_index]
            nxl = len(xl_stage)
            stage_name = sequence[stage_index]
            individualU_klns[stage_name] = {}
            #Preallocate all the energy memory
            u_klns[stage_name] = numpy.zeros([nc.nstates + xl_stage, nc.nstates + xl_stage, nc.retained_iters], numpy.float64)
            N_samples = nc.retained_iters
            N_ks[stage_name] = numpy.zeros(nc.nstates + xl_stage, numpy.int32)
            #Copy over original data
            u_klns[stage_name][:nc.nstates,:nc.nstates,:nc.retained_iters] = nc.u_kln
            N_ks[stage_name][:nc.nstates] = nc.N_k
            #Determine the set of changing basis functions
            nchanging = len(sequence[stage_index])
            basis_labels = []
            #Generate basis function containers, run through each one
            for label in sequence[stage_index]:
                basis_labels.append(label)
                if label == "P":
                    individualU_klns[stage_name]["P"] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
                    individualU_klns[stage_name]["P2"] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
                else:
                    individualU_klns[stage_name][label] = numpy.zeros(u_klns[stage_name].shape, numpy.float64)
            #Compute energies
            for i in xrange(nxl):
                lam = xl_stage
                u_klns[stage_name][:nc.nstates,i+nc.nstates,:] = Ustage[stage_index](lam)
            for i in xrange(nc.nstates + nxl):
                for label in sequence[stage_index]:
                    if   label == "E":
                        individualU_klns[stage_name][label][:nc.nstates,i,:] = const_E_matrix
                    elif label == "P":
                        individualU_klns[stage_name][label][:nc.nstates,i,:] = const_P_matrix
                        individualU_klns[stage_name]["P2"][:nc.nstates,i,:] = const_P2_matrix
                    elif label == "C":
                        individualU_klns[stage_name][label][:nc.nstates,i,:] = const_C_matrix
                    elif label == "A":
                        individualU_klns[stage_name][label][:nc.nstates,i,:] = const_A_matrix
                    elif label == "R":
                        individualU_klns[stage_name][label][:nc.nstates,i,:] = const_R_matrix
                    else:
                        print "Seriously, how did you make it here?"
                        sys.exit(1)
            if bootstrap:
                u_kln_boot = numpy.zeros(u_klns[stage_name].shape)
                individualU_klns_boot = {}
                for label in individualU_klns[stage_name].keys():
                    individualU_klns_boot[label] = numpy.zeros(u_kln_new.shape, numpy.float64)
                for state in xrange(u_kln_boot.shape[0]):
                    samplepool = random_integers(0,N_samples-1,N_samples) #Pull the indicies for the sample space, N number of times
                    for i in xrange(len(samplepool)): #Had to put this in its own loop as u_kln_new[state,:,samplepool] was returning a NxK matrix instead of a KxN
                        u_kln_boot[state,:,i] = u_klns[stage_name][state,:,samplepool[i]]
                        for label in individualU_klns[stage_name].keys():
                            individualU_kln_boot[label][state,:,i] = individualU_klns[stage_name][label][state,:,samplepool[i]]
                #Copy over shuffled data
                u_klns[stage_name] = u_kln_boot
                for label in individualU_klns[stage_name].keys:
                    individualU_klns[stage_name][label] = individualU_kln_boot[label]
            basis_labels = individualU_klns[stage_name].keys()
            Nbasis = len(basis_labels)
            #Prep MBAR
            if nc.mbar_ready:
                mbar = MBAR(u_klns[stage_name], N_ks[stage_name], verbose = verbose, method = 'adaptive', initial_f_k=numpy.concatenate((nc.mbar.f_k,numpy.zeros(nxl))))
            else:
                mbar = MBAR(u_klns[stage_name], N_ks[stage_name], verbose = verbose, method = 'adaptive')
            expected_values[stage_name] = {'labels':basis_labels, 'Nbasis':Nbasis}
            for label in basis_labels:
                if   label == "E":
                   expected_values[stage_name]["dswitch" + label] = self.basis.dh_e
                elif label == "P":
                    expected_values[stage_name]["dswitch" + label] = self.basis.dh_e
                elif label == "P2":
                    expected_values[stage_name]["dswitch" + label] = lambda X: 2*self.basis.h_e(X)*self.basis.dh_e(X)
                elif label == "C":
                    expected_values[stage_name]["dswitch" + label] = lambda X: X
                elif label == "A":
                    expected_values[stage_name]["dswitch" + label] = self.basis.dh_a
                elif label == "R":
                    expected_values[stage_name]["dswitch" + label] = self.basis.dh_r
            exclude_from_sorting = expected_values[stage_name].keys() 
            #Generate Expectations
            for i in xrange(Nbasis):
                label = basis_labels[i]
                (Eui, dEui) = mbar.computeExpectations(individualU_klns[stage_name][label])
                (Eui2, dEui2) = mbar.computeExpectations(individualU_klns[stage_name][label]**2)
                epected_values[stage_name]['var_u'+label] = Eui2 - Eui**2
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
        

    def calcvar_master(self, expected,  lam_master, return_error=False):
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

    def seqSolv(self, stage_range, sampled, extras, outlam=None):
        #determine the sequence to pull from the data
        if outlam is None:
            outlam = self.lam_range
        nout = len(outlam)
        all_ndx_sorted = numpy.zeros(nout, numpy.int32)
        extracount = 0
        for i in xrange(nout):
            #Determine if our lambda is somewhere in the sampled
            contianer = numpy.array([numpy.allclose(t,outlam[i]) for t in sampled])
            if not numpy.any(container):
                #If entry not part of simulated states, grap from extra
                all_ndx_sorted[i] = extras[extracount]
                extracount += 1
            else: #Pull from the real state
                all_ndx_sorted[i] = int(stage_range[container]) #There should be only one True here if set up stage_range and sampled correctly
        return all_ndx_sorted #This is already a 0->1 order

    #---------------------------------------------------------------------------------------------
    def vargenerate_master(self, sequence=None, lam_in=None, verbose=None, calculate_var=True, calculatedhdl=False, return_error=False, bootstrap_error=False, bootstrap_count=200): 
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
        valid_seqs = [ ['EP', 'C', 'AR'], ['EPA', 'C', 'R'], ['EP', 'A', 'C', 'R'], ['A', 'EP', 'C', 'R'] ]
        states = xrange(self.complex.nstates)
        #Figure out sequencing, need to determine actual state
        #Need the stage_ranges and the sampled for index sorting later
        if sequence is valid_seqs[0]: #ep c ar
            stage_ranges = { 'EP':states[nc.real_EPCAR:nc.real_CAR+1], 'C':states[nc.real_CAR:nc.real_AR+1], 'AR':states[nc.real_AR:nc.real_alloff+1] }
            sampled = { 'EP':nc.real_E_states[stage_ranges['EP']], 'C':nc.real_C_states[stage_ranges['C']], 'AR':nc.real_R_states[stage_ranges['AR']] }
            extra0 = self.seqGen(sampled['EP']) #Stage 0: EPCAR -> CAR
            extra1 = self.seqGen(sampled['C']) #Stage 1: CAR -> AR
            extra2 = self.seqGen(sampled['AR']) #Stage 2: AR -> off
            extra_lam = {'EP':extra0, 'C':extra1, 'AR':extra2}
        if sequence is valid_seqs[1]: #epa c r
            stage_ranges = { 'EPA':states[nc.real_EPCAR:nc.real_CR+1], 'C':states[nc.real_CR:nc.real_R+1], 'R':states[nc.real_R:nc.real_alloff+1] }
            sampled = { 'EPA':nc.real_E_states[stage_ranges['EPA']], 'C':nc.real_C_states[stage_ranges['C']], 'R':nc.real_R_states[stage_ranges['R']] }
            extra0 = self.seqGen(sampled['EPA']) #Stage 0: EPCAR -> CR
            extra1 = self.seqGen(sampled['C']) #Stage 1: CR -> R
            extra2 = self.seqGen(sampled['R']) #Stage 2: R -> off
            extra_lam = {'EPA':extra0, 'C':extra1, 'R':extra2}
        if sequence is valid_seqs[2]: #ep a c r
            stage_ranges = { 'EP':states[nc.real_EPCAR:nc.real_CAR+1], 'A':states[nc.real_CAR:nc.real_CR+1], 'C':states[nc.real_CR:nc.real_R+1], 'R':states[nc.real_R:nc.real_alloff+1] }
            sampled = { 'EP':nc.real_E_states[stage_ranges['EP']], 'A':nc.real_A_states[stage_ranges['A']], 'C':nc.real_C_states[stage_ranges['C']], 'R':nc.real_R_states[stage_ranges['R']] }
            extra0 = self.seqGen(sampled['EP']) #Stage 0: EPCAR -> CAR
            extra1 = self.seqGen(sampled['A']) #Stage 1: CAR -> CR
            extra2 = self.seqGen(sampled['C']) #Stage 2: CR -> R
            extra3 = self.seqGen(sampled['R']) #Stage 3: R -> off
            extra_lam = {'EP':extra0, 'A':extra1, 'C':extra2, 'R':extra3}
        if sequence is valid_seqs[3]: #a ep c r
            stage_ranges = { 'A':states[nc.real_EPCAR:nc.real_EPCR+1], 'EP':states[nc.real_EPCR:nc.real_CR+1], 'C':states[nc.real_CR:nc.real_R+1], 'R':states[nc.real_R:nc.real_alloff+1] }
            sampled = { 'A':nc.real_A_states[stage_ranges['A']], 'EP':nc.real_E_states[stage_ranges['EP']], 'C':nc.real_C_states[stage_ranges['C']], 'R':nc.real_R_states[stage_ranges['R']] }
            extra0 = self.seqGen(sampled['A']) #Stage 0: EPCAR -> EPCR
            extra1 = self.seqGen(smapled['EP']) #Stage 1: EPCR -> CR
            extra2 = self.seqGen(sampled['C']) #Stage 2: CR -> R
            extra3 = self.seqGen(sampled['R']) #Stage 3: R -> off
            extra_lam = {'A':extra0, 'EP':extra1, 'C':extra2, 'R':extra3}
        #Find the expectations:
        expectations = self.buildExpectations_master(self.complex, extra_lam, sequence, verbose=verbose)        
        sorted_ndxs  = {}
        #Generate the sorting algroithm
        for stage in sequence:
            sorted_ndxs[stage] = self.seqSolv(stage_range[stage], sampled[stage], extra_lam[stage])
            for key in expectations[sequence[stage]]['sorting_items']:
                expectations[stage][key] = expectations[stage][key][sorted_ndxs[stage]]
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
                for stage in sequence:
                    #Deterimine shape of output matrix [le,bootstrap_count]
                    bootstrap_ingegrands[stage] = numpy.zeros([Nlam,bootstrap_count])
                    bootstrap_dhdl[stage] = numpy.zeros([Nlam,bootstrap_count])
                    bootstrap_error[stage] = numpy.zeros([Nlam])
                #Generate bootstraped data
                for i in xrange(bootstrap_count):
                    print "Bootstrap Pass: %d / %d" (i+1,bootstrap_count) 
                    boot_expect = self.buildExpectations_master(self.complex, extra_lam, sequence, verbose=verbose, bootstrap=True)        
                    for stage in sequence:
                        for key in boot_expect[stage]['sorting_items']:
                            boot_expect[stage][key] = boot_expecti[stage][key][sorted_ndxs[stage]]
                        boot_integrand_holder, boot_variance_junk = self.calcvar_master(expectations[stage]. self.lam_range, return_error=False)
                        if calculatedhdl:
                            boot_dhdl_holder = self.calcdhdl_master(expectations[stage], self.lam_range, return_error=False)
                            bootstrap_dhdl[stage][:,i] = boot_dhdl_holder['natural']
                        bootstrap_integrands[stage][:,i] = boot_integrand_holder['natural']
                #Determine bootstrap error
                for stage in sequnce:
                    bootstrap_error[stage][:] = numpy.sqrt(numpy.var(bootstrap_integrands[stage][:,:],axis=1))
                    integrand[stage]['plus'] = integrand[stage]['natural'] + bootstrap_error[stage]
                    integrand[stage]['minus'] = integrand[stage]['natural'] - bootstrap_error[stage]
                    integrand[stage]['bootstrap_error'] = bootstrap_error[stage]
            if calculatedhdl:
                return integrand,variance,dhdl
            else:
                return integrand,variance
        else:
            return expectations

    #---------------------------------------------------------------------------------------------
    def vargenerate_electrostatics(self, lam_in_e=None, verbose=None, calculate_var=True, calculatedhdl=False, expect_method='complete', return_error=False, bootstrap_error=False, bootstrap_count=200, bootstrap_basis=None, bootstrap_lam=None, basis_derivatives=None):
        if verbose is None:
            verbose = self.default_verbosity
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
        if lam_in_e is None:
            xe = self.lam_range
        else:
            xe = lam_in_e
        extra_E_list = numpy.empty(0)
        for i in xe:
            if self.complex.Inversion: #Fix this eventually
                sampled_lam = self.basis.h_e_inv(self.complex.real_E_states)
            else:
                sampled_lam = self.complex.real_E_states
            if not numpy.any([numpy.allclose(t,i) for t in sampled_lam]):
                extra_E_list = numpy.append(extra_E_list,i)
        #Calculate the electrostatic expectations, basis derivatives should only be passed in if we are doing INVERSES
        expectations = self.buildExpected_electro(self.complex, extra_E_list, verbose=verbose, basis_derivatives=basis_derivatives)
        sorted_ndx = self.sequence_e(self.complex, extra_E_list, lam_out_e=xe)
        for key in expectations['sorting_items']:
            expectations[key] = expectations[key][sorted_ndx]
        if calculate_var:
            integrand,variance = self.calcvar_master(expectations, self.basis, xe, return_error=return_error)
            if calculatedhdl:
                dhdl = self.calcdhdl_master(expectations, self.basis, xe, return_error=return_error)
            #If bootstrap is on, run it
            if bootstrap_error:
                if bootstrap_basis is None: #Allows one to pass in prediced basis functions from the inv_var calls
                    bootstrap_basis=self.basis
                if bootstrap_lam is None:
                    bootstrap_lam=xe
                #Deterimine shape of output matrix [le,bootstrap_count]
                bootstrap_integrands = numpy.zeros([len(xe),bootstrap_count])
                bootstrap_dhdl = numpy.zeros([len(xe),bootstrap_count])
                bootstrap_error = numpy.zeros([len(xe)])
                for i in xrange(bootstrap_count):
                    print "Bootstrap pass: %d / %d" % (i+1,bootstrap_count)
                    #Normal case
                    boot_expect = self.buildExpected_electro(self.complex, extra_E_list, verbose=verbose, bootstrap=True)

                    for key in boot_expect['sorting_items']:
                        boot_expect[key] = boot_expect[key][sorted_ndx]

                    boot_integrand_holder, boot_variance_junk = self.calcvar_master(boot_expect, bootstrap_basis, bootstrap_lam, return_error=False)
                    if calculatedhdl:
                        boot_dhdl_holder = self.calcdhdl_master(boot_expect, bootstrap_basis, bootstrap_lam, return_error=False)
                        bootstrap_dhdl[:,i] = boot_dhdl_holder['natural']
                    bootstrap_integrands[:,i] = boot_integrand_holder['natural']
                #Calculate variance of the collection
                bootstrap_error[:] = numpy.sqrt(numpy.var(bootstrap_integrands[:,:], axis=1))
                integrand['plus'] = integrand['natural'] + bootstrap_error
                integrand['minus'] = integrand['natural'] - bootstrap_error
                integrand['bootstrap_error'] = bootstrap_error
                if calculatedhdl:
                    bootstrap_dhdl_error = numpy.sqrt(numpy.var(bootstrap_dhdl[:,:], axis=1))
                    dhdl['plus'] = dhdl['natural'] + bootstrap_dhdl_error
                    dhdl['minus'] = dhdl['natural'] - bootstrap_dhdl_error
                    dhdl['bootstrap_error'] = bootstrap_dhdl_error
            if calculatedhdl:
                return integrand,variance,dhdl
            else:
                return integrand,variance
        else:
            return expectations

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
                    bootstrap_dhdl_error = numpy.sqrt(numpy.var(bootstrap_dhdl[:,:], axis=1))
                    dhdl['plus'] = dhdl['natural'] + bootstrap_dhdl_error
                    dhdl['minus'] = dhdl['natural'] - bootstrap_dhdl_error
            if calculatedhdl:
                return integrand,variance,dhdl
            else:
                return integrand,variance
        else:
            return expectations

    #---------------------------------------------------------------------------------------------
    def vargenerate_cap(self, verbose=None, lam_in_r=None, calculate_var=True, calculatedhdl=False, return_error=False, bootstrap_error=False, bootstrap_count=200, bootstrap_basis=None, bootstrap_lam=None, single_basis=None, sampledR=[1.0,0.75,0.50,0.25,0.00]):
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
        if lam_in_r is None:
            xr = self.lam_range
        else:
            xr = lam_in_r
        lam_out = statelist(xr,xr,xr)
        
        #Construct the list of extra states we want to sample
        extra_R_list = numpy.empty(0)
        if single_basis:
            #Fill with junk
            sampledR = numpy.array([-1]*self.complex.nstates)
            #Set fully coupled and state of interst
            sampledR[0] = 1.0
            sampledR[single_basis]=0.00

        for j in zip(xr):
            if not numpy.any([numpy.allclose([u],[j]) for u in sampledR]) :
                extra_R_list = numpy.append(extra_R_list,j)
        extra_states = statelist(extra_R_list, extra_R_list, extra_R_list)

        #Find expectations, sort them
        expectations = self.buildExpected_cap(self.complex, extra_states, verbose=verbose, single_basis=single_basis)
        sorted_ndx = self.sequence_cap(self.complex, extra_states, lam_out=lam_out, single_basis=single_basis, sampledR=sampledR)
        for key in expectations['sorting_items']:
            expectations[key] = expectations[key][sorted_ndx]
        #Perform remaining calculations
        if calculate_var:
            integrand,variance = self.calcvar_master(expectations, self.basis, lam_out.R_states, return_error=return_error)
            if calculatedhdl:
                dhdl = self.calcdhdl_master(expectations, self.basis, lam_out.R_states, return_error=return_error)
            #If bootstrap is on, run it
            if bootstrap_error:
                if bootstrap_basis is None: #Allows one to pass in prediced basis functions from the inv_var calls
                    bootstrap_basis=self.basis
                if bootstrap_lam is None:
                    bootstrap_lam=lam_out.R_states
                #Deterimine shape of output matrix [le,bootstrap_count]
                bootstrap_integrands = numpy.zeros([lam_out.nstates,bootstrap_count])
                bootstrap_variance   = numpy.zeros([bootstrap_count])
                bootstrap_dhdl = numpy.zeros([lam_out.nstates,bootstrap_count])
                bootstrap_error = numpy.zeros([lam_out.nstates])
                for i in xrange(bootstrap_count):
                    print "Bootstrap pass: %d / %d" % (i+1,bootstrap_count)
                    #Normal case
                    boot_expect = self.buildExpected_cap(self.complex, extra_states, verbose=verbose, bootstrap=True, single_basis=single_basis)

                    for key in boot_expect['sorting_items']:
                        boot_expect[key] = boot_expect[key][sorted_ndx]

                    boot_integrand_holder, boot_variance_holder = self.calcvar_master(boot_expect, bootstrap_basis, bootstrap_lam, return_error=False)
                    if calculatedhdl:
                        boot_dhdl_holder = self.calcdhdl_master(boot_expect, bootstrap_basis, bootstrap_lam, return_error=False)
                        bootstrap_dhdl[:,i] = boot_dhdl_holder['natural']
                    bootstrap_integrands[:,i] = boot_integrand_holder['natural']
                    bootstrap_variance[i] = boot_variance_holder['natural']
                #Calculate variance of the collection
                bootstrap_error[:] = numpy.sqrt(numpy.var(bootstrap_integrands[:,:], axis=1))
                bootstrap_var_error = numpy.sqrt(numpy.var(bootstrap_variance))
                integrand['plus'] = integrand['natural'] + bootstrap_error
                integrand['minus'] = integrand['natural'] - bootstrap_error
                integrand['bootstrap_error'] = bootstrap_error
                variance['plus'] = variance['natural'] + bootstrap_var_error
                variance['minus'] = variance['natural'] - bootstrap_var_error
                variance['bootstrap_error'] = bootstrap_var_error
                if calculatedhdl:
                    bootstrap_dhdl_error = numpy.sqrt(numpy.var(bootstrap_dhdl[:,:], axis=1))
                    dhdl['plus'] = dhdl['natural'] + bootstrap_dhdl_error
                    dhdl['minus'] = dhdl['natural'] - bootstrap_dhdl_error
                    dhdl['bootstrap_error'] = bootstrap_dhdl_error
            if calculatedhdl:
                return integrand,variance,dhdl
            else:
                return integrand,variance
        else:
            return expectations

    def vargenerate_xform(self, const_matricies, verbose=None, lam_in_r=None, lam_in_a=None, lam_in_e=None, calculate_var=True, calculatedhdl=False, return_error=False, bootstrap_error=False, bootstrap_count=200, bootstrap_basis=None, bootstrap_lam=None):
        #special case for transforming molecule very simlar to the master version, only manually done for speed
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
        if lam_in_r is None:
            xr = self.lam_range
        else:
            xr = lam_in_r
        if lam_in_a is None:
            xa = self.lam_range
        else:
            xa = lam_in_a
        if lam_in_e is None:
            xe = self.lam_range
        else:
            xe = lam_in_e
        #Calculate the variance of the original basis set
        if not xr.shape == xa.shape or not xr.shape == xe.shape:
            print 'input lambdas not the same size!'
            print '# repulsive lambda:     %i' % (xr.size)
            print '# attractive lambda:    %i' % (xa.size)
            print '# electrostatic lambda: %i' % (xe.size)
            sys.exit(1)
        lam_out = statelist(xe,xr,xa)
        
        #Construct the list of extra states we want to sample
        extra_R_list = numpy.empty(0)
        extra_A_list = numpy.empty(0)
        extra_E_list = numpy.empty(0)
        sampledR = self.complex.real_R_states
        sampledA = self.complex.real_A_states
        if self.complex.Inversion: #Fix this eventually
            sampledE = self.basis.h_e_inv(self.complex.real_E_states)
        else:
            sampledE = self.complex.real_E_states
        for i,j,k in zip(xe,xr,xa):
            if not numpy.any([numpy.allclose([t,u,v],[i,j,k]) for t,u,v in zip(sampledE,sampledR,sampledA)]) :
                extra_E_list = numpy.append(extra_R_list,i)
                extra_R_list = numpy.append(extra_R_list,j)
                extra_A_list = numpy.append(extra_A_list,k)
        extra_states = statelist(extra_E_list, extra_R_list, extra_A_list)

        #Find expectations, sort them
        expectations = self.buildExpected_xform(const_matricies, self.complex, extra_states, verbose=verbose)
        sorted_ndx = self.sequence_master(self.complex, extra_states, lam_out=lam_out)
        for key in expectations['sorting_items']:
            expectations[key] = expectations[key][sorted_ndx]
        #Perform remaining calculations
        if calculate_var:
            integrand,variance = self.calcvar_master(expectations, self.basis, lam_out.R_states, return_error=return_error)
            if calculatedhdl:
                dhdl = self.calcdhdl_master(expectations, self.basis, lam_out.R_states, return_error=return_error)
            #If bootstrap is on, run it
            if bootstrap_error:
                if bootstrap_basis is None: #Allows one to pass in prediced basis functions from the inv_var calls
                    bootstrap_basis=self.basis
                if bootstrap_lam is None:
                    bootstrap_lam=lam_out.E_states
                #Deterimine shape of output matrix [le,bootstrap_count]
                bootstrap_integrands = numpy.zeros([lam_out.nstates,bootstrap_count])
                bootstrap_dhdl = numpy.zeros([lam_out.nstates,bootstrap_count])
                bootstrap_error = numpy.zeros([lam_out.nstates])
                for i in xrange(bootstrap_count):
                    print "Bootstrap pass: %d / %d" % (i+1,bootstrap_count)
                    #Normal case
                    boot_expect = self.buildExpected_xform(const_matricies, self.complex, extra_states, verbose=verbose, bootstrap=True)

                    for key in boot_expect['sorting_items']:
                        boot_expect[key] = boot_expect[key][sorted_ndx]

                    boot_integrand_holder, boot_variance_junk = self.calcvar_master(boot_expect, bootstrap_basis, bootstrap_lam, return_error=False)
                    if calculatedhdl:
                        boot_dhdl_holder = self.calcdhdl_master(boot_expect, bootstrap_basis, bootstrap_lam, return_error=False)
                        bootstrap_dhdl[:,i] = boot_dhdl_holder['natural']
                    bootstrap_integrands[:,i] = boot_integrand_holder['natural']
                #Calculate variance of the collection
                bootstrap_error[:] = numpy.sqrt(numpy.var(bootstrap_integrands[:,:], axis=1))
                integrand['plus'] = integrand['natural'] + bootstrap_error
                integrand['minus'] = integrand['natural'] - bootstrap_error
                integrand['bootstrap_error'] = bootstrap_error
                if calculatedhdl:
                    bootstrap_dhdl_error = numpy.sqrt(numpy.var(bootstrap_dhdl[:,:], axis=1))
                    dhdl['plus'] = dhdl['natural'] + bootstrap_dhdl_error
                    dhdl['minus'] = dhdl['natural'] - bootstrap_dhdl_error
                    dhdl['bootstrap_error'] = bootstrap_dhdl_error
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

    def inv_var_e(self, being_predicted_basis, predicted_lam_e, return_error=False, calculatedhdl=False, verbose=None, bootstrap_error=False, bootstrap_count=200):
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
        expectations = self.vargenerate_electrostatics(lam_in_e=lam_source_effective_e, verbose=verbose, calculate_var=False, return_error=return_error, basis_derivatives=being_predicted_basis)
        pre_integrand, pre_variance = self.calcvar_master(expectations, being_predicted_basis, predicted_lam_e, return_error=return_error)
        if bootstrap_error:
            if calculatedhdl:
                boot_integrands, boot_var, boot_dhdl = self.vargenerate_electrostatics(lam_in_e=lam_source_effective_e, verbose=verbose, calculate_var=True, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam_e)
            else:
                boot_integrands, boot_var = self.vargenerate_electrostatics(lam_in_e=lam_source_effective_e, verbose=verbose, calculate_var=False, calculatedhdl=True, return_error=return_error, bootstrap_error=True, bootstrap_count=bootstrap_count, bootstrap_basis=being_predicted_basis, bootstrap_lam=predicted_lam_e)
            pre_integrand['plus'] = pre_integrand['natural'] + boot_integrands['bootstrap_error']
            pre_integrand['minus'] = pre_integrand['natural'] - boot_integrands['bootstrap_error']
        if calculatedhdl:
            pre_dhdl = self.calcdhdl_master(expectations, being_predicted_basis, predicted_lam_e, return_error=return_error)
            if bootstrap_error:
                pre_dhdl['plus'] = pre_dhdl['natural'] + boot_dhdl['bootstrap_error']
                pre_dhdl['minus'] = pre_dhdl['natural'] - boot_dhdl['bootstrap_error']
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
    def free_energy(self, verbose=None, startstate=None, endstate=None):
        #Compute the free energy difference between states
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
            if startstate is None or startstate is 'EAR':
                nc.free_energy_start = nc.real_EAR
            elif startstate is 'AR':
                nc.free_energy_start = nc.real_AR
            elif startstate is 'R':
                nc.free_energy_start = nc.real_R
            elif startstate is 'Inverse':
                nc.free_energy_start = nc.real_inverse
            elif startstate is 'Alloff':
                nc.free_energy_start = nc.real_alloff
            else:
                nc.free_energy_start = startstate
            if endstate is None or endstate is 'EAR':
                nc.free_energy_end = nc.real_EAR
            elif endstate is 'AR':
                nc.free_energy_end = nc.real_AR
            elif endstate is 'R':
                nc.free_energy_end = nc.real_R
            elif endstate is 'Inverse':
                nc.free_energy_end = nc.real_inverse
            elif endstate is 'Alloff':
                nc.free_energy_end = nc.real_alloff
            else:
                nc.free_energy_end = endstate
        self.vacuum.DeltaF = self.vacuum.DeltaF_ij[self.vacuum.free_energy_start, self.vacuum.free_energy_end]
        self.complex.DeltaF = self.complex.DeltaF_ij[self.complex.free_energy_start, self.complex.free_energy_end]
        self.vacuum.dDeltaF = self.vacuum.dDeltaF_ij[self.vacuum.free_energy_start, self.vacuum.free_energy_end]
        self.complex.dDeltaF = self.complex.dDeltaF_ij[self.complex.free_energy_start, self.complex.free_energy_end]
        self.DeltaF = self.vacuum.DeltaF - self.complex.DeltaF
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
