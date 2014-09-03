"""
Class: ncdata
Manipulate the ncdata files analyze results from YANK. Most code from John Chodera's analyze.py script in YANK
"""

import numpy
import os
import os.path
import sys
import netCDF4 as netcdf # netcdf4-python
from pymbar import MBAR # multistate Bennett acceptance ratio
import timeseries # for statistical inefficiency analysis
import simtk.unit as units
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

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
            atom["x"] = "%8.4f" % coordinates[index,0]
            atom["y"] = "%8.4f" % coordinates[index,1]
            atom["z"] = "%8.4f" % coordinates[index,2]
            #               ResNumber ResName     AtomName AtomNumber  X-pos  Y-pos  Z-pos
            filename.write('%(Seqno)5s%(resName)5s%(atom)5s%(serial)5s %(x)8s %(y)8s %(z)8s\n' % atom)
            
        # Close file.
        #outfile.close()
    
        return

    def write_gro_replica_trajectories(self, directory, prefix, title, trajectory_by_state=True, fraction_to_write=None, equilibrated_data = False, uncorrelated_data = False, states_to_write=None):
        """Write out replica trajectories as multi-model GRO files.
    
        ARGUMENTS
           directory (string) - the directory to write files to
           prefix (string) - prefix for replica trajectory files
           title (string) - the title to give each GRO file
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
        #Determine which states we are writing, supports python list slicing
        if states_to_write is None:
            states_to_write = range(0,self.nstates)
        else:
            if type(states_to_write) in [list, tuple]:
                states_to_write = [range(0,self.nstates)[i] for i in states_to_write]
            else:
                states_to_write = range(0,self.nstates)[states_to_write]

        if trajectory_by_state:
            for state_index in states_to_write:
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
            for replica_index in states_to_write:
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
        print T
        for t in range(T-1):
            print t
            g_t[t] = timeseries.statisticalInefficiency(A_t[t:T])
            Neff_t[t] = (T-t+1) / g_t[t]
        
        Neff_max = Neff_t.max()
        t = Neff_t.argmax()
        g = g_t[t]
        
        return (t, g, Neff_max)

    def _subsample_kln(self, u_kln):
        #Try to load in the data
        if self.save_equil_data: #Check if we want to save/load equilibration data
            try:
                equil_data = numpy.load(os.path.join(self.source_directory, self.save_prefix + self.phase + '_equil_data_%s.npz' % self.subsample_method))
                if self.nequil is None:
                    self.nequil = equil_data['nequil']
                elif type(self.nequil) is int and self.subsample_method == 'per-state':
                    print "WARRNING: Per-state subsampling requested with only single value for equilibration..."
                    try:
                        self.nequil = equil_data['nequil']
                        print "Loading equilibration from file with %i states read" % self.nstates
                    except:
                        print "Assuming equal equilibration per state of %i" % self.nequil
                        self.nequil = numpy.array([self.nequil] * self.nstates)
                self.g_t = equil_data['g_t']
                Neff_max = equil_data['Neff_max']
                #Do equilibration if we have not already
                if self.subsample_method == 'per-state' and (len(self.g_t) < self.nstates or len(self.nequil) < self.nstates):
                    equil_loaded = False
                    raise IndexError
                else:
                    equil_loaded = True
            except:
                if self.subsample_method == 'per-state':
                    self.nequil = numpy.zeros([self.nstates], dtype=numpy.int32)
                    self.g_t = numpy.zeros([self.nstates])
                    Neff_max = numpy.zeros([self.nstates])
                    for k in xrange(self.nstates):
                        if self.verbose: print "Computing timeseries for state %i/%i" % (k,self.nstates-1)
                        self.nequil[k] = 0
                        self.g_t[k] = timeseries.statisticalInefficiency(u_kln[k,k,:])
                        Neff_max[k] = (u_kln[k,k,:].size + 1 ) / self.g_t[k]
                        #[self.nequil[k], self.g_t[k], Neff_max[k]] = self._detect_equilibration(u_kln[k,k,:])
                else:
                    if self.nequil is None:
                        [self.nequil, self.g_t, Neff_max] = self._detect_equilibration(self.u_n)
                    else:
                        [self.nequil_timeseries, self.g_t, Neff_max] = self._detect_equilibration(self.u_n)
                equil_loaded = False
            if not equil_loaded:
                numpy.savez(os.path.join(self.source_directory, self.save_prefix + self.phase + '_equil_data_%s.npz' % self.subsample_method), nequil=self.nequil, g_t=self.g_t, Neff_max=Neff_max)
        elif self.nequil is None:
            if self.subsample_method == 'per-state':
                self.nequil = numpy.zeros([self.nstates], dtype=numpy.int32)
                self.g_t = numpy.zeros([self.nstates])
                Neff_max = numpy.zeros([self.nstates])
                for k in xrange(self.nstates):
                    [self.nequil[k], self.g_t[k], Neff_max[k]] = self._detect_equilibration(u_kln[k,k,:])
                    if self.verbose: print "State %i equilibrated with %i samples" % (k, int(Neff_max[k]))
            else:
                [self.nequil, self.g_t, Neff_max] = self._detect_equilibration(self.u_n)

        if self.verbose: print [self.nequil, Neff_max]
        # 1) Discard equilibration data
        # 2) Subsample data to obtain uncorrelated samples
        self.N_k = numpy.zeros(self.nstates, numpy.int32)
        if self.subsample_method == 'per-state':
            # Discard samples
            nsamples_equil = self.niterations - self.nequil
            self.u_kln = numpy.zeros([self.nstates,self.nstates,nsamples_equil.max()])
            for k in xrange(self.nstates):
                self.u_kln[k,:,:nsamples_equil[k]] = u_kln[k,:,self.nequil[k]:]
            #Subsample
            transfer_retained_indices = numpy.zeros([self.nstates,nsamples_equil.max()], dtype=numpy.int32)
            for k in xrange(self.nstates):
                state_indices = timeseries.subsampleCorrelatedData(self.u_kln[k,k,:], g = self.g_t[k])
                self.N_k[k] = len(state_indices)
                transfer_retained_indices[k,:self.N_k[k]] = state_indices
            transfer_kln = numpy.zeros([self.nstates, self.nstates, self.N_k.max()])
            self.retained_indices = numpy.zeros([self.nstates,self.N_k.max()], dtype=numpy.int32)
            for k in xrange(self.nstates):
                self.retained_indices[k,:self.N_k[k]] = transfer_retained_indices[k,:self.N_k[k]] #Memory reduction
                transfer_kln[k,:,:self.N_k[k]] = self.u_kln[k,:,self.retained_indices[k,:self.N_k[k]]].T #Have to transpose since indexing in this way causes issues

            #Cut down on memory, once function is done, transfer_kln should be released
            self.u_kln = transfer_kln
        else:
            #Discard Samples
            self.u_kln = u_kln[:,:,self.nequil:]
            self.u_n = self.u_n[self.nequil:]
            #Subsamples
            indices = timeseries.subsampleCorrelatedData(self.u_n, g=self.g_t) # indices of uncorrelated samples
            self.u_kln = self.u_kln[:,:,indices]
            self.N_k[:] = len(indices)
            self.retained_indices = indices

        self.retained_iters = self.N_k
        return

    def determine_N_k(self, series):
        npoints = len(series)
        #Go backwards to speed up process
        N_k = npoints
        for i in xrange(npoints,0,-1):
            if not numpy.allclose(series[N_k-1:], numpy.zeros(len(series[N_k-1:]))):
                break
            else:
                N_k += -1
        return N_k

    def _presubsampled(self, u_kln):
        #Assume the u_kln is already subsampled
        self.N_k = numpy.zeros(self.nstates, dtype=numpy.int32)
        for k in xrange(self.nstates):
            self.N_k[k] = self.determine_N_k(u_kln[k,k,:])
        maxn = self.N_k.max()
        self.retained_indices = numpy.zeros([self.nstates, maxn])
        for k in xrange(self.nstates):
            self.retained_indices[k,:self.N_k[k]] = numpy.array(range(self.N_k[k]))
        self.u_kln = u_kln
        self.retained_iters = self.N_k
        return

    def _build_u_kln(self, nuse = None, raw_input=False):
        if not raw_input:
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
        else:
            u_kln = self.u_kln_raw

        # Compute total negative log probability over all iterations.
        self.u_n = numpy.zeros([self.niterations], numpy.float64)
        for iteration in range(self.niterations):
            self.u_n[iteration] = numpy.sum(numpy.diagonal(u_kln[:,:,iteration]))

        # Truncate to number of specified conforamtions to use, not really used
        if (nuse):
            u_kln_replica = u_kln_replica[:,:,0:nuse]
            self.u_kln = self.u_kln[:,:,0:nuse]
            self.u_n = self.u_n[0:nuse]

        #!!! Temporary fix
        self.u_kln_raw = u_kln
        
        #Subsample data
        if self.subsample_method is 'presubsampled':
            self._presubsampled(u_kln)
        else:
            self._subsample_kln(u_kln)

        return

    def detect_coupled_bases(self, basis_list):
        """
        This function detemines which state the requested combintation of basis functions is fully coupled.
        Accepts as List of Strings (chars):
        E = Electrostatics
        A = Attractive
        R = Repulsive
        C = Caping basis
        RETURNS:
        integer index of state
        """
        #Create list of states to filter down
        states = numpy.array(range(self.nstates))
        #Find the Intersections
        for basis in basis_list:
            states = numpy.intersect1d(self.coupled_states[basis], states) #Find the state indicies which overlap with where the basis is fully coupled
        return state_index
        #Remove states where all other basis functions are not 0

    def _AutoAlchemyStates(self, phase, real_R_states=None, real_A_states=None, real_E_states=None, real_C_states=None, alchemy_source=None):
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
            real_PMEFull_states = list()
            crunchE = True
        else:
            crunchE = False
        #Detect for the cap basis property
        if numpy.all([hasattr(state, 'ligandCapToFull') for state in protocol]) and real_C_states is None:
            real_C_states = list()
            crunchC = True
        else:
            crunchC = False
        #Import from the alchemy file if need be
        for state in protocol: #Go through each state
            if crunchE:
                real_E_states.append(state.ligandElectrostatics)
                try:
                    real_PMEFull_states.append(state.ligandPMEFull)
                except:
                    real_PMEFull_states.append(None)
            if crunchR:
                real_R_states.append(state.ligandRepulsion)
            if crunchA:
                real_A_states.append(state.ligandAttraction)
            if crunchC:
                real_C_states.append(state.ligandCapToFull)
        if numpy.all([i is None for i in real_PMEFull_states]): #Must put [...] around otherwise it creates the generator object which numpy.all evals to True
            self.PME_isolated = False
        else:
            self.PME_isolated = True
        #Determine cutoffs
        self.real_E_states = numpy.array(real_E_states)
        self.real_PMEFull_states = numpy.array(real_PMEFull_states)
        self.real_R_states = numpy.array(real_R_states)
        self.real_A_states = numpy.array(real_A_states)
        self.real_C_states = numpy.array(real_C_states)
        indicies = numpy.array(range(len(real_E_states)))
        #Determine Inversion
        if numpy.any(self.real_E_states < 0) or numpy.any(numpy.logical_and(self.real_PMEFull_states < 0,numpy.array([i is not None for i in self.real_PMEFull_states]))):
            self.Inversion = True
        else:
            self.Inversion = False
        #Set the indicies, trap TypeError (logical_and false everywhere) as None (i.e. state not found in alchemy)
        if crunchC: #Check for the cap potential
            print "Not Coded Yet!"
            exit(1)
            #Create the Combinations
            basisVars = ["E", "A", "R", "C"]
            mappedStates = [self.real_E_states, self.real_R_states, self.real_C_states, self.real_A_states]
            nBasis = len(basisVars)
            coupled_states = {}
            decoupled_states = {}
            for iBasis in xrange(nBasis):
                coupled_states[basisVars[iBasis]] = numpy.where(mappedStates[iBasis] == 1.00)[0] #need the [0] to extract the array from the basis
                decoupled_states[basisVars[iBasis]] = numpy.where(mappedStates[iBasis] == 0.00)[0]
            self.coupled_states = coupled_states
            self.decoupled_states = decoupled_states
            self.basisVars = basisVars
        else:
            if self.PME_isolated: #Logic to solve for isolated PME case
                try: #Figure out the Fully coupled state
                    self.real_EAR = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == 1, self.real_PMEFull_states == 1), numpy.logical_and(self.real_R_states == 1, self.real_A_states == 1)) ])
                except TypeError:
                    self.real_EAR = None
                try:
                    self.real_AR = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == 0, self.real_PMEFull_states == 0), numpy.logical_and(self.real_R_states == 1, self.real_A_states == 1)) ])
                except TypeError:
                    self.real_AR = None
                try:
                    self.real_R = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == 0, self.real_PMEFull_states == 0), numpy.logical_and(self.real_R_states == 1, self.real_A_states == 0)) ])
                except TypeError:
                    self.real_R = None
                try:
                    self.real_alloff = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == 0, self.real_PMEFull_states == 0), numpy.logical_and(self.real_R_states == 0, self.real_A_states == 0)) ])
                except:
                    self.real_alloff = None
                try:
                    self.real_PMEAR = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == 0, self.real_PMEFull_states == 1), numpy.logical_and(self.real_R_states == 1, self.real_A_states == 1)) ])
                except TypeError:
                    self.real_PMEAR = None
                try:
                    self.real_PMEsolve = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == 0, numpy.logical_and(self.real_PMEFull_states != 1, self.real_PMEFull_states != 0)), numpy.logical_and(self.real_R_states == 1, self.real_A_states == 1)) ])
                except TypeError:
                    self.real_PMEsolve = None
                if self.Inversion:
                    self.real_inverse = int(indicies[ numpy.logical_and(numpy.logical_and(numpy.logical_and(self.real_E_states == -1, self.real_PMEFull_states == -1), self.real_R_states == 1), self.real_A_states==1) ])
            else:
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
                if self.Inversion:
                    self.real_inverse = int(indicies[ numpy.logical_and(numpy.logical_and(self.real_E_states == -1, self.real_R_states == 1), self.real_A_states==1) ])
        #Now that all the sorting and variable assignment has been done, must set the PME states which were not defined to the electrostatic state as thats how its coded (helps sorting algorithm later)
        #This algorighm also ensures that real_PMEFull_states is not dtype=object
        nstates = len(self.real_E_states)
        tempPME = numpy.zeros(nstates)
        for i in xrange(nstates):
            if self.real_PMEFull_states[i] is None: #Find where they are none
                tempPME[i] = self.real_E_states[i] #Assign them equal to the E state
            else:
                tempPME[i] = self.real_PMEFull_states[i]
        self.real_PMEFull_states = tempPME
        return

    def compute_mbar(self):
        if 'method' in self.kwargs:
            method=self.kwargs['method']
        else: 
            method='adaptive'
        if self.mbar_f_ki is not None:
            self.mbar = MBAR(self.u_kln, self.N_k, verbose = self.verbose, method = method, initial_f_k=self.mbar_f_ki, subsampling_protocol=[{'method':'L-BFGS-B','options':{'disp':self.verbose}}], subsampling=1)
        else:
            self.mbar = MBAR(self.u_kln, self.N_k, verbose = self.verbose, method = method, subsampling_protocol=[{'method':'L-BFGS-B','options':{'disp':self.verbose}}], subsampling=1)
        self.mbar_ready = True

    def __init__(self, phase, source_directory, verbose=False, real_R_states = None, real_A_states = None, real_E_states = None, compute_mbar = False, alchemy_source = None, save_equil_data=False, save_prefix="", run_checks=False, nequil=None, subsample_method='all', u_kln_input=None, temp_in=298, mbar_f_ki=None, **kwargs):
        self.phase = phase
        self.verbose = verbose
        if type(save_prefix) is not str: save_prefix = ""
        self.save_prefix = save_prefix
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
        if nequil is not None:
            nequil = int(nequil)
        self.nequil = nequil
        #Subsample method:
        ## 'all' - use a single timeseries to subsample each state uniformly, use for HREX data only
        ## 'per-state' - subsample each state manually. 
        valid_subsamples = ['all', 'per-state', 'presubsampled']
        if subsample_method not in valid_subsamples:
            print "Invalid subsample method '%s', defaulting to 'all'" % subsample_method
            subsample_method = 'all'
        self.subsample_method = subsample_method
        #self.manual_subsample = manual_subsample
        self.mbar_f_ki=mbar_f_ki
        self.kwargs = kwargs
        self.source_directory = source_directory

        if u_kln_input is not None:
            self.u_kln_raw = u_kln_input 
            self.temperature = temp_in * units.kelvin
            self.kT = kB * self.temperature
            self.kcalmolsq = (self.kT / units.kilocalories_per_mole)**2
            self.kcalmol = (self.kT / units.kilocalories_per_mole)
            self.kJmolsq = (self.kT / units.kilojoules_per_mole)**2
            self.kJmol = (self.kT / units.kilojoules_per_mole)
            self.niterations = self.u_kln_raw.shape[2]
            self.nstates = self.u_kln_raw.shape[1]
            #self.natoms = self.ncfile.variables['positions'].shape[2]
            u_n = numpy.zeros([self.niterations], numpy.float64)
            for iteration in range(self.niterations):
                u_n[iteration] = numpy.sum(numpy.diagonal(self.u_kln_raw[:,:,iteration]))
            raw_input = True
        else:
            #Import file and grab the constants
            fullpath = os.path.join(self.source_directory, save_prefix + phase + '.nc')
            if (not os.path.exists(fullpath)): #Check for path
                print save_prefix + phase + ' file does not exsist!'
                print 'Checking for stock ' + phase + '.nc file'
                stockpath = os.path.join(self.source_directory, phase + '.nc')
                if not os.path.exists(stockpath):
                    print 'No NC file found for ' + phase + ' phase!'
                    sys.exit(1)
                else:
                    print 'Default named ' + phase + '.nc file found, using default'
                    fullpath = stockpath
            
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


            if run_checks:
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
            self.u_n = self._extract_u_n(self.ncfile, verbose=self.verbose)
            raw_input = False

        self.save_equil_data = save_equil_data
        self._build_u_kln(raw_input = raw_input)
        # Read reference PDB file.
        if self.phase in ['vacuum', 'solvent']:
            self.reference_pdb_filename = os.path.join(self.source_directory, "ligand.pdb")
        else:
            self.reference_pdb_filename = os.path.join(self.source_directory, "complex.pdb")
        self.atoms = self._read_pdb(self.reference_pdb_filename)

        if compute_mbar:
            self.compute_mbar()
        else:
            self.mbar_ready = False
        self.expected_done = False
        
        return

