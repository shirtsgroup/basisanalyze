"""
Class: LinFunctions
Custom Linear basis functions that I can just import
"""
import numpy
from scipy import linspace
from scipy.interpolate import UnivariateSpline as US
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

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
        #Comment next 3 lines to make purely monotonic
        for j in m_to_mono:
           tangents[j] = 0
           tangents[j+1] = 0
        #Build the monotonics
        for i in m_to_change:
            #check to see if i is in m_to_mono and dont touch it if it is
            #Comment next 3 lines to make purley monotonic
            if i in m_to_mono:
                continue
            else:
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
            x_to_hermite = linspace(x[i],x[i+1],n_between+2)
            (yr_hermite_out,dyr_herm_out) = self._hermite_spline(x_to_hermite,y[i],y[i+1],m[i],m[i+1])
            x_filled = numpy.append(x_filled[:-1], x_to_hermite)
            yr_filled = numpy.append(yr_filled[:-1], yr_hermite_out)
            dyr_filled = numpy.append(dyr_filled[:-1], dyr_herm_out)
        return x_filled,yr_filled,dyr_filled
    #---------------------------------------------------------------------------------------------
    def _assignLinear(self, S, dS, ddS):
        S = lambda L: L
        dS = lambda L: 1
        ddS = lambda L: 0
        return
    def _assignLinA(self, S, dS, ddS, C):
        self.h_r_const = C #Constant for Naden's H_R(lambda)
        S = lambda L: (self.h_r_const**L - 1)/(self.h_r_const - 1)
        dS = lambda L: (numpy.log(self.h_r_const)*self.h_r_const**L)/(self.h_r_const-1)
        ddS = lambda L: ((numpy.log(self.h_r_const)**2)*self.h_r_const**L)/(self.h_r_const-1)
        return
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
        x = linspace(0,1,501)
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
        self.h_r_inv = lambda h: h
        self.h_a_inv = lambda h: h
        self.h_e_inv = lambda h: h

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
        self.h_a_inv = lambda h: h

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
        self.h_a_inv = lambda h: h
    
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
        
    def _unboxOptimal(self, C1=1.61995584, C2=-0.8889962, C3=0.02552684, **kwargs):
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
        x = linspace(0,1,501)
        y = self.h_r(x)
        self.h_r_inv = IUS(y,x)
        self.h_a_inv = lambda h: h
        self.h_e_inv = lambda h: h

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

    def _unboxHermiteOptimal(self):
        #This is a special constructor that was built from an optimization routine
        Npoints=5
        x_herm = numpy.concatenate( (linspace(0,0.3,Npoints), linspace(.3,1,Npoints)[1:]) )
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
        self.h_e_inv = lambda h: h
    def _unboxInversion(self, **kwargs):
        subbasis = LinFunctions(method = self.submethod, **kwargs)
        inversion = lambda F: (2*F - 1)
        inv_inversion = lambda R: (R + 1)/2
        dinversion = lambda dF: (2*dF)
        d2inversion = lambda d2F: (2*d2F)
        self.h_r = lambda L: inversion(subbasis.h_r(L))
        self.dh_r = lambda L: dinversion(subbasis.dh_r(L))
        self.h_a = lambda L: inversion (subbasis.h_a(L))
        self.dh_a = lambda L: dinversion(subbasis.dh_a(L))
        self.h_e = lambda L: inversion(subbasis.h_e(L))
        self.dh_e = lambda L: dinversion(subbasis.dh_e(L))
        #self.h_r_inv = lambda h: inv_inversion(subbasis.h_r_inv(h))
        #self.h_a_inv = lambda h: inv_inversion(subbasis.h_a_inv(h))
        #self.h_e_inv = lambda h: inv_inversion(subbasis.h_e_inv(h))
        self.h_r_inv = lambda h: subbasis.h_r_inv(inv_inversion(h))
        self.h_a_inv = lambda h: subbasis.h_a_inv(inv_inversion(h))
        self.h_e_inv = lambda h: subbasis.h_e_inv(inv_inversion(h))
    def _piecewiseSplitH(self, Lin, scaling, step):
        #Quickest way i could think of to handle single numbers and arrays
        L = numpy.empty(0)
        L = numpy.append(L,Lin)
        #3 conditions: L below range, L above range, L in range
        scaledL = scaling*L-step
        #Below range
        low = numpy.where(scaledL < 0)
        #Above range
        high = numpy.where(scaledL >1)
        scaledL[low] = 0
        scaledL[high] = 1
        return scaledL 
    def _piecewiseSplitdH(self, Lin, scaling, step):
        L = numpy.empty(0)
        L = numpy.append(L,Lin)
        #3 conditions: L below range, L above range, L in range
        scaledL = scaling*L-step
        dscaledL = numpy.array([scaling]*L.size)
        zeros = numpy.logical_or(scaledL < 0, scaledL>1)
        dscaledL[zeros] = 0
        return dscaledL
    def _noneto1(self, x):
        if x is None:
            return 1
        else:
            return x
    def _unboxSplit(self, **kwargs):
        self.base_basis = LinFunctions(method=self.baselinebasis, **kwargs) #Default basis method to obey
        #Determine ordering
        self.orderE, self.orderA, self.orderR = (self._noneto1(self.orderE), self._noneto1(self.orderA), self._noneto1(self.orderR))
        orderEAR = numpy.array([self.orderE, self.orderA, self.orderR])
        if not 1 in orderEAR:
            print "Must have some force be first to decouple!"
            raise
        if numpy.any(orderEAR > 3):
            print "Only entries up to 3 are allowed in coupling order"
            raise
        if 3 in orderEAR and not 2 in orderEAR:
            print "You have a 3rd step without a 2nd step..."
            raise
        nsteps = orderEAR.max()
        if nsteps > 1:
            self.multiple_lam = True
        else:
            self.multiple_lam = False
        inverseh = lambda h, scale, step: (h + step)/scale
        #Do steps
        self.h_r = lambda L: self.base_basis.h_r(self._piecewiseSplitH(L, nsteps, self.orderR-1))
        self.dh_r = lambda L: self._piecewiseSplitdH(L,nsteps,self.orderR-1) * self.base_basis.dh_r(self._piecewiseSplitH(L, nsteps, self.orderR-1))
        self.h_r_inv = lambda h: inverseh(self.base_basis.h_r_inv(h), nsteps, self.orderR-1)
        self.h_a = lambda L: self.base_basis.h_a(self._piecewiseSplitH(L, nsteps, self.orderA-1))
        self.dh_a = lambda L: self._piecewiseSplitdH(L,nsteps,self.orderA-1) * self.base_basis.dh_a(self._piecewiseSplitH(L, nsteps, self.orderA-1))
        self.h_a_inv = lambda h: inverseh(self.base_basis.h_a_inv(h), nsteps, self.orderA-1)
        self.h_e = lambda L: self.base_basis.h_e(self._piecewiseSplitH(L, nsteps, self.orderE-1))
        self.dh_e = lambda L: self._piecewiseSplitdH(L,nsteps,self.orderE-1) * self.base_basis.dh_e(self._piecewiseSplitH(L, nsteps, self.orderE-1))
        self.h_e_inv = lambda h: inverseh(self.base_basis.h_e_inv(h), nsteps, self.orderE-1)


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

        elif self.method is 'HermiteGeneral':
            if 'lam_range' in kwargs:
                lam_range=kwargs['lam_range']
            else:
                lam_range=linspace(0,1,11)
            #if 'interm_n' in kwargs:
            #    interm_n=kwargs['interm_n']
            #else:
            #    interm_n=2
            if 'fullg_r' in kwargs:
                fullg_r = kwargs['fullg_r']
            else:
                fullg_r = linspace(0,1,len(lam_range))
            if 'fullg_a' in kwargs:
                fullg_a = kwargs['fullg_a']
            else:
                fullg_a = linspace(0,1,len(lam_range))
            if 'fullg_e' in kwargs:
                fullg_e = kwargs['fullg_e']
            else:
                fullg_e = linspace(0,1,len(lam_range))
            self._unboxHermiteGeneral(lam_range, fullg_r, fullg_a, fullg_e)

        elif self.method is 'HermiteOptimal':
            #Special case that manipulates the hermite splines
            self._unboxHermiteOptimal()

        elif self.method is 'Inversion':
            if 'submethod' in kwargs:
                self.submethod = kwargs['submethod']
            else:
                self.submethod = 'PureLin'
            self._unboxInversion(**kwargs)

        elif self.method is 'SplitTerms':
            if 'baselinebasis' in kwargs:
                self.baselinebasis = kwargs['baselinebasis']
            else:
                self.baselinebasis = 'Optimal'
            #Determine sequence of entries. Higher order means that force is fully coupled FIRST (i.e., lower lambda)
            # Must have at least 1 entry
            if 'orderE' in kwargs:
                self.orderE = kwargs['orderE']
            else:
                self.orderE = None
            if 'orderA' in kwargs:
                self.orderA = kwargs['orderA']
            else:
                self.orderA = None
            if 'orderR' in kwargs:
                self.orderR = kwargs['orderR']
            else:
                self.orderR = None
            self._unboxSplit(**kwargs)

        else:
            print "No valid basis method selected!"
            raise 

                
        return

