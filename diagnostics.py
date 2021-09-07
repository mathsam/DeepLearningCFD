import numpy as np
import scipy.fftpack as fftpack
import scipy.io

"""
deals with output from qg_model using NetCDF output
assume the field has shape as

    psi(time_step (optional), real_and_imag, ky, kx, z)
"""

def get_vorticity(psik):
    """
    Calculates spectral relative vorticity from spectral streamfunction field
    psik
    @param psik stream function in spectral space
                returned from real2complex, complex numpy array
                with shape (time (optional), ky, kx, z(optional))
    @return zetak relative vorticity in spectral space
    """
    return laplacian(psik, 1)

def laplacian(psik, order=1):
    """
    Calculate laplacian to the nth order
    
    Args:
        psik: complex field with shape (time (optional), ky, kx, z(optional))
        order: evaluate \nabla^(2*order)
            default is the normal lapacian operator
    
    Return:
        zetak: complex spectrum
    """
    num_zdim = not _is_single_layer(psik)
    ky_pos = -2 - num_zdim
    kmax = psik.shape[ky_pos] - 1

    kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
    k2 = kx_**2 + ky_**2
    k2.shape = (1,)*(psik.ndim + ky_pos) + (kmax+1, 2*kmax+1) + (1,)*num_zdim
    zetak = psik * (-k2)**order
    return zetak
    
def jacobian(Ak, Bk):
    """
    Calculate J(Ak, Bk)
    
    Args:
        Ak, Bk: complex fields with shape (time(optional), ky, kx)
    
    Return:
        Jacob: in physical space
    """
    Dx_Ak = partial_x(Ak)
    Dx_Bk = partial_x(Bk)
    Dy_Ak = partial_y(Ak)
    Dy_Bk = partial_y(Bk)
    Dx_Ag = spec2grid(Dx_Ak)
    Dx_Bg = spec2grid(Dx_Bk)
    Dy_Ag = spec2grid(Dy_Ak)
    Dy_Bg = spec2grid(Dy_Bk)
    return Dx_Ag*Dy_Bg - Dy_Ag*Dx_Bg
    
    
def get_PV(psik, F):
    """
    Calculate spectrum of potential vorticity given spectrum of streamfunction
    
    Args:
        psik: stream function in spectral space, returned from real2complex.
              shape is (time_step(optional), ky, kx, z)
        F: F param in model
        
    Returns:
        pvk: potential vorticity spectrum. same shape as psik
        
    Assume linear profile therefore dz = 1/nz. For two layer model considered
    here, thus dz = 0.5.
    Refers to Vallis (2006), page 223, Eq. (5.137). F = f0^2/g'. H1 and H2
    correspond dz.
    """
    if _is_single_layer(psik) or psik.shape[-1] != 2:
        raise TypeError('stream function must have two layers')
    kmax = psik.shape[-3] - 1
    kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
    k2 = kx_**2 + ky_**2
    k2.shape = (1,)*(psik.ndim-3) + psik.shape[-3:-1]
    
    pvk = np.empty_like(psik)
    pvk[..., 0] = -k2*psik[..., 0] + 2*F*(psik[..., 1] - psik[..., 0])
    pvk[..., 1] = -k2*psik[..., 1] + 2*F*(psik[..., 0] - psik[..., 1])
    return pvk
    
def prod_domain_ave_int(field1, field2):
    """
    Calculate domain averaged integral of product (field1*field2)
    Do the integral in spectrum space as
        integral(field1*field2) = 4*pi^2*sum(Re(field1*conj(field2))), where
        field1 and field2 are returned real2complex (only contains half of the
        spectrum coefficient (ky>0 parts)
        
        So the domained averaged integral is sum(Re(field1*conj(field2)))
        
    Args:
        field1, field2: numpy array returned from real2complex
                        shape is (time_step, ky, kx, z(optional))
        
    Returns:
        prod_int: numpy array, integral for each time step in each layer
                  shape is (time_step, z(optional))
    """
    if field1.shape != field2.shape:
        raise TypeError("field1 and field2 don't have same shape")
    prod_int = np.real(field1*np.conj(field2))
    if len(field1.shape) == 4:
        prod_int = np.sum(np.sum(prod_int, -2), -2)
    elif len(field1.shape) == 3:
        prod_int = np.sum(np.sum(prod_int, -1), -1)
    else:
        raise NotImplementedError("unknown shape")
    return 2*prod_int
    
def prod_spectrum(field1, field2):
    """
    Spectrum of product field1*field2. Intend for flux v'*\tau
    
    Args:
        field1, field2: spec field with shape
                        (time(optional), ky, kx, z(optional))
    """
    if field1.shape != field2.shape:
        raise TypeError('field1 and field2 have different shapes')
    
    prod2d = np.real(field1*np.conj(field2))
    if not _is_single_time(field1.shape):
        prod2d_ave = np.mean(prod2d, 0)
    else:
        prod2d_ave = prod2d
    
    if _is_single_layer(field1):
        nky, nkx = field1.shape[-2:]
    else:
        nky, nkx = field1.shape[-3:-1]
    kmax = nky - 1
    ksqd_ = np.zeros((nky, nkx), dtype=float)
    for j in range(0, nky):
        for i in range(0, nkx):
            ksqd_[j,i] = (i-kmax)**2 + j**2
    radius_arr = np.floor(np.sqrt(ksqd_)).astype(int)

    if _is_single_layer(field1):
        spec1d = np.zeros(kmax)
    else:
        spec1d = np.zeros((kmax, field1.shape[-1]))
    for i in range(0,kmax):
        spec1d[i,...]   = np.sum(prod2d_ave[radius_arr == i+1,...], 0)
    return np.arange(1,kmax+1), 2*spec1d.squeeze()
    
def prod_spectrum_zonal(field1, field2):
    """
    Spectrum of product field1*field2. Sum over zonal wavenumber kx
    
    Args:
        field1, field2: complex spec fields with shape 
            (time(optional), ky, kx, z(optional))
        
    Return:
        kxs: 1 to kmax (511 in most of my simulations)
        spec: 1d array
    """
    if field1.shape != field2.shape:
        raise TypeError('field1 and field2 have different shapes')

    prod2d = np.real(field1*np.conj(field2))
    if not _is_single_time(field1.shape):
        prod2d_ave = np.mean(prod2d, 0)
    else:
        prod2d_ave = prod2d
    prod1d_ave = np.sum(prod2d_ave, 0)
    
    if _is_single_layer(field1):
        nky, nkx = field1.shape[-2:]
    else:
        nky, nkx = field1.shape[-3:-1]
    kmax = nky - 1
    spec1d = (prod1d_ave + prod1d_ave[::-1])[-kmax:]
    return np.arange(1,kmax+1), 2*spec1d.squeeze()

def prod_spectrum_meridional(field1, field2):
    """
    Spectrum of product field1*field2. Sum over zonal wavenumber ky
    
    Args:
        field1, field2: complex spec fields with shape 
            (time(optional), ky, kx, z(optional))
        
    Return:
        kys: 1 to kmax (511 in most of my simulations)
        spec: 1d array
    """
    if field1.shape != field2.shape:
        raise TypeError('field1 and field2 have different shapes')

    prod2d = np.real(field1*np.conj(field2))
    if not _is_single_time(field1.shape):
        prod2d_ave = np.mean(prod2d, 0)
    else:
        prod2d_ave = prod2d
    prod1d_ave = np.sum(prod2d_ave, 1)
    
    if _is_single_layer(field1):
        nky, nkx = field1.shape[-2:]
    else:
        nky, nkx = field1.shape[-3:-1]
    kmax = nky - 1
    return np.arange(1,kmax+1), 2*prod1d_ave[1:].squeeze()

def hypervis_filter(kmax, filter_tune=1.0, filter_exp=4, dealiasing='isotropic', 
                    filter_type='hyperviscous', Nexp=1.0):
    """Return the hyperviscous filter in use for the spectrum
    
    Args:
        kmax: interger. For example 511 for resolution 1024x1024
        filter_tune: filter parameter. Default is 1.0
        filter_exp: integer. Corresponds to del^(2*filter_exp)
        dealiasing: 'isotropic'|'orszag'
        filter_type: 'hyperviscous' is currently only one implemented
        Nexp: 1(default)|2, SUQG requires Nexp=2
    
    Return:
        filter2d: 2d array with shape (ky, kx)
    
    In the model q = filter2d*q is applied before the next time stepping
    """
    kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
    k2 = kx_**2 + ky_**2
    kmax_da2 = 8./9. * (kmax+1)**2
    ngrid = 2*(kmax+1)
    
    if filter_type == 'hyperviscous':
        filter2d = 1./(1 + filter_tune * (4*np.pi/ngrid**Nexp) 
                                       * (k2/kmax_da2)**filter_exp)
    else:
        raise NotImplementedError('other filter types than hyperviscous not implemented')
                                   
    if dealiasing == 'isotropic':
        filter2d[k2 >= 8./9.*(kmax+1)**2] = 0
    elif dealiasing == 'orszag':
        filter2d[k2 >= 4./3.*(kmax+1)**2] = 0
    else:
        raise NotImplementedError('dealiasing type not implemented')
    return filter2d

def hypervis_filter_rate(kmax, dt, filter_tune=1.0, filter_exp=4, dealiasing='orszag', 
                    filter_type='hyperviscous', Nexp=1.0):
    """Return the hyperviscous filter rate in use for the spectrum
    
    Args:
        kmax: interger. For example 511 for resolution 1024x1024
        dt: time step
        filter_tune: filter parameter, default is 1.0
        filter_exp: integer. Corresponds to del^(2*filter_exp)
        dealiasing: 'isotropic'|'orszag'
        filter_type: 'hyperviscous' is currently only one implemented
        Nexp: 1(default)|2, SUQG requires Nexp=2
    
    Return:
        filter_rate_2d: 2d array with shape (ky, kx)
    
    The effect of filter:
        dq/dt = ... + filter_rate_2d*q
    """
    kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
    k2 = kx_**2 + ky_**2
    kmax_da2 = 8./9. * (kmax+1)**2
    ngrid = 2*(kmax+1)
    
    if filter_type == 'hyperviscous':
        filter2d = filter_tune*(4*np.pi/ngrid**Nexp) * (k2/kmax_da2)**filter_exp
    else:
        raise NotImplementedError('other filter types than hyperviscous not implemented')

    if dealiasing == 'isotropic':
        filter2d[k2 >= 8./9.*(kmax+1)**2] = 0
    elif dealiasing == 'orszag':
        filter2d[k2 >= 4./3.*(kmax+1)**2] = 0
    else:
        raise NotImplementedError('dealiasing type not implemented')
    return (-1)**(filter_exp+1) * filter2d/(2*dt)

def time_step(qs, dt_tune, beta, kmax):
    """get adapted time step fro enstrophy field (in physical space)
    
    Args:
        qs: pv spectrum with shape (time(optional), ky, kx, z(optional))
        dt_tune: tuing parameter
        beta: beta parameter
        kmax: reso parameter
    
    Return:
        dt: a float or numpy array
    """
    if _is_single_layer(qs):
        nz = 1
    else:
        nz = qs.shape[-1]
    if _is_single_time(qs):
        enstrophy = np.sum(np.real(qs*np.conj(qs)))/nz
        dt = dt_tune*np.pi/(kmax*np.sqrt(np.max([enstrophy, beta, 1.0])))
    else:
        dt = np.zeros(qs.shape[0])
        for i in range(len(dt)):
            enstrophy = np.sum(np.real(qs[i]*np.conj(qs[i])))/nz
            dt[i] = dt_tune*np.pi/(kmax*np.sqrt(np.max([enstrophy.max(), beta, 1.0])))
    return dt

def get_betay(pvg, beta):
    """
    Given a potential vorticity field in physical space, return the beta*y that
    needs to be added to it
    
    Args:
        pvg: a numpy array with shape (time_step(optional), ny, nx, nz)
        beta: a float
        
    Returns:
        beta_y: a numpy array with shape (1 (optional), ny, 1, 1)
    """
    ny = pvg.shape[-3]
    beta_y = beta*np.linspace(-np.pi, np.pi, ny)
    beta_y.shape = (1,)*(pvg.ndim-3) + (ny, 1, 1)
    return beta_y
    

def get_velocities(psik):
    """
    Get velocitie fields from spectral stream function
    
    Args:
        psik: spectral stream function field returned from real2complex
              (time_step(optional), ky, kx, z(optional))
        
    Returns:
        (u, v): a tuple of zonal and meridional velocities spectral field
        
    Raises:
        TypeError: input doesn't seem to be spectral psi field
    """
    if not _is_single_layer(psik):
        kmax = psik.shape[-3] - 1
        if kmax%2 == 0:
            raise TypeError('This is probably nnot a SPECTRAL psi field')
            
        kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
        kx_.shape = (1,)*(psik.ndim-3) + psik.shape[-3:-1] + (1,)
        ky_.shape = (1,)*(psik.ndim-3) + psik.shape[-3:-1] + (1,)
        uk = -1j*ky_*psik
        vk =  1j*kx_*psik
        return uk, vk
    else:
        kmax = psik.shape[-2] - 1
        if kmax%2 == 0:
            raise TypeError('This is probably nnot a SPECTRAL psi field')
            
        kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
        kx_.shape = (1,)*(psik.ndim-2) + psik.shape[-2:]
        ky_.shape = (1,)*(psik.ndim-2) + psik.shape[-2:]
        uk = -1j*ky_*psik
        vk =  1j*kx_*psik
        return uk, vk
    
def partial_x(spec_field):
    """
    Calculate the partial derivative with respect to x
    
    Args:
        spec_field: spectral field with shape (time_step(optional), ky, kx, z(optional))
        
    Returns:
        dfield_dx: spectral field with the same shape as input
    """
    if not _is_single_layer(spec_field):
        kmax = spec_field.shape[-3] - 1
        if kmax%2 == 0:
            raise TypeError('This is probably nnot a SPECTRAL psi field')
    
        kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
        kx_.shape = (1,)*(spec_field.ndim-3) + spec_field.shape[-3:-1] + (1,)
        dfield_dx = 1j*kx_*spec_field
        return dfield_dx
    else:
        kmax = spec_field.shape[-2] - 1
        if kmax%2 == 0:
            raise TypeError('This is probably nnot a SPECTRAL psi field')
        kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
        kx_.shape = (1,)*(spec_field.ndim-2) + spec_field.shape[-2:]
        dfield_dx = 1j*kx_*spec_field
        return dfield_dx
    
def partial_y(spec_field):
    """
    Calculate the partial derivative with respect to y
    
    Args:
        spec_field: spectral field with shape (time_step(optional), ky, kx, z(optional))
        
    Returns:
        dfield_dy: spectral field with the same shape as input
    """
    if not _is_single_layer(spec_field):
        kmax = spec_field.shape[-3] - 1
        if kmax%2 == 0:
            raise TypeError('This is probably nnot a SPECTRAL psi field')
    
        kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
        ky_.shape = (1,)*(spec_field.ndim-3) + spec_field.shape[-3:-1] + (1,)
        dfield_dy = 1j*ky_*spec_field
        return dfield_dy
    else:
        kmax = spec_field.shape[-2] - 1
        if kmax%2 == 0:
            raise TypeError('This is probably nnot a SPECTRAL psi field')
    
        kx_, ky_ = np.meshgrid(range(-kmax, kmax+1), range(0, kmax+1))
        ky_.shape = (1,)*(spec_field.ndim-2) + spec_field.shape[-2:]
        dfield_dy = 1j*ky_*spec_field
        return dfield_dy
    
def real2complex(rfield):
    """
    convert raw qg_model output to complex numpy array
    suppose input has shape
        psi(time_step (optional), real_and_imag, ky, kx, z(optional))
    """
    if rfield.shape[-2]+1 == 2*rfield.shape[-3]:
        return rfield[...,0,:,:,:]+1j*rfield[...,1,:,:,:]
    elif rfield.shape[-1]+1 == 2*rfield.shape[-2]:
        return rfield[...,0,:,:]+1j*rfield[...,1,:,:]
    else:
        raise NotImplementedError('Unrecognized field type')

def fullspec(hfield, single_layer=False):
    """
    Assumes 'hfield' to contain upper-half plane of spectral field, 
    and specifies lower half plane by conjugate 
    symmetry (since physical field is assumed real-valued).  'hfield'
    should have shape (...,kmax+1,2*kmax+1,nz), kmax = 2^n-1,
    hence physical resolution will be 2^(n+1) x 2^(n+1) x nz.  
    NOTE:  The top row of the input field corresponds to ky = 0,
    the kx<0 part is NOT assumed a priori to be conjugate-
    symmetric with the kx>0 part.
    
    Args:
        sfield: complex spectrum field with shape (t(optional), ky, kx, z)
        single_layer: True|False. If True, spec_field's shape is
            (time_step(optional), ky, kx)
    """
    if not isinstance(hfield, np.ndarray):
        raise TypeError("input needs to be numpy array")
    if hfield.ndim < 2:
        raise ValueError("array must be at least 2 dimensional")
    
    if not single_layer:
        nky, nkx = hfield.shape[-3:-1]
    
        if nkx+1 != 2*nky:
            raise ValueError("hfield must have dim (..., kmax+1, 2*kmax+1)")
        hres = nkx + 1
        kmax = nky - 1
        fk = np.zeros(hfield.shape[:-3]+(hres,hres)+(hfield.shape[-1],), 
                    dtype=complex)
        
        fup = np.copy(hfield)
        fup[...,0,kmax-1::-1,:] = fup.conj()[...,0,kmax+1:,:]
        #fup[...,0,kmax,:] = 0. # a littile confused whether should do this
        fdn = np.copy(fup.conj()[..., nky-1:0:-1, nkx-1::-1,:])
        fk[..., nky:, 1:,:] = fup
        fk[...,1:nky, 1:,:] = fdn
        return fk
    else:
        nky, nkx = hfield.shape[-2:]

        if nkx+1 != 2*nky:
            raise ValueError("hfield must have dim (..., kmax+1, 2*kmax+1)")
        hres = nkx + 1
        kmax = nky - 1
        fk = np.zeros(hfield.shape[:-2]+(hres,hres), dtype=complex)
        
        fup = np.copy(hfield)
        fup[...,0,kmax-1::-1] = fup.conj()[...,0,kmax+1:]
        #fup[...,0,kmax,:] = 0. # a littile confused whether should do this
        fdn = np.copy(fup.conj()[..., nky-1:0:-1, nkx-1::-1])
        fk[..., nky:, 1:] = fup
        fk[...,1:nky, 1:] = fdn
        return fk

def _is_single_time(shape):
    """determine if the `shape` corresponds to single time or multiple time 
    array
    Single time would be (ky, kx, z(optional)
    Multiple time would be (time, ky, kx, z(optional)
    
    Args:
        shape: tuple or numpy array. If tuple, assume to be the shape of spectral
               field
    """
    if isinstance(shape, tuple):
        res_ndims = len(shape) - (not _is_single_layer(shape))
        if res_ndims == 3:
            return False
        elif res_ndims == 2:
            return True
        else:
            raise ValueError('cannot determine is single time or not')
    elif isinstance(shape, np.ndarray):
        res_ndims = shape.ndim - (not _is_single_layer(shape))
        if res_ndims == 3:
            return False
        elif res_ndims == 2:
            return True
        else:
            raise ValueError('cannot determine is single time or not')
    else:
        raise TypeError('unrecognized type')

def _is_single_layer(shape):
    """determine the given `shape` tuple or numpy array field corresponds to 
    single layer or multiple layers.
    For complex field or tuple:
        Single layer would be (time(optional), ky, kx)
        Multi-layers would be (time(optional), ky, kx, z)
    For real field:
        Single layer would be (time(optional), y, x)
        Multi-layers would be (time(optional), y, x, z)
    The difference is just that whether the last dimension correponds to z
    
    Args:
        shape: a tuple consists of integers with len to be at least 2
               or a numpy array
        
    Return:
        True|False
    
    Exception:
        ValueError: cannot determine
    
    Note:
        whether z-dim exist is determined from the relation kx = 2*ky - 1 or
        nx = ny
    """
    if isinstance(shape, np.ndarray) and np.iscomplexobj(shape):
        shape = shape.shape
    if isinstance(shape, tuple):
        if len(shape) < 2:
            raise ValueError('shape should have at least 2 dimensions')
        if len(shape) == 2:
            if shape[0]*2 - 1 == shape[1]:
                return True
            else:
                raise ValueError('does not seem to be a spectral field')
        n1, n2, n3 = shape[-3:]
        n1_n2_is_ky_kx = (n1*2 -1 == n2)
        n2_n3_is_ky_kx = (n2*2 -1 == n3)
        if n1_n2_is_ky_kx and not n2_n3_is_ky_kx:
            return False
        if n2_n3_is_ky_kx and not n1_n2_is_ky_kx:
            return True
        raise ValueError('Unable to determine whether single layer or not')
    if isinstance(shape, np.ndarray) and np.isrealobj(shape):
        shape = shape.shape
        if len(shape) < 2:
            raise ValueError('shape should have at least 2 dimensions')
        if len(shape) == 2:
            if shape[0] == shape[1] and np.log2(shape[0]) == int(np.log2(shape[0])):
                return True
            else:
                raise ValueError('does not seem to be a physical field')
        n1, n2, n3 = shape[-3:]
        n1_n2_is_y_x = (n1 == n2) and (np.log2(n1) == int(np.log2(n1)))
        n2_n3_is_y_x = (n2 == n3) and (np.log2(n2) == int(np.log2(n2)))
        if n1_n2_is_y_x and not n2_n3_is_y_x:
            return False
        if n2_n3_is_y_x and not n1_n2_is_y_x:
            return True
        raise ValueError('Unable to determine whether single layer or not')
    raise TypeError('Unrecogonized type')

def spec2grid(sfield):
    """
    Transform one frame of SQG model
    output to a grided (physical) representation.  Assumes 'sfield'
    to be up-half plane, and specifies lower half plane by conjugate
    sym (since physical field is assumed real-valued).  Input field
    should have dimensions  (...,kmax+1,2*kmax+1,nz), where
    kmax=2^n-1, hence physical resolution will be 2^(n+1) x 2^(n+1).
    NOTE: top row of the input field corresponds to ky = 0, the
    kx<0 part is NOT assumed a priori to be conjugate- symmetric
    with the kx>0 part.  NOTE: grid2spec(spec2grid(fk)) = fk.
    OPTIONAL: da = true pads input with 0s before transfoming to
    gridspace, for dealiased products.  Default is da = false.
    
    Args:
        sfield: complex spectrum field with shape (t(optional), ky, kx, z(optional))
    """        
    if not _is_single_layer(sfield):
        hres = sfield.shape[-2] + 1
        fk = fullspec(sfield)
        fk = fftpack.ifftshift(fk, axes=(-2,-3))
        return hres*hres*np.real(fftpack.ifft2(fk, axes=(-2,-3)))
    else:
        hres = sfield.shape[-1] + 1
        fk = fullspec(sfield, True)
        fk = fftpack.ifftshift(fk, axes=(-1,-2))
        return hres*hres*np.real(fftpack.ifft2(fk, axes=(-1,-2)))

def grid2spec(gfield):
    """Transform a field on physical grid to spectral space
    
    Args:
        gfield: numpy array with shape (time(optional), y, x, z(optional))
    """
    if _is_single_layer(gfield):
        y_loc = -2
    else:
        y_loc = -3
    ny, nx = gfield.shape[y_loc], gfield.shape[y_loc+1]
    fk = fftpack.fft2(gfield, axes=(y_loc+1, y_loc))/ny/nx
    fk = fftpack.fftshift(fk, axes=(y_loc+1, y_loc))
    kmax = ny//2 - 1
    if _is_single_layer(gfield):
        sfield = fk[..., kmax+1:,1:]
        sfield[...,0,:kmax] = 0
        return sfield.copy()
    else:
        sfield = fk[..., kmax+1:,1:,:]
        sfield[...,0,:kmax,:] = 0
        return sfield.copy()

def energy_spec(psic):
    """
    Given stream function at one layer of one time step, returns its energy 
    spectrum
    @param psic complex numpy array
                returned from real2complex, has shape (time (optional), ky, kx)
    @return (wavenumber, Ek, EKEk) 1d numpy array
    """
    if not hasattr(energy_spec, 'precal') \
       or (energy_spec.nky, energy_spec.nkx) != psic.shape:
        nky, nkx = psic.shape[-2:]
        energy_spec.nkx   = psic.shape[-1]
        energy_spec.nky   = psic.shape[-2]
        energy_spec.ksqd_ = np.zeros((nky, nkx), dtype=float)
        
        kmax = energy_spec.nky - 1
        for j in range(0, nky):
            for i in range(0, nkx):
                energy_spec.ksqd_[j,i] = (i-kmax)**2 + j**2
        
        radius_arr = np.floor(np.sqrt(energy_spec.ksqd_)).astype(int)
        energy_spec.radius_mask = []
        for i in range(0, kmax):
            energy_spec.radius_mask.append(radius_arr == i+1)
        #print "precalculation done"
    
    kmax = energy_spec.nky - 1
    indx_kx0 = (energy_spec.nkx-1)/2
    KE2d  = np.zeros((energy_spec.nky, energy_spec.nkx), dtype=float)
    Ek    = np.zeros(kmax,       dtype=float)
    EKEk  = np.zeros(kmax,       dtype=float)

    if psic.ndim == 3:
        for i in range(0, psic.shape[0]):
            KE2d += np.abs(psic[i,...]*psic[i,...].conj())
        KE2d *= energy_spec.ksqd_
        KE2d /= psic.shape[0]
    else:
        KE2d  = energy_spec.ksqd_*np.abs(psic*psic.conj())

    for i in range(0,kmax):
        Ek[i]   = np.sum(KE2d[energy_spec.radius_mask[i]])
    
    KE2d[:,indx_kx0] = 0.

    for i in range(0,kmax):
        EKEk[i]   = np.sum(KE2d[energy_spec.radius_mask[i]])

    return np.arange(1,kmax+1), Ek, EKEk
    
def _barotropic_Ek_ncchain(psi):
    """
    process NetCDFChain object with lots of files
    @param psi has shape (time, real_and_imag, ky, kx, z)
    """
    if not isinstance(psi, nc_tools.NetCDFChain):
        raise TypeError("Not NetCDFChain object")
    nky, nkx = psi.shape[-3:-1]
    ksqd_    = np.zeros((nky, nkx), dtype=float)
    kmax     = nky - 1
    indx_kx0 = (nkx-1)/2
    KE2d  = np.zeros((nky, nkx), dtype=float)
    Ek    = np.zeros(kmax,       dtype=float)
    EKEk  = np.zeros(kmax,       dtype=float)
    
    for j in range(0, nky):
        for i in range(0, nkx):
            ksqd_[j,i] = (i-kmax)**2 + j**2
        
    radius_arr = np.floor(np.sqrt(ksqd_)).astype(int)

    for i, dt in enumerate(psi.time_steps_each_file):
        #reading each file all together; if read in each time step each time
        #reading files will take too much time
        psi_seg = psi[psi.time_steps_before[i]:psi.time_steps_before[i]+dt]
        if psi_seg.ndim != 5:
            raise TypeError("file does not have correct number of dimensions")
        psi_seg = np.mean(psi_seg, psi_seg.ndim-1) #barotropic field
        KE2d += np.mean(np.sum(psi_seg*psi_seg, 1), 0)
        
    KE2d *= ksqd_
    KE2d /= len(psi.sorted_files)

    for i in range(0,kmax):
        Ek[i]   = np.sum(KE2d[radius_arr == i+1])
    
    KE2d[:,indx_kx0] = 0.

    for i in range(0,kmax):
        EKEk[i]   = np.sum(KE2d[radius_arr == i+1])

    return np.arange(1,kmax+1), Ek, EKEk
    
    
def barotropic_Ek(psic):
    """
    barotropic energy spectrum for multiple times and multiple layers
    @param psic complex numpy array
                returned from real2complex, has shape 
                (time (optional), ky, kx, z)
            OR  NetCDFChain object
                has shape (time, real_and_imag, ky, kx, z)
    @return (wavenumber, Ek, EKEk) 1d numpy array
    """
    if isinstance(psic, scipy.io.netcdf.netcdf_variable):
        psic = psic[:]
        psic = real2complex(psic)
    if isinstance(psic, np.ndarray):
        barotropic_psic = np.mean(psic, psic.ndim - 1)
        return energy_spec(barotropic_psic)
    elif isinstance(psic, nc_tools.NetCDFChain):
        return _barotropic_Ek_ncchain(psic)
    else:
        raise TypeError("Input type not supported")

def _barotropic_spec_ncchain(psi):
    """
    process NetCDFChain object with lots of files
    @param psi has shape (time, real_and_imag, ky, kx, z)
    
    Return:
        sum up psi(k)*conj(psi(k)) for total wavenumber k
    """
    if not isinstance(psi, nc_tools.NetCDFChain):
        raise TypeError("Not NetCDFChain object")
    nky, nkx = psi.shape[-3:-1]
    ksqd_    = np.zeros((nky, nkx), dtype=float)
    kmax     = nky - 1
    indx_kx0 = (nkx-1)/2
    KE2d  = np.zeros((nky, nkx), dtype=float)
    Ek    = np.zeros(kmax,       dtype=float)
    EKEk  = np.zeros(kmax,       dtype=float)
    
    for j in range(0, nky):
        for i in range(0, nkx):
            ksqd_[j,i] = (i-kmax)**2 + j**2
        
    radius_arr = np.floor(np.sqrt(ksqd_)).astype(int)

    for i, dt in enumerate(psi.time_steps_each_file):
        #reading each file all together; if read in each time step each time
        #reading files will take too much time
        psi_seg = psi[psi.time_steps_before[i]:psi.time_steps_before[i]+dt]
        if psi_seg.ndim != 5:
            raise TypeError("file does not have correct number of dimensions")
        psi_seg = np.mean(psi_seg, psi_seg.ndim-1) #barotropic field
        KE2d += np.mean(np.sum(psi_seg*psi_seg, 1), 0)

    KE2d /= len(psi.sorted_files)

    for i in range(0,kmax):
        Ek[i]   = np.sum(KE2d[radius_arr == i+1])
    
    KE2d[:,indx_kx0] = 0.

    for i in range(0,kmax):
        EKEk[i]   = np.sum(KE2d[radius_arr == i+1])

    return np.arange(1,kmax+1), Ek, EKEk
    
    
def barotropic_spec(psic):
    """
    barotropic energy spectrum for multiple times and multiple layers
    @param psic complex numpy array
                returned from real2complex, has shape 
                (time (optional), ky, kx, z)
            OR  NetCDFChain object
                has shape (time, real_and_imag, ky, kx, z)
    @return (wavenumber, Ek, EKEk) 1d numpy array
        Ek is sum of psic(k)*conj(psic(k)) at total wavenumber k
    """
    if isinstance(psic, scipy.io.netcdf.netcdf_variable):
        psic = psic[:]
        psic = real2complex(psic)
    if isinstance(psic, np.ndarray):
        raise NotImplementedError('Please use prod_spec instead')
    elif isinstance(psic, nc_tools.NetCDFChain):
        return _barotropic_spec_ncchain(psic)
    else:
        raise TypeError("Input type not supported")
        
def filter(spec, k_min, k_max, remove_zonal=False):
    """filter the spectrum
    keep only the spetrum component with wavenumber (sqrt(kx**2+ky**2)) within
    [k_min, kmax]
    
    Args:
        spec: complex numpy array with shape (time(optional), ky, kx, z(optional))
        k_min, k_max: integer|None, mininum and maximum wavenumbers to keep
            if set to None, then means no lower or upper bound
        remove_zonal: True|False (default)
        
    Return:
        filtered_spec: same size as the input `spec`
    """
    if not _is_single_layer(spec):
        nky, nkx = spec.shape[-3:-1]
        ksqd_    = np.zeros((nky, nkx), dtype=float)
        kmax     = nky - 1
        
        for j in range(0, nky):
            for i in range(0, nkx):
                ksqd_[j,i] = (i-kmax)**2 + j**2
        mask = np.ones_like(ksqd_)
        if k_min:
            mask[ksqd_ < k_min**2] = 0.
        if k_max:
            mask[ksqd_ > k_max**2] = 0.
        if remove_zonal:
            mask[:,(nkx-1)/2] = 0.
        mask.shape = (1,)*(spec.ndim-3) + mask.shape + (1,)
        return mask*spec
    else:
        nky, nkx = spec.shape[-2:]
        ksqd_    = np.zeros((nky, nkx), dtype=float)
        kmax     = nky - 1
        
        for j in range(0, nky):
            for i in range(0, nkx):
                ksqd_[j,i] = (i-kmax)**2 + j**2
        mask = np.ones_like(ksqd_)
        if k_min:
            mask[ksqd_ < k_min**2] = 0.
        if k_max:
            mask[ksqd_ > k_max**2] = 0.
        if remove_zonal:
            mask[:,(nkx-1)/2] = 0.
        mask.shape = (1,)*(spec.ndim-2) + mask.shape
        return mask*spec
    
def filter_zonal(spec, kx_min, kx_max, remove_zonal=False):
    """filter the spectrum in the zonal direction
    keep only the spetrum component with zonal wavenumber kx within
    [k_min, kmax]
    
    Args:
        spec: complex numpy array with shape (time(optional), ky, kx, z)
        kx_min, kx_max: integer|None, mininum and maximum wavenumbers to keep
            if set to None, then means no lower or upper bound
        remove_zonal: True|False (default)
        
    Return:
        filtered_spec: same size as the input `spec`
    """
    nky, nkx = spec.shape[-3:-1]
    abskx_    = np.zeros((nky, nkx), dtype=float)
    kmax     = nky - 1
    
    for j in range(0, nky):
        for i in range(0, nkx):
            abskx_[j,i] = np.abs(i-kmax)
    mask = np.ones_like(abskx_)
    if kx_min:
        mask[abskx_ < kx_min] = 0.
    if kx_max:
        mask[abskx_ > kx_max] = 0.
    if remove_zonal:
        mask[:,(nkx-1)/2] = 0.
    mask.shape = (1,)*(spec.ndim-3) + mask.shape + (1,)
    return mask*spec

def get_eddy(fields):
    """get eddy from spectral field `fields`
    
    Args:
        fields: spectral field with shape (time(optional), ky, kx, z(optional))
        
    Return:
        eddy_fields: spectral field of the eddy part flow which has the same 
            shape as `fields`
    """
    eddy_fields = fields.copy()
    if _is_single_layer(fields):
        nkx = fields.shape[-1]
        eddy_fields[...,(nkx-1)/2] = 0
    else:
        nkx = fields.shape[-2]
        eddy_fields[...,(nkx-1)/2,:] = 0
    return eddy_fields

def get_zonalmean(fields):
    """get zonal flow from spectral field `fields`
    """
    zonal_fields = np.zeros_like(fields)
    if _is_single_layer(fields):
        nkx = fields.shape[-1]
        zonal_fields[...,(nkx-1)/2] = fields[...,(nkx-1)/2]
    else:
        nkx = fields.shape[-2]
        zonal_fields[...,(nkx-1)/2,:] = fields[...,(nkx-1)/2,:]
    return zonal_fields
        