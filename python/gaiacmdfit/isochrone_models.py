import os
import numpy as np
import time
from scipy.stats import binned_statistic_2d
from chronos import isochrone
import matplotlib.pyplot as plt
from dlnpyutils import utils as dln,plotting as pl
from astropy.convolution import convolve as aconvolve
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from astropy.table import Table
from scipy.interpolate import interp1d
from . import utils

ISOGRID = isochrone.load()

class CMDModel(object):

    def __init__(self):
        models,ages,metals = loadmodels()
        self._models = models
        self.ages = ages
        self.metals = metals
        # Grid information
        self.xr = [-1.0,4.0]
        self.yr = [-10.0,10.0]
        self.dx = 0.01
        self.dy = 0.01
        self.nx = int((self.xr[1]-self.xr[0])/self.dx)+1
        self.ny = int((self.yr[1]-self.yr[0])/self.dy)+1
        self.xbins = np.arange(self.nx)*self.dx+self.xr[0]-0.5*self.dx   # bin edges
        self.ybins = np.arange(self.ny)*self.dy+self.yr[0]-0.5*self.dy
        self.bprpbins = np.arange(self.nx)*self.dx+self.xr[0]
        self.gmagbins = np.arange(self.ny)*self.dy+self.yr[0]
        
    def __getitem__(self,index):
        """ Get a single pre-computed cmd model """
        if len(index) != 2:
            raise IndexError("Index must have two elements")
        xind,yind = index
        if isinstance(xind,int)==False or isinstance(yind,int)==False:
            age,metal = xind,yind
            xind = np.argmin(np.abs(self.ages-age))
            yind = np.argmin(np.abs(self.metals-metal))
        return self._models[xind][yind]
        
    def __call__(self,xargs,*pars):
        """ Create the model. """
        return cmdmodel(xargs,*pars,cmd=self)

    def model(self,xargs,*pars):
        return self(xargs,*pars)

    def jac(self,xargs,*pars):
        m,j = cmdmodel(xargs,*pars,cmd=self,dojacobian=True)
        return j
    
    def __repr__(self):
        out = self.__class__.__name__+'\n'
        out += '{:.1f} <= Age <= {:.1f} Myr, in steps of {:.1f} Myr ({:d} ages)\n'.format(self.ages[0],self.ages[-1],self.ages[1]-self.ages[0],len(self.ages))
        out += '{:.2f} <= [M/H] <= {:.2f}, in steps of {:.1f} ({:d} metallicities)\n'.format(self.metals[0],self.metals[-1],self.metals[1]-self.metals[0],len(self.metals))
        out += '{:.2f} <= BP-RP <= {:.2f}\n'.format(self.xr[0],self.xr[1])
        out += '{:.2f} <= G <= {:.2f}\n'.format(self.yr[0],self.yr[1])
        return out 

def fineisochrone(iso,n=10000):
    """ Finely sample isochrone """

    data = iso.data
    ndata = len(data)
    osamp = n/ndata    # oversampling factor
    
    # Initialize the new isochrone
    newiso = Table(np.zeros(n,dtype=data.dtype))
        
    ## Create normalized index array (from 0-1)
    #indx = np.arange(ndata).astype(float)/(ndata-1)
    #newindx = np.arange(n).astype(float)/(n-1)

    ## Interpolate all of the relevant columns
    #for n in data.colnames:
    #    newval = interp1d(indx,data[n])(newindx)
    #    newiso[n] = newval

    # Label loop
    out = []
    labels = np.unique(data['LABEL'])
    for i in range(len(labels)):
        ind, = np.where(data['LABEL']==labels[i])
        nind = len(ind)
        # single point
        if nind==1:
            out1 = data[ind].copy()
        else:
            indx = np.arange(len(ind)).astype(float)/(nind-1)
            nnew = int(np.ceil(nind*osamp))
            newindx = np.arange(nnew).astype(float)/(nnew-1)
            out1 = np.zeros(nnew,dtype=data.dtype)
            # Interpolate all of the relevant columns
            for n in data.colnames:
                newval = interp1d(indx,data[n][ind])(newindx)
                out1[n] = newval
        out.append(out1)

    # Now combine the labels into one table
    nfinal = np.sum([len(o) for o in out])
    newiso = Table(np.zeros(nfinal,dtype=data.dtype))
    count = 0
    for i in range(len(out)):
        out1 = out[i]
        newiso[count:count+len(out1)] = out1
        count += len(out1)
        
    # Number of stars per isochrone point (for 1 Msun)
    pdf = np.maximum(np.diff(newiso['INT_IMF'].data),0.0)
    pdf = np.hstack((pdf[0], pdf))
    newiso['PDF'] = pdf

    #newiso = isochrone.Isochrone(newiso)
    
    return newiso
    
    
def loadmodels():
    """ Load all models."""
    datadir = utils.datadir()+'/synthgiants/'
    ages = np.arange(0.5,13,0.1)
    metals = np.arange(-2.0,0.25,0.05)
    models = len(ages)*[None]
    for i in range(len(ages)):
        age = ages[i]
        amodels = len(metals)*[None]
        for j in range(len(metals)):
            metal = metals[j]
            synfile = datadir+'/syna{:03d}m{:.2f}.npz'.format(int(np.round(age*10)),metal)
            syntab = np.load(synfile)['tab']
            amodels[j] = syntab
        models[i] = amodels
    return models,ages,metals
    
def makemodel(age,metal,xr=[-1.0,4.0],yr=[-10.0,10.0],dx=0.01,dy=0.01):
    """ Make single 2D synthetic CMD model """

    iso = ISOGRID(age,metal)
    #syn = iso.synth(2000000,minmass=0.9)
    syn = iso.synth(totmass=1e7) #,minlabel=2)
    bprp = np.array(syn['GAIAEDR3_GBPMAG']-syn['GAIAEDR3_GRPMAG'])
    gmag = np.array(syn['GAIAEDR3_GMAG'])
    label = np.array(syn['LABEL'])
    nx = int((xr[1]-xr[0])/dx)+1
    ny = int((yr[1]-yr[0])/dy)+1
    xbins = np.arange(nx)*dx+xr[0]-0.5*dx   # bin edges
    ybins = np.arange(ny)*dy+yr[0]-0.5*dy
    o = binned_statistic_2d(bprp,gmag,gmag,statistic='count',bins=(xbins,ybins))
    res,xedge,yedge,binnumber = o
    o = binned_statistic_2d(bprp,gmag,label,statistic='mean',bins=(xbins,ybins))
    labelres,_,_,_ = o
    # res is [Nx,Ny]
    
    ygd,xgd = np.where(res > 0)
    dt = [('age',float),('metal',float),('xind',int),('yind',int),
          ('bprp',float),('gmag',float),('label',float),('count',int)]
    tab = np.zeros(len(xgd),dtype=np.dtype(dt))
    tab['age'] = age
    tab['metal'] = metal
    tab['xind'] = xgd
    tab['yind'] = ygd
    tab['bprp'] = xbins[ygd]
    tab['gmag'] = ybins[xgd]
    tab['label'] = labelres[ygd,xgd]
    tab['count'] = res[ygd,xgd]
    
    return tab
    
def makemodels(clobber=False):
    """ Make 2D synthetic cmd models on fine metallicity/age grid """
    ages = np.arange(0.5,13,0.1)
    metals = np.arange(-2.0,0.25,0.05)

    for i in range(len(ages)):
        age = ages[i]
        for j in range(len(metals)):
            metal = metals[j]
            outfile = 'synth/syna{:03d}m{:.2f}.npz'.format(int(np.round(age*10)),metal)
            if os.path.exists(outfile) and clobber==False:
                continue
            tab = makemodel(age*1e9,metal)
            print(i+1,j+1,age,metal,len(tab))
            np.savez(outfile,tab=tab)

    import pdb; pdb.set_trace()

def cmdmodel(xargs,*pars,cmd=None,dojacobian=False):
    """ Produce a CMD model."""

    print('cmdmodel:',pars)
    
    # Absolute magnitude grid information
    xr = [-1.0,4.0]
    yr = [-10.0,10.0]
    dx = 0.01
    dy = 0.01
    nx = int((xr[1]-xr[0])/dx)+1
    ny = int((yr[1]-yr[0])/dy)+1
    xbins = np.arange(nx)*dx+xr[0]-0.5*dx   # bin edges
    ybins = np.arange(ny)*dy+yr[0]-0.5*dy
    bprpbins = np.arange(nx)*dx+xr[0]
    absgmagbins = np.arange(ny)*dy+yr[0]
    
    # pars
    # -total mass
    # -mean age
    # -age dispersion
    # -mean metallicity
    # -metallicity dispersion
    # -mean distance
    # -distance dispersion
    totmass,mnage,sigage,mnmetal,sigmetal,mndist,sigdist = pars
    
    # Each precomputed CMD model has a total mass of 1e7 solar masses
    # For the gaussians, go +/- 2 sigma, or to the edge
    allages = np.arange(0.5,13,0.1)
    allmetals = np.arange(-2.0,0.25,0.05)
    gdages, = np.where(np.abs(allages-mnage) <= 2*sigage)
    gdmetal, = np.where(np.abs(allmetals-mnmetal) <= 2*sigmetal)
    ages = allages[gdages]
    metals = allmetals[gdmetal]
    nages = len(ages)
    nmetals = len(metals)
    fracages = np.exp(-0.5*(ages-mnage)**2/sigage**2)
    fracmetal = np.exp(-0.5*(metals-mnmetal)**2/sigmetal**2)
    fraction = fracages.reshape(-1,1) * fracmetal.reshape(1,-1)
    fraction /= np.sum(fraction)  # normalize
    fraction *= totmass/1e7

    # Start CMD in absolute magnitudes
    absarr = np.zeros((nx,ny,nages,nmetals),float)
    absim = np.zeros((nx,ny),float)
    for i in range(len(ages)):
        age = ages[i]
        #print(i,age)
        for j in range(len(metals)):
            metal = metals[j]
            frac = fraction[i,j]
            # Load the precomputed cmd
            if cmd is None:
                synfile = 'synth/syna{:03d}m{:.2f}.npz'.format(int(np.round(age*10)),metal)
                syntab = np.load(synfile)['tab']
            else:
                syntab = cmd[age,metal]
            xind = syntab['xind'].astype(int)
            yind = syntab['yind'].astype(int)
            absarr[yind,xind,i,j] += syntab['count']
            absim[yind,xind] += syntab['count']*frac
            #import pdb; pdb.set_trace()
            
    # Absolute magnitude bins with stars
    #absygd,absxgd = np.where(absim > 0)
    #bprp = bprpbins[absygd]
    #absgmag = absgmagbins[absxgd]
    #count = absim[absygd,absxgd] 

    # Generate discrete stars from this absolute magnitude distribution
    nstars = int(np.sum(absim))
    np.random.seed(1)
    stars = dln.random_distribution(absim,nstars)
    absbprp = dln.scale(stars[:,0],[0,absim.shape[0]-1],[bprpbins[0],bprpbins[-1]])
    absgmag = dln.scale(stars[:,1],[0,absim.shape[1]-1],[absgmagbins[0],absgmagbins[-1]])
    
    # Now add distance
    #  We're going to "rename" the absolute "absgmag" array to gmag+mean distance modulus
    #  then smooth along the gmag axis to create the dispersion in distance
    mndm = 5*np.log10(mndist)+10
    gmag = absgmag + mndm
    gmagbins = absgmagbins + mndm
    #sigdm = 5*np.log10(mndist+sigdist)-5*np.log10(mndist)
    sigdm = 5*np.log10(1+sigdist/mndist)
    sigpix = sigdm/dy
    fwhmdm = 2.35*sigdm
    fwhmpix = fwhmdm/dy   # in pixels
    # lag = np.arange(201)-100  # in pixels
    # distfrac = np.exp(-0.5*lag**2/sigpix**2)
    # distfrac /= np.sum(distfrac)
    # # add +/-100 pixel buffer, to allow for the shift
    # tempim = np.zeros((absim.shape[0],absim.shape[1]+200),float)
    # for i in range(len(lag)):
    #     lag1 = lag[i]
    #     tempim[absygd,absxgd+lag1+100] += count*distfrac[i]
    # synim = tempim[:,100:-100]   # trim buffer
    # # non-zero pixels
    # ygd,xgd = np.where(synim > 0)
    # synbprp = bprpbins[ygd]
    # syngmag = gmagbins[xgd]
    # syncount = synim[ygd,xgd]

    distrnd = np.random.randn(nstars)
    distscatter = distrnd*sigdm + mndm
    synbprp = absbprp
    syngmag = absgmag + distscatter

    # Add photometric uncertainties
    # log error vs. G
    magerrcoef = np.array([ 1.04713727e-03, -2.96578636e-02,  2.98404246e-01, -4.75305609e+00])
    colerrcoef = np.array([-1.24587713e-03,  7.75731705e-02, -1.23578577e+00,  2.72041956e+00])
    magerr = 10**np.polyval(magerrcoef,syngmag)
    colerr = 10**np.polyval(colerrcoef,syngmag)
    colrnd = np.random.randn(nstars)
    magrnd = np.random.randn(nstars)
    synbprp_werr = synbprp + colrnd*colerr
    syngmag_werr = syngmag + magrnd*magerr
    
    # Now put everything on the final observed grid
    colorbin2d,magbin2d = xargs[:,0],xargs[:,1]
    colorbins = np.unique(colorbin2d)
    magbins = np.unique(magbin2d)
    ncol = len(colorbins)
    nmag = len(magbins)
    colr = [colorbins[0],colorbins[-1]]
    magr = [magbins[0],magbins[-1]]
    dcol = np.round(colorbins[1]-colorbins[0],3)
    dmag = np.round(magbins[1]-magbins[0],3)
    #ncol = int((colr[1]-colr[0])/dcol)+1
    #nmag = int((magr[1]-magr[0])/dmag)+1
    finalim = np.zeros((ncol,nmag),float)
    # Figure out which final bins our syn counts land in
    colind = ((synbprp_werr-colr[0])/dcol).astype(int)
    magind = ((syngmag_werr-magr[0])/dmag).astype(int)
    inbounds, = np.where((colind >= 0) & (colind <= (ncol-1)) &
                         (magind >= 0) & (magind <= (nmag-1)))
    for i in inbounds:
        finalim[colind[i],magind[i]] += 1   #syncount[i]
    finalim = finalim.ravel()
        
    # Jacobian
    if dojacobian:
        print('cmdjacobian')
        jac = np.zeros((len(finalim),7),float)
        f0 = finalim
    
        # Parameter 1) amplitude (totmass)
        #---------------------------------
        jac[:,0] = f0 / totmass
    
        # Parameter 2) mean age (mnage)
        #------------------------------
        mnage_step = 0.1*mnage
        # new age/metallicity fractions
        fracages2 = np.exp(-0.5*(ages-(mnage+mnage_step))**2/sigage**2)
        fracmetal2 = np.exp(-0.5*(metals-mnmetal)**2/sigmetal**2)
        fraction2 = fracages2.reshape(-1,1) * fracmetal2.reshape(1,-1)
        fraction2 /= np.sum(fraction2)  # normalize
        fraction2 *= totmass/1e7
        # make new absolute magnitude image
        absim2 = np.zeros(absim.shape,float)
        for i in range(len(ages)):
            for j in range(len(metals)):
                absim2 += absarr[:,:,i,j]*fraction2[i,j]
        # generate new set of stars
        np.random.seed(1)  # use seed as before
        stars2 = dln.random_distribution(absim2,nstars)
        absbprp2 = dln.scale(stars2[:,0],[0,absim.shape[0]-1],[bprpbins[0],bprpbins[-1]])
        absgmag2 = dln.scale(stars2[:,1],[0,absim.shape[1]-1],[absgmagbins[0],absgmagbins[-1]])
        # distance scatter
        synbprp2 = absbprp2
        syngmag2 = absgmag2 + distscatter  # same distance scatter as before
        # photometric uncertainties
        magerr2 = 10**np.polyval(magerrcoef,syngmag2)
        colerr2 = 10**np.polyval(colerrcoef,syngmag2)
        synbprp2_werr = synbprp2 + colrnd*colerr2
        syngmag2_werr = syngmag2 + magrnd*magerr2
        # on final grid
        colind2 = ((synbprp2_werr-colr[0])/dcol).astype(int)
        magind2 = ((syngmag2_werr-magr[0])/dmag).astype(int)
        inbounds2, = np.where((colind2 >= 0) & (colind2 <= (ncol-1)) &
                              (magind2 >= 0) & (magind2 <= (nmag-1)))
        f2 = np.zeros((ncol,nmag),float)
        for i in inbounds2:
            f2[colind2[i],magind2[i]] += 1
        f2 = f2.ravel()    
        jac[:,1] = (f2-f0)/mnage_step
    
        # Parameter 3) sigma age (sigage)
        #--------------------------------
        sigage_step = np.maximum(0.5*sigage,0.2*mnage)
        # new age/metallicity fractions
        fracages3 = np.exp(-0.5*(ages-mnage)**2/(sigage+sigage_step)**2)
        fracmetal3 = np.exp(-0.5*(metals-mnmetal)**2/sigmetal**2)
        fraction3 = fracages3.reshape(-1,1) * fracmetal3.reshape(1,-1)
        fraction3 /= np.sum(fraction3)  # normalize
        fraction3 *= totmass/1e7
        # make new absolute magnitude image
        absim3 = np.zeros(absim.shape,float)
        for i in range(len(ages)):
            for j in range(len(metals)):
                absim3 += absarr[:,:,i,j]*fraction3[i,j]
        # generate new set of stars
        np.random.seed(1)  # use seed as before
        stars3 = dln.random_distribution(absim3,nstars)
        absbprp3 = dln.scale(stars3[:,0],[0,absim.shape[0]-1],[bprpbins[0],bprpbins[-1]])
        absgmag3 = dln.scale(stars3[:,1],[0,absim.shape[1]-1],[absgmagbins[0],absgmagbins[-1]])
        # distance scatter
        synbprp3 = absbprp3
        syngmag3 = absgmag3 + distscatter  # same distance scatter as before
        # photometric uncertainties
        magerr3 = 10**np.polyval(magerrcoef,syngmag3)
        colerr3 = 10**np.polyval(colerrcoef,syngmag3)
        synbprp3_werr = synbprp3 + colrnd*colerr3
        syngmag3_werr = syngmag3 + magrnd*magerr3
        # on final grid
        colind3 = ((synbprp3_werr-colr[0])/dcol).astype(int)
        magind3 = ((syngmag3_werr-magr[0])/dmag).astype(int)
        inbounds3, = np.where((colind3 >= 0) & (colind3 <= (ncol-1)) &
                              (magind3 >= 0) & (magind3 <= (nmag-1)))
        f3 = np.zeros((ncol,nmag),float)
        for i in inbounds3:
            f3[colind3[i],magind3[i]] += 1
        f3 = f3.ravel() 
        jac[:,2] = (f3-f0)/sigage_step
    
        # Parameter 4) mean metallicity (mnmetal)
        #-----------------------------------------
        mnmetal_step = 0.1
        # new age/metallicity fractions
        fracages4 = np.exp(-0.5*(ages-mnage)**2/sigage**2)
        fracmetal4 = np.exp(-0.5*(metals-(mnmetal+mnmetal_step))**2/sigmetal**2)
        fraction4 = fracages4.reshape(-1,1) * fracmetal4.reshape(1,-1)
        fraction4 /= np.sum(fraction4)  # normalize
        fraction4 *= totmass/1e7
        # make new absolute magnitude image
        absim4 = np.zeros(absim.shape,float)
        for i in range(len(ages)):
            for j in range(len(metals)):
                absim4 += absarr[:,:,i,j]*fraction4[i,j]
        # generate new set of stars
        np.random.seed(1)  # use seed as before
        stars4 = dln.random_distribution(absim4,nstars)
        absbprp4 = dln.scale(stars4[:,0],[0,absim.shape[0]-1],[bprpbins[0],bprpbins[-1]])
        absgmag4 = dln.scale(stars4[:,1],[0,absim.shape[1]-1],[absgmagbins[0],absgmagbins[-1]])
        # distance scatter
        synbprp4 = absbprp4
        syngmag4 = absgmag4 + distscatter  # same distance scatter as before
        # photometric uncertainties
        magerr4 = 10**np.polyval(magerrcoef,syngmag4)
        colerr4 = 10**np.polyval(colerrcoef,syngmag4)
        synbprp4_werr = synbprp4 + colrnd*colerr4
        syngmag4_werr = syngmag4 + magrnd*magerr4
        # on final grid
        colind4 = ((synbprp4_werr-colr[0])/dcol).astype(int)
        magind4 = ((syngmag4_werr-magr[0])/dmag).astype(int)
        inbounds4, = np.where((colind4 >= 0) & (colind4 <= (ncol-1)) &
                              (magind4 >= 0) & (magind4 <= (nmag-1)))
        f4 = np.zeros((ncol,nmag),float)
        for i in inbounds4:
            f4[colind4[i],magind4[i]] += 1
        f4 = f4.ravel() 
        jac[:,3] = (f4-f0)/mnmetal_step
        
        # Parameter 5) sigma metallicity (sigmetal)
        #-----------------------------------------
        sigmetal_step = 0.1
        # new age/metallicity fractions
        fracages5 = np.exp(-0.5*(ages-mnage)**2/sigage**2)
        fracmetal5 = np.exp(-0.5*(metals-mnmetal)**2/(sigmetal+sigmetal_step)**2)
        fraction5 = fracages5.reshape(-1,1) * fracmetal5.reshape(1,-1)
        fraction5 /= np.sum(fraction5)  # normalize
        fraction5 *= totmass/1e7
        # make new absolute magnitude image
        absim5 = np.zeros(absim.shape,float)
        for i in range(len(ages)):
            for j in range(len(metals)):
                absim5 += absarr[:,:,i,j]*fraction5[i,j]
        # generate new set of stars
        np.random.seed(1)  # use seed as before
        stars5 = dln.random_distribution(absim5,nstars)
        absbprp5 = dln.scale(stars5[:,0],[0,absim.shape[0]-1],[bprpbins[0],bprpbins[-1]])
        absgmag5 = dln.scale(stars5[:,1],[0,absim.shape[1]-1],[absgmagbins[0],absgmagbins[-1]])
        # distance scatter
        synbprp5 = absbprp5
        syngmag5 = absgmag5 + distscatter  # same distance scatter as before
        # photometric uncertainties
        magerr5 = 10**np.polyval(magerrcoef,syngmag5)
        colerr5 = 10**np.polyval(colerrcoef,syngmag5)
        synbprp5_werr = synbprp5 + colrnd*colerr5
        syngmag5_werr = syngmag5 + magrnd*magerr5
        # on final grid
        colind5 = ((synbprp5_werr-colr[0])/dcol).astype(int)
        magind5 = ((syngmag5_werr-magr[0])/dmag).astype(int)
        inbounds5, = np.where((colind5 >= 0) & (colind5 <= (ncol-1)) &
                              (magind5 >= 0) & (magind5 <= (nmag-1)))
        f5 = np.zeros((ncol,nmag),float)
        for i in inbounds5:
            f5[colind5[i],magind5[i]] += 1
        f5 = f5.ravel() 
        jac[:,4] = (f5-f0)/sigmetal_step
        
        # Parameter 6) mean distance (mndist/mndm)
        #-----------------------------------------
        mndm_step = 0.1
        mndist_step = mndist*(10**(mndm_step/5)-1)
        #mndist_step = 10**((mndm+mndm_step+5)/5)/1e3 - 10**((mndm+5)/5)/1e3
        syngmag6 = syngmag_werr.copy() + mndm_step
        colind6 = colind.copy()   # colors haven't changed
        magind6 = ((syngmag6-magr[0])/dmag).astype(int)
        inbounds6, = np.where((colind6 >= 0) & (colind6 <= (ncol-1)) &
                              (magind6 >= 0) & (magind6 <= (nmag-1)))
        f6 = np.zeros((ncol,nmag),float)
        for i in inbounds6:
            f6[colind6[i],magind6[i]] += 1
        f6 = f6.ravel()    
        jac[:,5] = (f6-f0)/mndist_step
    
        # Parameter 7) sigma distance (sigdist/sigdm)
        #--------------------------------------------
        sigdm_step = 0.1
        sigdist_step = mndist*(10**(mndm_step/5)-1)
        # Add distance scatter
        distscatter7 = distrnd*(sigdm+sigdm_step) + mndm
        syngmag7 = absgmag + distscatter7
        # Add photometric uncertainties
        syngmag7_werr = syngmag7 + magrnd*magerr
        colind7 = colind.copy()   # colors haven't changed
        magind7 = ((syngmag7_werr-magr[0])/dmag).astype(int)
        inbounds7, = np.where((colind7 >= 0) & (colind7 <= (ncol-1)) &
                              (magind7 >= 0) & (magind7 <= (nmag-1)))
        f7 = np.zeros((ncol,nmag),float)
        for i in inbounds7:
            f7[colind7[i],magind7[i]] += 1
        f7 = f7.ravel()    
        jac[:,6] = (f7-f0)/sigdist_step

        return finalim,jac
        
        
    #return absim,synim,finalim
    return finalim
