#----------------------------
#   NAME
#----------------------------
# balff_run.py
#----------------------------
#   PURPOSE/DESCRIPTION
#----------------------------
# Running the pymc MCMC sampler on a dataarray of objects using the
# balff_mpd class with a Robust Adaptive Metropolis Hastings (RAM)
# stepping as default (can be changed with --step)
#----------------------------
#   INPUTS:
#----------------------------
# datafile         : Binary fits table containing the data of the individual fields/surveys to
#                    estimate the individual marginalized posterior distribution in balff_mpd.py
#                    A minimal version of this file can be generated with balff_createDataArray.py
# --contamfrac     : Contamination fraction to apply to sampl in datafile. Default value is 0.42
# --lookuptable    : Binary look-up table (.npz) for datafile used for the normalizing probabilities when
#                    estimating ln(mpd). If not provided the values will be calculated from scratch which
#                    is slow. A look-up table can be generated with balff_createLookupTable.py
# --Niter          : Number of MCMC iterations.             DEFAULT = 10000
# --Nburn          : Number of MCMC burn-in iterations.     DEFAULT = 1000
# --Nthin          : Factor to thin MCMC output with.       DEFAULT = 10
# --step           : Stepping in MCMC chaing to perform.
#                    Chose between:
#                         Robust Adaptive Metropolis Hastings (DEFAULT)
#                         'metropolis'
#                    To add more see pymc.StepMethodRegistry
# --pdist          : Proposal distribution to use for step method. Choose between:
#                         'Normal'     Normal distribution (DEFAULT)
#                         'T'          Student T distribution
# --samprange      : Setting allowed values to sample from as
#                    kmin kmax log10Lstarmin log10Lstarmax logNmin logNmax
#                    The default ranges are:
#                          kmin           = -4.0
#                          kmax           =  4.0
#                          logLstarmin    = -3.0
#                          logLstarmax    =  3.0
#                          logNmin        =  0.0
#                          logNmax        = 50.0
# --chainstart     : The position in parameter space to start MCMC chain. Expects 3 values:
#                    k log10Lstar log10N. Default is 'None', i.e. random starting position
# --errdist        : The error distribution to use in balff_mpd model. Chose between
#                       'normal'         A normal (gauss) distribution of errors (DEFAULT)
#                       'lognormal'      A log normal distribution of errors
#                       'normalmag'      A log normal in L turned into normal in mag
#                       'magbias'        Use error distribution taking magnification bias into account
#                                        where p(mu) is a sum of up to 3 gaussian distributions which
#                                        should be provided for each field in ./balff_data/fields_magbiaspdf.txt
# --selectionfct   : Selection function to use (0 -> step, 1 -> tabualted)
# --lnptime        : Set this keyword to print the time steps for each individual lnprob calculation
# --datafileonly   : To run only using the objects in the actual data array. Needed when running simulated data
#                    or the Bouwens only data as these tables don't include the empty BoRG fields.
# --emptysim       : To run using simulated empty fields. Essentially what this does is ignoring the BoRG/UDF/ERS
#                    empty fields and substitute them with N (defined in balff_mpd.__init__) fields using
#                    dLfiledlim and Lfieldlim of FIELD1 in datafile. Only works if --datafileonly is not set
# --LFredshift     : The redshift at which the luminoisity function is determined. Used for distance
#                    modulus when turning 5-sigma limiting magnitude into absolute luminosity and
#                    selection function integrations. DEFAULT luminoisity function redshift is LFredshift=8
# --verbose        : Toggle verbosity
# --help           : Printing help menu
#----------------------------
#   OUTPUTS:
#----------------------------
#
#----------------------------
#   EXAMPLES/USAGE
#----------------------------
# see the shell script balff_run_commands.sh
#----------------------------
#   MODULES
#----------------------------
from my_pymc_steps import RobustAdaptiveMetro
import balff_mpd as mpd
import argparse
import sys
import numpy as np
import pdb
import pymc
# import pymc3 as pymc
import subprocess
import os
# from pymc.Matplot import plot
# import pymc.plots as plot
from pymc import HalfCauchy, Model, Normal, Uniform, sample
import time
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import Parameters, minimize, fit_report#, Minimizer, Model, 
import emcee
import zeus
import corner

from gsf.minimizer import Minimizer
from gsf.function import savecpkl #get_leastsq, print_err, str2bool

#-------------------------------------------------------------------------------------------------------------
# new step method
class simplegibbs(object):
    def __init__(self, stochastic, *args, **kwargs):
        super(pymc.step_method, self).__init__(stochastic, *args, **kwargs)
        self._id = 'simplegibbs'
        self._tuning_info = ['none']

    def step(self):
        self.stochastic.value = self.stochastic.random()

    def competence(self, stochastic):
        return 3

    def current_state(self):
        state = {}
        for s in self._state:
            state[s] = getattr(self, s)
        return state

#-------------------------------------------------------------------------------------------------------------
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("datafile", type=str, help="Fits binary table containing data for objects")
# ---- optional arguments ----
parser.add_argument("--contamfrac", type=float, help="Contamination fraction to apply to sample (default=0.42)")
parser.add_argument("--lookuptable", type=str, help="Look-up table for datafile to use.")
parser.add_argument("--Niter", type=int, help="Number of MCMC iterations")
parser.add_argument("--Nburn", type=int, help="Number of MCMC burn-in iterations")
parser.add_argument("--Nthin", type=int, help="Factor to thin MCMC output with")
parser.add_argument("--samprange", type=float, nargs=6, help="Allowed values to sample from: kmin kmax log10Lstarmin log10Lstarmax log10Nmin log10Nmax")
parser.add_argument("--chainstart", type=float, nargs=3, help="Position to start MCMC chain: k log10Lstar log10N")
parser.add_argument("--step", type=str, help="Stepping method to use in MCMC chain")
parser.add_argument("--pdist", type=str, help="Proposal distribution to use in stepping method")
parser.add_argument("--errdist", type=str, help="Error distribution to use")
parser.add_argument("--selectionfct", type=float, help="Selection function to use (0 -> step, 1 -> tabualted)")
parser.add_argument("--lnptime", action="store_true", help="Print time steps for individual lnprob calculations")
parser.add_argument("--datafileonly", action="store_true", help="Only consider fields in data file (not empty fields)")
parser.add_argument("--emptysim", action="store_true", help="Use simulated empty fields")
parser.add_argument("--LFredshift", type=float, help="Redshift of Luminosity Function to be determined (default is LFredshift=8")
parser.add_argument("-v", "--verbose", action="store_true", help="Set verbosity on")
args = parser.parse_args()
#-------------------------------------------------------------------------------------------------------------
if args.verbose:
    print(' ')
    print(':: '+sys.argv[0]+' :: -- START OF PROGRAM -- ')
    print(' ')
#-------------------------------------------------------------------------------------------------------------
if args.lookuptable:
    lutabval  = args.lookuptable
else:
    lutabval  = None
if args.verbose: print(' - Lookuptable         :',lutabval)
#-------------------------------------------------------------------------------------------------------------
if args.selectionfct:
    selfctval = args.selectionfct
else:
    selfctval = 0
if args.verbose: print(' - Selection function  :',selfctval)
if selfctval == 1:
    loadval = True
else:
    loadval = False
#-------------------------------------------------------------------------------------------------------------
if args.errdist:
    errdist  = args.errdist
    if errdist not in ['normal','lognormal','normalmag','magbias']:
        if args.verbose: print('\n - NB! False errdist "'+errdist+'" provided, using default errdist=normal')
        errdist  = 'normal'
else:
    errdist  = 'normal'
if args.verbose: print(' - Error distribution  :',errdist)
#-------------------------------------------------------------------------------------------------------------
if args.contamfrac != None:
    contamfrac = args.contamfrac
else:
    contamfrac = 0.42
if args.verbose: print(' - Contamfrac          :',contamfrac)
#-------------------------------------------------------------------------------------------------------------
if args.datafileonly:
    datafileonly = True
else:
    datafileonly = False
if args.verbose: print(' - Datafile only?      :',datafileonly)
#-------------------------------------------------------------------------------------------------------------
if args.chainstart:
    chainstart = [args.chainstart[0],args.chainstart[1],args.chainstart[2]]
    if args.verbose: print(' - Start MCMC chain at :',chainstart)
else:
    chainstart = None
#-------------------------------------------------------------------------------------------------------------
if args.emptysim:
    emptysim = True
else:
    emptysim = False
if args.verbose: print(' - Using empty sims?   :',emptysim)
#-------------------------------------------------------------------------------------------------------------
if args.LFredshift:
    LFredshift = args.LFredshift
else:
    LFredshift = 8.0
if args.verbose: print(' - What z is the LF at?:',LFredshift)
#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
if args.samprange:
    kmin           = np.float(args.samprange[0])
    kmax           = np.float(args.samprange[1])
    logLstarmin    = np.float(args.samprange[2])
    logLstarmax    = np.float(args.samprange[3])
    logNmin        = np.float(args.samprange[4])
    logNmax        = np.float(args.samprange[5])
else:
    kmin           =  -4.0
    kmax           =   4.0
    logLstarmin    =  -3.0
    logLstarmax    =   3.0
    logNmin        =   1.0
    logNmax        =  50.0
#-------------------------------------------------------------------------------------------------------------
# Loading data to be fitted
if args.verbose: print(' - Loading data and creating balff_mpd class')
mpdclass       = mpd.balff_mpd(args.datafile,lookuptable=lutabval, verbose=args.verbose,
                                 errdist=errdist,logLmin=-4,contamfrac=contamfrac,
                                 datafileonly=datafileonly,selectionfunction=selfctval,
                                 emptysim=emptysim,loadselfct=loadval,LFredshift=LFredshift)

data           = mpdclass.data['LOBJ']
Nobj           = mpdclass.Nobj
print('There are ',Nobj,' High z objects')

#-------------------------------------------------------------------------------------------------------------
# Stochastic variables (not detemined completely by the parents, i.e., drawn from a probability distribution)
if args.verbose: print(' - Defining probaility distributions for k, L* and Nobjuniverse draws')
docstr         = 'Theta containing the Schecter shape parameter k and log10 of the Schechter scale paramter L* in ' \
                 'units [1e44 erg/s] as well as log10 of the inferred number of galaxies in the Universe Nobjuniverse'
thetamin       = [kmin,logLstarmin,logNmin]
thetamax       = [kmax,logLstarmax,logNmax]

# with Model() as model: 
#     thetasample    = Uniform('theta', size=3, 
#                               lower=thetamin,upper=thetamax,
#                               observed=False,doc=docstr,verbose=0,value=chainstart
#                               )
thetasample = Parameters()
thetasample.add('k', value=(thetamin[0]+thetamax[0])/2, min=thetamin[0], max=thetamax[0], vary=True)
thetasample.add('logL', value=0, min=thetamin[1], max=thetamax[1], vary=True)
thetasample.add('logN', value=5, min=thetamin[2], max=thetamax[2], vary=True)
ndim = 3

if args.verbose: print(' - Sample range k      : [',thetamin[0],',',thetamax[0],']')
if args.verbose: print(' - Sample range logL*  : [',thetamin[1],',',thetamax[1],']')
if args.verbose: print(' - Sample range logN   : [',thetamin[2],',',thetamax[2],']')

#-------------------------------------------------------------------------------------------------------------
# Selecting a subset of the high-z candidates from a Bernoulli distribution provided a contamination fraction
useallcontam = 'yes'
if useallcontam == 'yes':
    bernoulliprob = 0.0
elif useallcontam == 'no':
    bernoulliprob = mpdclass.contamfracarr
if args.verbose: print(' - Defining HighZ canidates to include given contamination (use all? ',useallcontam,')')

from scipy.stats import bernoulli
# with Model() as model: 
#     HighZ = pymc.Bernoulli('highz', size=Nobj, p=1-bernoulliprob)
HighZ = bernoulli.rvs(1-bernoulliprob, size=Nobj)

#-------------------------------------------------------------------------------------------------------------
# Data model. The sampling of the data with drawn variables
if args.verbose: print(' - Defining mpd model of data  @ ',time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

# @pymc.stochastic(dtype=float,observed=True,plot=False)
# def mpdmodel(value=data, theta=thetasample, HighZ=HighZ):
#     """
#     The (simulated) data to reproduce with the model
#     Returns the log-likelihood of a 1D Schechter function with a
#     gaussian error term which depends on the shape parameter, k,
#     and the scale parameter L*/[1e44 erg/s]

#     -- PARAMETERS --
#     value  = data array loaded into the mpdclass
#     theta  = The parameters to sample over: k, logL*, logN
#     HighZ  = binary list of high z candidates to include
#              (taking contamination into account)
#     """
#     param     = [theta[0],10**theta[1],10**theta[2]]

#     if args.lnptime: start = time.time()

#     lnprob = mpdclass.getlnprob_field_contam(param,HighZ)

#     if args.lnptime:
#         stop  = time.time()
#         print('ln(prop) dt [s]  : ',stop-start)

#     return lnprob

def mpdmodel(theta, HighZ, f_val, f_resid,
             out=None, lnpreject=-np.inf):
    """
    The (simulated) data to reproduce with the model
    Returns the log-likelihood of a 1D Schechter function with a
    gaussian error term which depends on the shape parameter, k,
    and the scale parameter L*/[1e44 erg/s]

    -- PARAMETERS --
    value  = data array loaded into the mpdclass
    theta  = The parameters to sample over: k, logL*, logN
    HighZ  = binary list (1 or 0) of high z candidates to include
             (taking contamination into account)
    """
    if not f_val:
        theta = theta.valuesdict()
        param = [theta['k'], 10**theta['logL'], 10**theta['logN']]
    else:
        param = [theta[0],10**theta[1],10**theta[2]]

    if args.lnptime: start = time.time()

    lnprob = mpdclass.getlnprob_field_contam(param, HighZ)

    if f_resid:
        # lnprob = -0.5 * (np.sum(resid[con_res]**2)
        return lnprob / (-0.5)

    # Check range:
    if not out==None:
        ii = 0
        for key in out.params:
            if out.params[key].vary:
                cmin = out.params[key].min
                cmax = out.params[key].max
                if param[ii]<cmin or param[ii]>cmax or np.isnan(param[ii]):
                    # print('outside the range')
                    return lnpreject
                # theta[key].value = param[ii]
                ii += 1

    if args.lnptime:
        stop  = time.time()
        print('ln(prop) dt [s]  : ',stop-start)

    return lnprob


#-------------------------------------------------------------------------------------------------------------
# get a pymc sampler
if args.verbose: print(' - Initialize the pymc MCMC sampler')
contstr      = 'contamfrac'+str(contamfrac).replace('.','p')

if not os.path.isdir('./balff_output/'):
    if args.verbose: print(' - Did not find "./balff_output/" so creating it')
    os.mkdir('./balff_output/')

if args.lookuptable:  # set data base file to save to.
    dbname = './balff_output/'+args.lookuptable.split('/')[-1].replace('.npz','_pymcchains_'+contstr+'.pickle')
else:
    dbname = './balff_output/'+args.datafile.split('/')[-1].replace('.fits','_pymcchains_'+contstr+'.pickle')

if args.emptysim:
    dbname = dbname.replace('.pickle','_emptysim.pickle')

# MM = pymc.MCMC({'theta':thetasample,'mpdmodel':mpdmodel},db='pickle',dbname=dbname) # no sampling over HighZ
fit_name = 'nelder'

f_val, f_resid = False, True
out = minimize(mpdmodel, thetasample,
            args=(HighZ, f_val, f_resid),#self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust), 
            #   kwargs = {'HighZ':HighZ},
            method=fit_name)

# @@@
print(fit_report(out))

#-------------------------------------------------------------------------------------------------------------
# # selecting MCMC step method to use
# if args.verbose: print(' - Select step method for k logL* and logN samples')

# pdist = 'Normal' # Default proposal distribution
# if args.pdist:
#     pdist = args.pdist
# if args.verbose: print('   (Using the proposal distribution "',pdist,'")')

# if args.step == 'metropolis':
#     pdist_sd = 3.0
#     MM.use_step_method(pymc.Metropolis, thetasample, proposal_sd=pdist_sd, proposal_distribution=pdist)
# else: # use RAM as default
#     target_rate   = 0.4 # Desired acceptance rate; covariance matrix adjusted to obtain this.
#     dk    = np.abs(args.samprange[1]-args.samprange[0])
#     dlogL = np.abs(args.samprange[3]-args.samprange[2])
#     dlogN = np.abs(args.samprange[5]-args.samprange[4])
#     covar_guess = np.array([[ (dk/100.)**2, 0.0, 0.0],[ 0.0, (dlogL/100.)**2, 0.0],[ 0.0, 0.0, (dlogN/100.)**2]])
#     MM.use_step_method(RobustAdaptiveMetro,thetasample,target_rate,verbose=0,
#                        proposal_covar=covar_guess,proposal_distribution=pdist)

# if args.verbose:
#     print(' - Chosen step method for theta',MM.step_method_dict[thetasample])

#-------------------------------------------------------------------------------------------------------------
# # Before we start the MCMC sampler test to make sure that the RAM step is properly initialized.
# if args.step != 'metropolis':
#     RAM = MM.step_method_dict[thetasample][0]
#     assert RAM._dim == 3

#-------------------------------------------------------------------------------------------------------------
# DEFAULT values
if args.verbose: print(' - MCMC sampling start @ ',time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
Niter = 10 #10000
Nburn = int(Niter/2) # 1000
Nthin = 2
Nwalk = 30
# Overwrite if any given on the commandline
if args.Niter: Niter = args.Niter
if args.Niter: Nburn = args.Nburn
if args.Niter: Nthin = args.Nthin
if args.verbose: print('   (running with Niter =',Niter,', Nburn = ',Nburn,', Nthin = ',Nthin,')')

# MM.db # opening data base file to write to disk
# MM.sample(iter=Niter,burn=Nburn,thin=Nthin,progress_bar=True)      # sample the model
# if args.verbose: print('\n - MCMC sampling done  @ ',time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

# dbstate              = MM.db.getstate()['step_methods']

# if args.step == 'metropolis':
#     ent = 0
#     accept = dbstate[list(dbstate.keys())[ent]]['accepted']
#     reject = dbstate[list(dbstate.keys())[ent]]['rejected']
# else:
#     ent = 0
#     accept = dbstate[list(dbstate.keys())[ent]]['_accepted']
#     reject = dbstate[list(dbstate.keys())[ent]]['_rejected']

# acceptancerate_theta = accept / (reject+accept)

# if args.verbose:
#     print('      with theta = (k,L*,N) acceptance rate =',acceptancerate_theta)
#     print('      No. of accepted steps = '+str(accept))
#     print('      No. of rejected steps = '+str(reject))

# MM.db.close()
# if args.verbose: print(' - Saved chains to ',dbname)

f_val, f_resid = True, False
moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
sampler = emcee.EnsembleSampler(Nwalk, ndim, mpdmodel, \
    args=(HighZ, f_val, f_resid),\
    moves=moves,\
    kwargs={'out': out, 'lnpreject':-np.inf, 
    # 'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior, 'SNlim':self.SNlim, 'NRbb_lim':self.NRbb_lim},\
    }
    )

amp_shuffle = 1e-2
pos = amp_shuffle * np.random.randn(Nwalk, ndim)
aa = 0
for aatmp,key in enumerate(out.params.valuesdict()):
    if out.params[key].vary:
        pos[:,aa] += out.params[key].value
        aa += 1

sampler.run_mcmc(pos, Niter, progress=True)
flat_samples = sampler.get_chain(discard=Nburn, thin=Nthin, flat=True)

#-------------------------------------------------------------------------------------------------------------
# Plot sampler;
DIR_PLOT = './balff_plots/'
if not os.path.exists(DIR_PLOT):
    os.mkdir(DIR_PLOT)

_, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = []
for key in out.params.valuesdict():
    if out.params[key].vary:
        labels.append(key)

for i in range(ndim):
    if ndim>1:
        ax = axes[i]
    else:
        ax = axes
    ax.plot(sampler.get_chain()[:,:,i], alpha=0.3, color='k')
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel(labels[i])
if ndim>1:
    axes[-1].set_xlabel("step number")
else:
    axes.set_xlabel("step number")

plt.savefig('%s/chain.png'%(DIR_PLOT))
plt.close()
# For memory optimization;
del samples, axes

#-------------------------------------------------------------------------------------------------------------
#----------- Save pckl file
#-------- store chain into a cpkl file
burnin = int(Niter/2)
#burnin   = 0 # Since already burnt in.
savepath = DIR_PLOT

# Similar for nested;
# Dummy just to get structures;
print('\nRunning dummy sampler. Disregard message from here;\n')
f_val, f_resid = False, False
mini = Minimizer(mpdmodel, thetasample, 
        fcn_args=[HighZ, f_val, f_resid], 
        f_disp=False, nan_policy='omit',
        moves=moves,
        #kwargs={'f_val': True, 'NRbb_lim':self.NRbb_lim},\
        )

res = mini.emcee(burn=0, steps=3, thin=1, nwalkers=Nwalk, 
                params=out.params, is_weighted=True, ntemps=0, workers=0, 
                float_behavior='posterior', progress=False)
print('\nto here.\n')

# Update;
var_names = []#res.var_names
params_value = {}
ii = 0
for key in out.params:
    if out.params[key].vary:
        var_names.append(key)
        params_value[key] = np.nanmedian(flat_samples[Nburn:,ii])
        ii += 1
flatchain = pd.DataFrame(data=flat_samples[Nburn:,:], columns=var_names)

use_pickl = True
if use_pickl:
    cpklname = 'chain_corner.cpkl'
    savecpkl({'chain':flatchain,
                    'burnin':burnin, 'nwalkers':Nwalk,'niter':Niter,'ndim':ndim},
                    savepath+cpklname) # Already burn in

#-------------------------------------------------------------------------------------------------------------
if args.verbose: print(' - Plot logNobjuniverse, k and logLstar samples')
# plot(MM)                                         # plotting results

# thetafile = './balff_plots/'+dbname.split('balff_output/')[-1].replace('.pickle','_theta.png')
# if args.verbose: print(' - Moving theta_2.png to',thetafile)
# out = subprocess.getoutput('mv theta_2.png '+thetafile)

# if args.verbose: print('\n - Printing Summary')
# statval = 1 # resetting stat indicator
# ssval = MM.stats() # checking that stats can be created

# if (ssval['theta'] == None): statval = 0
# if statval == 1:
#     MM.theta.summary()
#     if args.lookuptable:
#         statcsv = './balff_output/'+args.lookuptable.split('/')[-1].replace('.npz','_pymcchains_stat_'+contstr+'.csv')
#     else:
#         statcsv = './balff_output/'+args.datafile.split('/')[-1].replace('.fits','_pymcchains_stat_'+contstr+'.csv')

#     if args.emptysim:
#         statcsv = statcsv.replace('.csv','_emptysim.csv')

#     if args.verbose: print('\n - Writing stat of chains to',statcsv)
#     MM.write_csv(statcsv, variables=["theta"])

#-------------------------------------------------------------------------------------------------------------
# quick plot of 3D sample space

levels = [0.68, 0.95, 0.997]
quantiles = [0.01, 0.99]
val_truth = []
for par in var_names:
    val_truth.append(params_value[par])

fig1 = corner.corner(flatchain, labels=var_names, 
                     label_kwargs={'fontsize':16}, quantiles=quantiles, show_titles=False, 
                     title_kwargs={"fontsize": 14}, 
                     truths=val_truth, plot_datapoints=False, plot_contours=True, no_fill_contours=True, 
                     plot_density=False, levels=levels, truth_color='gray', color='#4682b4'
                     )

fig1.savefig(DIR_PLOT + 'corner.png')

# qp2 = 1
# if qp2 == 1:
#     samplespacefig = DIR_PLOT+dbname.split('balff_output/')[-1].replace('.pickle','_theta_2Dsamplespace.png')
#     db = pymc.database.pickle.load(dbname)

#     kdrawn         = db.trace('theta')[:,0]
#     logLstardrawn  = db.trace('theta')[:,1]
#     logNdrawn      = db.trace('theta')[:,2]
#     print('k    unique', len(np.unique(kdrawn)))
#     print('logL unique', len(np.unique(logLstardrawn)))
#     print('logN unique', len(np.unique(logNdrawn)))

#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(12, 12))

#     plt.subplot(2, 2, 1)
#     plt.plot(kdrawn, logLstardrawn, 'ob',alpha=0.2)
#     plt.xlim(np.min(kdrawn),np.max(kdrawn))
#     plt.ylim(np.min(logLstardrawn),np.max(logLstardrawn))
#     plt.xlabel('k')
#     plt.ylabel('log10(L* / [1e44erg/s])')
#     if args.chainstart: plt.plot(chainstart[0],chainstart[1],'rs',ms=10)

#     plt.subplot(2, 2, 2)
#     plt.plot(logNdrawn, logLstardrawn, 'ob',alpha=0.2)
#     plt.xlim(np.min(logNdrawn),np.max(logNdrawn))
#     plt.ylim(np.min(logLstardrawn),np.max(logLstardrawn))
#     plt.xlabel('log10(N)')
#     plt.ylabel('log10(L* / [1e44erg/s])')
#     if args.chainstart: plt.plot(chainstart[2],chainstart[1],'rs',ms=10)

#     plt.subplot(2, 2, 3)
#     plt.plot(kdrawn, logNdrawn, 'ob',alpha=0.2)
#     plt.xlim(np.min(kdrawn),np.max(kdrawn))
#     plt.ylim(np.min(logNdrawn),np.max(logNdrawn))
#     plt.xlabel('k')
#     plt.ylabel('log10(N)')
#     if args.chainstart: plt.plot(chainstart[0],chainstart[2],'rs',ms=10)

#     plt.savefig(samplespacefig)
#     if args.verbose: print('\n - Saved ',samplespacefig)
#-------------------------------------------------------------------------------------------------------------
if args.verbose:
    print(' ')
    print(':: '+sys.argv[0]+' :: -- END OF PROGRAM -- ')
    print(' ')
#-------------------------------------------------------------------------------------------------------------



