# ==============================================================================
# EXAMPLE PARAMETER FILE
# This parameter file provides an example setup for running the WARPFIELD
# code. This file contains documentation for all possible parameters.
# @author: Jia Wei Teh
# ==============================================================================

# Notes:
# 
# 1. Most parameters have DEFAULT values. If not specified in the parameter file, 
# the DEFAULT value will instead be used.
# 
# 2. The parameters need not be specified in order. 
# 
# 3. Some parameters will only run under certain conditions; if the conditions 
# are not met, the parameter will be ignored. Conditions are documented in this
# file
# 
# 4. Format: [parameter] [4 spaces] [value]


# =============================================================================
# Basic information
# =============================================================================

# Model name?
# type: string
# DEFAULT: WARPFIELD_DEFAULT
model_name    example

# Directory for output files?
# WARNING: MUST provide absolute path.
# type: string
# DEFAULT: def_dir: directory in which WARPFIELD is being ran, i.e., path/to/warpfield/output/
out_dir    def_dir

# Output intensity level in terminal?
# type: int
# Available values:
# -- 0  only errors
# -- 1  basic output
# -- 2  everything (intense)
# DEFAULT: 1
verbose    1

# Output format?
# Available values:
# -- ASCII
# -- FITS (TBD)
# DEFAULT: ASCII
output_format    ASCII

# =============================================================================
# WARPFIELD main parameters
# =============================================================================
# Note: Here are the main parameters which WARPFIELD depends on. 

# Enable random input parameters?
# This option will provide randomized inputs of cloud mass (log_mCloud), star 
# forming effeciency (sfe), cloud density (n_cloud), metallicity.
# The user must also define the min/max limit of parameter, from which the 
# randomizer will draw (see parameters below with `rand` prefix).
# Setting this to 1 will cause the parameters log_mCloud, metallicity, n_cloud, and sfe 
# to be ignored.
# Available values:
# -- 0 Disable random input
# -- 1 Enable random input
# DEFAULT: 0 
rand_input    0

# Log cloud mass (unit: solar mass). 
# This will be the initial mass of the molecular cloud.
# This parameter will be ignored, if rand_input is set to 1.
# type: float
# DEFAULT: 6.0
log_mCloud    6.0

# Is log_mCloud given as mass BEFORE or AFTER cluster formation?
# Available values:
# -- 0 mCloud represents cloud mass AFTER star formation
# -- 1 mCloud represents cloud mass BEFORE star formation
# DEFAULT: 1
mCloud_beforeSF    1

# Star formation effeciency (SFE). 
# This sets the fraction of cloud mass that will be converted into the mass 
# of the initial star cluster.
# This parameter will be ignored, if rand_input is set to 1.
# type: float
# Available values: 
# 0 < SFE < 1
# DEFAULT: 0.01
sfe    0.01

# Density of cloud (unit: cm^-3). 
# This parameter will be ignored, if rand_input is set to 1.
# type: float
# DEFAULT: 1000
n_cloud    1000

# Cloud metallicity (unit: solar metallicity, $Z_\odot$).
# This parameter will be ignored, if rand_input is set to 1.
# type: float
# Available values:
# -- 0.15
# -- 1 (solar metallicity)
# DEFAULT: 0.15
metallicity    0.15

# Include stochastic sampling of IMF in the creation of initial cluster?
# This parameter will be ignored, if rand_input is set to 1.
# Available values:
# -- 0  Do not sample stochastically. Scale outputs by assuming that the IMF
#       is fully sampled.
# -- 1  Stochastically sample the IMF.
# DEFAULT: 0
stochastic_sampling    0

# How many iterations / how many SLUG cluster do you want to create?
# This parameter will be ignored if stochastic_sampling is set to 0 (false).
# As a rule of thumb we suggest n_trails = 1e6/mCluster = 1e6/(mCloud*sfe).
# In other words, for high mass clusters where we expect fully sampled IMF,
# there is no need for a lot of iterations; conversely, for low mass clusters
# where stochasticity is important, we require more iterations for better
# understanding of the distribution of outputs (such as the ionising luminosity).
# DEFAULT: 1
n_trials    1

# If rand_input is set to 1, what is the min/max range of cloud mass?
# Values will be drawn from a flat distribution in log space between given limits.
# Clearly, this option will be ignored if rand_input is set to 0. 
# DEFAULT: 5,7.47
rand_log_mCloud    5, 7.47

# If rand_input is set to 1, what is the min/max range of sfe?
# Values will be drawn from a flat distribution in log space between given limits.
# Clearly, this option will be ignored if rand_input is set to 0. 
# DEFAULT: 0.01, 0.10
rand_sfe   0.01, 0.10

# If rand_input is set to 1, what is the min/max range of cloud number density?
# Values will be drawn from a flat distribution in log space between given limits.
# Clearly, this option will be ignored if rand_input is set to 0. 
# DEFAULT: 100, 1000
rand_n_cloud    100., 1000.

# If rand_input is set to 1, what is the min/max range of cloud metallicity?
# Values will be drawn from a flat distribution in linear space between given limits.
# Clearly, this option will be ignored if rand_input is set to 0. 
# Right now, there are only two values, though.
# Available values:
# -- 0.15
# -- 1 (solar metallicity)
rand_metallicity    0.15, 1

# =============================================================================
# parameters for (re)collapsing events
# =============================================================================
# Note: Should event of recollapse occur, we provide users the ability to 
# tweak parameters that dictates subseqeunt expansion events.

# Start expansion again after recollapse?
# Available values:
# -- 0 (no re-expansion)
# -- 1 (re-expansion)
# DEFAULT: 1
mult_exp    0

# At what radius r_coll should recollapse occur? (unit: pc)
# If the shell has radius smaller than r_coll AND having negative velocity, 
# the cloud recollapses.
# type: float
# DEFAULT: 1.0
r_coll    1.0

# Form stars again after recollapse? This parameter will only take effect if 
# mult_exp is set to 1 (re-expansion). Otherwise, this will be ignored.
# Available values:
# -- 0 No starburst after collapse
# -- 1 Starburst occur; use the same sfe as the first expansion event.
# -- 2 Starburst occur; value of sfe is re-determined such that the 
#       specified sfe per free-fall time parameter is achieved (see sfe_tff).
# DEFAULT: 1
mult_SF    1

# Star formation efficiency per free-fall time. This parameter will only take 
# effect if mult_SF is set to 2. Otherwise, this will be ignored.
# See also mult_SF.
# type: float
# DEFAULT: 0.01
sfe_tff    0.01

# =============================================================================
# parameters for stellar evolution models
# =============================================================================

# Sets the initial mass function. This parameter takes in string of the .imf file,
# which contains the PDF of the imf function.
# type: str
# Available values:
# -- chabrier.imf
# -- kroupa.imf
# DEFAULT: kroupa.imf
imf    kroupa.imf

# Sets the stellar tracks used.
# DEFAULT: geneva
# rotation?
# BH cutoff?
# clustermass?
stellar_tracks    geneva


# =============================================================================
# parameters for density of cloud
# =============================================================================
# Note: the choice of density profile of cloud will affect the mass profile. 

# Set the core number density of the molecular cloud (unit: 1/cm^3)
# type: float
# DEFAULT: 1000
dens_cloud    1000

# What is the density profile of the cloud? How does density scale with radius?
# type: str
# Available values:
# -- pL_prof  power-law profile. 
# -- bE_prof  Bonnor-Ebert density profile.
# DEFAULT: bE_prof
dens_profile    bE_prof

# If bE_prof (Bonner-Ebert) is specified for dens_profile, then the user must
# also define the parameter g_BE = rho_centre/rho_edge, such that all clouds 
# exceeding this value are grativationally unstable. The corresponding mass, 
# according to literature, is the critical mass known as Bonner-Ebert mass. 
# See The Formation of Stars (Stahler and Palla 2004), p247.
# DEFAULT: 14.1
dens_g_bE    14.1

# If pL_prof (power law) is specified for dens_profile, then the user must 
# also define the power-law coefficient, alpha. Alpha is defined as follows:
#       rho_cloud(r):
#           = rho_0                         for r <= r_core
#           = rho_0 * ( r / r_core)**alpha  for r_core < r <= r_cloud
#           = rho_ambISM                    for r > r_cloud
# type: float
# Available values: 
# -2 <= alpha <= 0
# Here, alpha = 0 corresponds to an homogeneous cloud, 
# whereas alpha = -2 corresponds to single isothermal sphere.
# DEFAULT: -2
dens_a_pL    -2

# What is the core radius of the molecular cloud? (unit: pc)
# type: string
# TODO: Do not change as it has not been tested.
# DEFAULT: 0.099
dens_rcore    0.099

# =============================================================================
# parameters for fragmentation of cloud
# =============================================================================
# Note: In addition to energy loss due to cooling, the expansion of shell will 
# switch from energy-driven (phase 1) to momentum-driven (phase 2) if shell
# fragmentation occurs. Here we allow users to determine if such process
# occurs, and if so, tweak the parameters controlling them.

# Allow shell fragmentation?
# Available values:
# -- 0 No
# -- 1 Yes
# DEFAULT: 0
frag_enabled    0

# Minimum radius at which shell fragmentation is allowed to occur. (unit: r_shell)
# This is set such that fragmentation will not occur at early phases 
# when cluster is (with high probability) embedded in cloud.
# This parameter will be ignored if frag_enabled is set to 0 (false).
# type: float
# DEFAULT: 0.1 (10% r_shell)
frag_r_min    0.1

# Allow shell fragmentation due to gravitational collapse?
# This parameter will be ignored if frag_enabled is set to 0 (false).
# -- 0 No
# -- 1 Yes
# DEFAULT: 0
frag_grav    0

# What is coefficient for the equation of gravitational instability?
# This parameter will only take effect if both frag_grav and frag_enabled is 1.
# DEFAULT: 0.67 (We adopt values from McCray and Kafatos 1987)
# see https://articles.adsabs.harvard.edu/pdf/1987ApJ...317..190M
frag_grav_coeff    0.67

# Allow shell fragmentation due to Rayleigh-Taylor instability?
# I.e., fragmentation occurs when shell accelerates (r_shell_dotdot > 0).
# This parameter will be ignored if frag_enabled is set to 0 (false).
# -- 0 No
# -- 1 Yes
# DEFAULT: 0
frag_RTinstab    0

# Allow shell fragmentation due to density inhomogeneties?
# I.e., fragmentation occurs as soon as r_shell = r_cloud
# This parameter will be ignored if frag_enabled is set to 0 (false).
# -- 0 No
# -- 1 Yes
# DEFAULT: 0
frag_densInhom    0

# What is the cover fraction? I.e., what fraction of shell remains 
# to be used for calculation after fragmentation?
# This parameter will mainly affect the number density of the phase.
# This parameter will be ignored if frag_enabled is set to 0 (false).
# Available values:
# 0 < frag_cf <= 1
# -- 1 whole shell is considered in the next phase (i.e., momentum-driven)
# -- 0 there is no more shell
# DEFAULT: 1
frag_cf    1

# Take into account timescale for fragmentation?
# Available values:
# -- 0 no, approximate as instantaneous fragmentation
# -- 1 yes
# DEFAULT: 1
frag_enable_timescale    1

# =============================================================================
# parameters dictating the stopping of simulation
# =============================================================================

# Density at which the shell is considered dissolved. (unit: 1/cm^3)
# Shell with density below this threshold for an extended period of time 
# (see stop_t_diss) will be considered dissolved and indistinguishable from the 
# diffuse ambient ISM.
# Ideally, this should be the same value as n_ISM.
# type: float
# DEFAULT: 1
stop_n_diss    1

# How long after n_shell < n_diss is satistied (continually) that the 
# shell is considered dissolved? (unit: Myr)
# See also the stop_n_diss parameter.
# type: float
# DEFAULT: 1
stop_t_diss    1.0

# Maximum radius of shell expansion? (unit: pc)
# If shell radius exceeds this threshold, consider the shell destroyed and halt the simulation.
# Set to an arbritrary high value (> 1e3) if stopping is not desired.
# DEFAULT: 1e3 (at this point the galactic shear will have disrupted the cloud).
stop_r    1e3

# What is the maximum simulation running time?
# After this period of time, the simulation will stop running.
# Avoid values greater than the last possible SN (i.e., 44 Myr for single cluster)
# Available units:
# -- Myr 
# -- tff (free-fall time)
# DEFAULT value: 15.05
# DEFAULT unit: Myr
stop_t    15.05
stop_t_unit    Myr


# =============================================================================
# parameters for WARPFIELD outputs
# =============================================================================

# Write and save output?
# Available values:
# -- 0 Do not save output (why?)
# -- 1 Save output
# TODO: The ability to select what output you want. E.g., only output radius and Lbol data.
# This allows smaller size of output folder and to store only necessary data.
# DEFAULT: 1
write_main    1

# Save initial stellar properties obtained from SB99/SLUG?
# Available values:
# -- 0 Do not save output
# -- 1 Save output
# DEFAULT: 0
write_stellar_prop    0

# Save density and temperature structure of the bubble? 
# Available values:
# -- 0 Do not save output
# -- 1 Save output
# DEFAULT: 0
write_bubble    0




# write_shell




# =============================================================================
# parameters/constants for miscellaneous properties of cloud/ISM/simulation.
# =============================================================================
# Note: Here we provide users the ability to adjust the value of these parameters;
# however, they are (mostly) standard constants and are rarely changed. 
# Unless necessary, these parameters should be kept at the default value.

# Include the effect of gravity in phase I (energy phase)?
# Available values:
# -- 0 do not consider gravity
# -- 1 gravity please
# DEFAULT: 1
inc_grav    1

# Add fraction of mass injected into the cloud due to sweeping of cold material
# from protostars and disks inside star clusters?
# This will affect in particular the total mass loss rate of cluster, Mdot, and 
# consequently the escape velocity.
# type: float
# DEFAULT: 0.0 (i.e., no extra mass loss)
f_Mcold_W    0.0
f_Mcold_SN    0.0

# What is the velocity of supernova ejecta? (units: cm/s)
# type: float
# DEFAULT: 1e9
v_SN    1e9

# Dust cross-section at solar metallicity? (unit: cm^2)
# If non-solar metallicity is given, the repective dust cross-section, sigma_d,
# will be scaled linearly such that:
#           sigma_d = sigma_0 * (Z/Z_sol)
# DEFAULT: 1.5e-21; see Draine 2011.
sigma0    1.5e-21

# Metallicity below which there is no dust? (unit: solar metallicity)
# Consequently any Z < z_nodust we have sigmad = 0.
# DEFAULT: 0.05
z_nodust    0.05

# Mean mass per nucleus and the mean mass per particle? (unit: cgs)
# We assume the standard composition of 1 He atom every 10 H atoms.
# DEFAULT:  -- u_n = (14/11)*m_H
#           -- u_p = (14/23)*m_H
u_n    2.1287915392418182e-24
u_p    1.0181176926808696e-24

# Temperature of ionised and neutral H region? (unit: K)
# DEFAULT:  -- t_ion = 1e4
#           -- t_neu = 100
t_ion    1e4
t_neu    100

# What is the number density of the ambient ISM? (unit: 1/cm^3)
# type: float
# DEFAULT: 0.1 
n_ISM    0.1

# The Rosseland mean dust opacity kappa_IR. This parameter relates to the calculation 
# of tau_IR, the optical depth of the shell in the IR by:
#           tau_IR = kappa_IR * \int u_n * n_sh dr
# For simplicity we do not relate kappa_IR to dust temperature, but adopt a 
# constant value kappa_IR = 4 cm^2/g
# DEFAULT: 4
kappa_IR    4

# The thermalisation efficiency for colliding winds and supernova ejecta.
# See Stevens and Hartwell 2003 or Kavanagh 2020 for a review.
# The new mechanical energy will thus be:
#       Lw_new = thermcoeff * Lw_old
# DEFAULT: 1.0
thermcoeff_wind    1.0 
thermcoeff_SN    1.0 






