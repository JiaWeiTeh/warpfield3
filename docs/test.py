.. highlight:: rest

.. _sec-parameters:

Parameter Specifications
========================

Creating the parameter file
---------------------------

This section documents how one can create and format parameter files (``*.param``). Here are some basic rules:

* Most parameters have default values (given in ``[]`` below). If a parameter is not specified in the parameter file, the default parameter-value pair will be assumed.

* The parameters need not be specified in order. 

* Some parameters will only run under certain conditions; if the conditions are not met, these parameters will be ignored. These conditions, if present, are documented below.

* Lines beginning with ``#`` in the ``.param`` file will be ignored. 

* One can specify parameters in the file using the following syntax:

.. code-block:: console

    parameter1    value1
    parameter2    value2
    ...      
     
.. _ssec-basic-params:
    
Basic parameters
----------------

These sets the basic parameters for the run.

* ``model_name [WARPFIELD_DEFAULT]``: The model name. This sets the prefix for all output filenames. WARNING: Do not include spaces.
* ``out_dir [def_dir]``: The directory for output files. This is the absolute path in which output files will be stored. The default directory is the root directory in which WARPFIELD is being run.
* ``verbose [1]``: The output intensity level in terminal. ``1`` for minimal output, ``2`` for basic output, and ``3`` for everything. 
* ``output_format [ASCII]``: The output format. In the future we will implement options for FITS files.

Main parameters for WARPFIELD 
-----------------------------

These are the main parameters which WARPFIELD depends on. 

* ``rand_input [0]``: Enable random input parameters? ``1`` to enable randomiser, ``0`` to disable. If ``1``, WARPFIELD will generate randomised inputs of ``log_mCloud``, ``sfe``, ``nCore``, and ``metallicity``. The user must then define the limits of the randomiser:

    * ``rand_log_mCloud [5,7.47]``: Log cloud mass (unit: solar mass)
    * ``rand_sfe [0.01, 0.10]``: Star formation effeciency
    * ``rand_n_cloud [100, 1000]``: Number density of cloud core (unit: cm\ :math:`^{-3}`)
    * ``rand_metallicity [0.15, 1]``: Cloud metallicity (unit: solar metallicity, :math:`Z_\odot`)

* ``log_mCloud [6.0]``:  Log cloud mass (unit: solar mass). This will be the initial mass of the molecular cloud. This parameter is ignored if ``rand_input`` is set to ``1``.

    * ``mCloud_beforeSF [1]``: Is ``log_mCloud`` given as mass before or after cluster formation? ``0`` indicates after star formation, and ``1`` indicates indicates before.

* ``sfe [0.01]``: Star formation effeciency (``0 < sfe < 1``). This sets the fraction of cloud mass that is converted into the mass of the initial star cluster. This parameter is ignored if ``rand_input`` is set to ``1``.
* ``nCore [1000]``: Number density of cloud core (unit: cm\ :math:`^{-3}`). This parameter is ignored if ``rand_input`` is set to ``1``.
* ``metallicity [1]``: Cloud metallicity (unit: solar metallicity, :math:`Z_\odot`). Currently available values are 1 :math:`Z_\odot` and 0.15 :math:`Z_\odot`. This parameter is ignored if ``rand_input`` is set to ``1``.
* ``stochastic_sampling [0]``: Include stochastic sampling of IMF in the creation of initial cluster? If set to ``1``, apply scaling relations when computing cluster properties assuming that the IMF is fully sampled. This parameter is ignored if ``rand_input`` is set to ``1``.

Parameters for the density profile of the cloud
-----------------------------------------------


This section sets the density profile, :math:`\rho(r)`, of the cloud.

* ``dens_profile [bE_prof]``: How does the density scale with radius?
    
    * ``bE_prof``: Bonnor-Ebert sphere profile (see `Ebert 1955 <https://ui.adsabs.harvard.edu/abs/1955ZA.....37..217E/abstract>`_; `Bonnor 1956 <https://ui.adsabs.harvard.edu/abs/1956MNRAS.116..351B/abstract>`_).

        * ``dens_g_bE [14.1]``: If Bonnor-Ebert is selected, then the user must also define the parameter :math:`g_{\rm BE} = \frac{\rho\_{\rm centre}}{\rho\_{\rm edge}}`, such that all clouds exceeding this value are gravitationally unstable. The corresponding mass is the critical mass known as the Bonner-Ebert mass (e.g., see `Stahler and Palla 2004 <https://ui.adsabs.harvard.edu/abs/2004fost.book.....S/abstract>`_). 

    * ``pL_prof``: Power-law profile. Setting ``dens_a_pL = 0`` (see below) gives a homogeneous cloud, whereas ``dens_a_pL = -2`` gives an isothermal sphere. 

        * ``dens_a_pL [-2]``: If power-law is selected, then the user must also define the power-law coefficient :math:`\alpha`, which takes value between :math:`-2\leq\alpha<0`. Suppose that :math:`r_0` is the core radius, :math:`\rho_0` is the core density, and :math:`\rho_{\rm ambISM}` is the density of the ISM, then :math:`\alpha` is defined such that:

        .. math:: \rho_{\rm cloud}(r) = \left\{\begin{array}{lll} \rho_0 , & r \leq r_0 \\ \rho_0 \times (r / r_0)^\alpha, & r_0 < r \leq r_{\rm cloud} \\ \rho_{\rm ambISM}, & r > r_{\rm cloud} \end{array} \right.

        * ``dens_navg_pL [170]``: The average number density of the cloud (unit: cm\ :math:`^{-3}`).       
        
        
Other parameters
----------------

Here, we provide users the ability to adjust the value of these parameters;
however, they are (mostly) standard constants and are rarely being changed. 
Unless necessary, these parameters should be kept at their default values.


* ``mu_n [2.1287915392418182e-24]``: The mean mass per nucleus (unit: g). We assume the standard composition of 1 He atom every 10 H atoms. By default, :math:`\mu_{\rm n} = (14/11)m_{\rm H}`.
* ``mu_p [1.0181176926808696e-24]``: The mean mass per particle (unit: g). We assume the standard composition of 1 He atom every 10 H atoms. By default, :math:`\mu_{\rm p} = (14/23)m_{\rm H}`.
* ``nISM [10]``: The number density of the ambient ISM (unit: cm\ :math:`^{-3}`).       
* ``t_ion [1e4]``: Temperature of ionised region (unit: K).
* ``t_neu [1e2]``: Temperature of neutral region (unit: K).
* ``sigma0 [1.5e-21]``: Dust cross-section at solar metallicity (unit: cm\ :math:`^2`). Thus for other metallicities the dust cross section is scaled as :math:`\sigma_d = \sigma_0 * (Z/Z_\odot)`.
* ``z_nodust [0.05]``: Metallicity below which there is effectively no dust (i.e., :math:`\sigma_d = 0`. Unit: :math:`Z_\odot`). 
* ``gamma_adia [1.6666666666666667]``: The adiabatic index (:math:`\gamma_{\rm adia} = 5/3`).
* ``gamma_mag [1.3333333333333333]``: The effective magnetic adiabatic index (:math:`\gamma_{\rm mag} = 4/3`). Setting to ``0`` implies a constant magnetic field strength throughout the model, whereas ``4/3`` implies conservation of magnetic flux and is what would be expected in the absence of dynamo action or magnetic reconnection (sphere). See `Henney et al 2005 <https://ui.adsabs.harvard.edu/abs/2005ApJ...621..328H/abstract>`_, Appendix C.
* ``alpha_B [2.59e-13]``: The case B recombination coefficient (unit: cm\ :math:`^{3}`/s). See `Osterbrock and Ferland 2006 <https://ui.adsabs.harvard.edu/abs/2006agna.book.....O/abstract>`_.     

# # The Rosseland mean dust opacity kappa_IR. This parameter relates to the calculation 
# # of tau_IR, the optical depth of the shell in the IR by:
# #           tau_IR = kappa_IR * \int u_n * n_sh dr
# # For simplicity we do not relate kappa_IR to dust temperature, but adopt a 
# # constant value kappa_IR = 4 cm^2/g
# # DEFAULT: 4
# kappa_IR    4


# # The thermalisation efficiency for colliding winds and supernova ejecta.
# # See Stevens and Hartwell 2003 or Kavanagh 2020 for a review.
# # The new mechanical energy will thus be:
# #       Lw_new = thermcoeff * Lw_old
# # DEFAULT: 1.0
# thermcoeff_wind    1.0 
# thermcoeff_SN    1.0 

        
        

    




# # =============================================================================
# # parameters for (re)collapsing events
# # =============================================================================
# # Note: Should event of recollapse occur, we provide users the ability to 
# # tweak parameters that dictates subseqeunt expansion events.

# # Start expansion again after recollapse?
# # Available values:
# # -- 0 (no re-expansion)
# # -- 1 (re-expansion)
# # DEFAULT: 1
# mult_exp    0

# # At what radius r_coll should recollapse occur? (unit: pc)
# # If the shell has radius smaller than r_coll AND having negative velocity, 
# # the cloud recollapses.
# # type: float
# # DEFAULT: 1.0
# r_coll    1.0

# # Form stars again after recollapse? This parameter will only take effect if 
# # mult_exp is set to 1 (re-expansion). Otherwise, this will be ignored.
# # Available values:
# # -- 0 No starburst after collapse
# # -- 1 Starburst occur; use the same sfe as the first expansion event.
# # -- 2 Starburst occur; value of sfe is re-determined such that the 
# #       specified sfe per free-fall time parameter is achieved (see sfe_tff).
# # DEFAULT: 1
# mult_SF    1

# # Star formation efficiency per free-fall time. This parameter will only take 
# # effect if mult_SF is set to 2. Otherwise, this will be ignored.
# # See also mult_SF.
# # type: float
# # DEFAULT: 0.01
# sfe_tff    0.01

# # =============================================================================
# # parameters for stellar evolution models
# # =============================================================================

# # Sets the initial mass function. This parameter takes in string of the .imf file,
# # which contains the PDF of the imf function.
# # type: str
# # Available values:
# # -- chabrier.imf
# # -- kroupa.imf
# # DEFAULT: kroupa.imf
# imf    kroupa.imf

# # Sets the stellar tracks used.
# # DEFAULT: geneva
# # rotation?
# # BH cutoff?
# # clustermass?
# stellar_tracks    geneva




# # =============================================================================
# # parameters for fragmentation of cloud
# # =============================================================================
# # Note: In addition to energy loss due to cooling, the expansion of shell will 
# # switch from energy-driven (phase 1) to momentum-driven (phase 2) if shell
# # fragmentation occurs. Here we allow users to determine if such process
# # occurs, and if so, tweak the parameters controlling them.

# # Allow shell fragmentation?
# # Available values:
# # -- 0 No
# # -- 1 Yes
# # DEFAULT: 0
# frag_enabled    0

# # Minimum radius at which shell fragmentation is allowed to occur. (unit: r_shell)
# # This is set such that fragmentation will not occur at early phases 
# # when cluster is (with high probability) embedded in cloud.
# # This parameter will be ignored if frag_enabled is set to 0 (false).
# # type: float
# # DEFAULT: 0.1 (10% r_shell)
# frag_r_min    0.1

# # Allow shell fragmentation due to gravitational collapse?
# # This parameter will be ignored if frag_enabled is set to 0 (false).
# # -- 0 No
# # -- 1 Yes
# # DEFAULT: 0
# frag_grav    0

# # What is coefficient for the equation of gravitational instability?
# # This parameter will only take effect if both frag_grav and frag_enabled is 1.
# # DEFAULT: 0.67 (We adopt values from McCray and Kafatos 1987)
# # see https://articles.adsabs.harvard.edu/pdf/1987ApJ...317..190M
# frag_grav_coeff    0.67

# # Allow shell fragmentation due to Rayleigh-Taylor instability?
# # I.e., fragmentation occurs when shell accelerates (r_shell_dotdot > 0).
# # This parameter will be ignored if frag_enabled is set to 0 (false).
# # -- 0 No
# # -- 1 Yes
# # DEFAULT: 0
# frag_RTinstab    0

# # Allow shell fragmentation due to density inhomogeneties?
# # I.e., fragmentation occurs as soon as r_shell = r_cloud
# # This parameter will be ignored if frag_enabled is set to 0 (false).
# # -- 0 No
# # -- 1 Yes
# # DEFAULT: 0
# frag_densInhom    0

# # What is the cover fraction? I.e., what fraction of shell remains 
# # to be used for calculation after fragmentation?
# # This parameter will mainly affect the number density of the phase.
# # This parameter will be ignored if frag_enabled is set to 0 (false).
# # Available values:
# # 0 < frag_cf <= 1
# # -- 1 whole shell is considered in the next phase (i.e., momentum-driven)
# # -- 0 there is no more shell
# # DEFAULT: 1
# frag_cf    1

# # Take into account timescale for fragmentation?
# # Available values:
# # -- 0 no, approximate as instantaneous fragmentation
# # -- 1 yes
# # DEFAULT: 1
# frag_enable_timescale    1



# # =============================================================================
# # parameters for WARPFIELD outputs
# # =============================================================================

# # Write and save output?
# # Available values:
# # -- 0 Do not save output (why?)
# # -- 1 Save output
# # TODO: The ability to select what output you want. E.g., only output radius and Lbol data.
# # This allows smaller size of output folder and to store only necessary data.
# # DEFAULT: 1
# write_main    1

# # Save initial stellar properties obtained from SB99/SLUG?
# # Available values:
# # -- 0 Do not save output
# # -- 1 Save output
# # DEFAULT: 0
# write_stellar_prop    0

# # Save density and temperature structure of the bubble? 
# # Available values:
# # -- 0 Do not save output
# # -- 1 Save output
# # DEFAULT: 0
# write_bubble    0

# # Create bubble.in file for CLOUDY?
# # Available values:
# # -- 0 Do not save output
# # -- 1 Save output
# # DEFAULT: 0
# write_bubble_CLOUDY    0

# # Save structure of the shell? 
# # Available values:
# # -- 0 Do not save output
# # -- 1 Save output
# # DEFAULT: 0
# write_shell    0


# # =============================================================================
# # parameters for integrators
# # =============================================================================




# # =============================================================================
# # parameters for bubble structures
# # =============================================================================
# # Note: This section includes parameters dictating the computation of bubble structure.

# # What is the relative radius xi = r/R2, at which to measure the bubble temperature?
# # See Weaver+77, Equation 23.  
# # Available values:
# # 0 < xi_Tb < 1
# # DEFAULT: 0.99
# xi_Tb    0.99


# # =============================================================================
# # parameters/constants for miscellaneous properties of cloud/ISM/simulation.
# # =============================================================================
# # Note: Here we provide users the ability to adjust the value of these parameters;
# # however, they are (mostly) standard constants and are rarely changed. 
# # Unless necessary, these parameters should be kept at the default value.

# # Include the effect of gravity in phase I (energy phase)?
# # Available values:
# # -- 0 do not consider gravity
# # -- 1 gravity please
# # DEFAULT: 1
# inc_grav    1

# # Add fraction of mass injected into the cloud due to sweeping of cold material
# # from protostars and disks inside star clusters?
# # This will affect in particular the total mass loss rate of cluster, Mdot, and 
# # consequently the escape velocity.
# # type: float
# # DEFAULT: 0.0 (i.e., no extra mass loss)
# f_Mcold_W    0.0
# f_Mcold_SN    0.0

# # What is the velocity of supernova ejecta? (units: cm/s)
# # type: float
# # DEFAULT: 1e9
# v_SN    1e9

# # Dust cross-section at solar metallicity? (unit: cm^2)
# # If non-solar metallicity is given, the repective dust cross-section, sigma_d,
# # will be scaled linearly such that:
# #           sigma_d = sigma_0 * (Z/Z_sol)
# # DEFAULT: 1.5e-21; see Draine 2011.
# sigma0    1.5e-21

# # Metallicity below which there is no dust? (unit: solar metallicity)
# # Consequently any Z < z_nodust we have sigmad = 0.
# # DEFAULT: 0.05
# z_nodust    0.05

# # Mean mass per nucleus and the mean mass per particle? (unit: cgs, i.e. g)
# # We assume the standard composition of 1 He atom every 10 H atoms.
# # DEFAULT:  -- mu_n = (14/11)*m_H
# #           -- mu_p = (14/23)*m_H
# mu_n    2.1287915392418182e-24
# mu_p    1.0181176926808696e-24

# # Temperature of ionised and neutral H region? (unit: K)
# # DEFAULT:  -- t_ion = 1e4
# #           -- t_neu = 100
# t_ion    1e4
# t_neu    100

# # What is the number density of the ambient ISM? (unit: 1/cm^3)
# # type: float
# # DEFAULT: 0.1 
# nISM    10

# # The Rosseland mean dust opacity kappa_IR. This parameter relates to the calculation 
# # of tau_IR, the optical depth of the shell in the IR by:
# #           tau_IR = kappa_IR * \int u_n * n_sh dr
# # For simplicity we do not relate kappa_IR to dust temperature, but adopt a 
# # constant value kappa_IR = 4 cm^2/g
# # DEFAULT: 4
# kappa_IR    4

# # What is the adiabatic index?
# # DEFAULT: 5/3
# gamma_adia    1.6666666666666667

# # The thermalisation efficiency for colliding winds and supernova ejecta.
# # See Stevens and Hartwell 2003 or Kavanagh 2020 for a review.
# # The new mechanical energy will thus be:
# #       Lw_new = thermcoeff * Lw_old
# # DEFAULT: 1.0
# thermcoeff_wind    1.0 
# thermcoeff_SN    1.0 

# # The case B recombination coefficient (unit: cm3/s)
# # Osterbrock and Ferland 2006
# alpha_B    2.59e-13

# # The effective magnetic adiabatic index?
# # Available values:
# # 0: Implies a constant magnetic field strength throughout the model.
# # 4/3: Implies conservation of magnetic flux and is what would be expected 
# #       in the absence of dynamo action or magnetic reconnection. (sphere)
# # See Henney et al 2005 Apped C: https://ui.adsabs.harvard.edu/abs/2005ApJ...621..328H/abstract
# # DEFAULT: 4/3
# gamma_mag    1.3333333333333333

# # BMW, nMW
# # TODO
# log_BMW    -4.3125
# log_nMW    2.065

# # What the thermal confuction coefficient C? (units: cgs)
# # C, where thermal conductivity k = C * T**(5/2), see Spitzer 1962. 
# # C is a weak function of temperature, but can be treated as constant.
# # Available values: 
# # -- 1.2e-6 (used in Weaver 1977 and Harper-Clark & Murray 2009, 
# #    presumably cited from Spitzer 1962 Table 5.1)
# # -- 6e-7 (used in  MacLow 1988, in agreement with Spitzer 1956)
# # DEFAULT: 1.2e-6
# c_therm    1.2e-6

# * ``rCore [0.099]``: What is the core radius of the molecular cloud? (unit: pc)










