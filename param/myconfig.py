

#include coverfraction? And with which end value?
frag_cover=True
cf_end=0.1

r_Tb=0.90 #temperature measure radius (<1!)

#relative tolerance of stiffest ODE solver (energy phase)
rtol=1e-3 #decrease to 1e-4 if you have crashes --> slower but more stable

###################################################
# folder to save output in
basedir =    './model_test_output/' 


#number of processors to use
n_proc = 1
# full path to warpfield code files
path_to_code = "./"