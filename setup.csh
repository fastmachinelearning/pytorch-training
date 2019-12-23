if ( ! ($?PYTHONPATH)) then 
  setenv PYTHONPATH  `pwd`/models:
else
  setenv PYTHONPATH `pwd`/models:$PYTHONPATH
endif

#============================================================================#
#---------------------------   Getting the data   ---------------------------#
#============================================================================#
if ( ! -d data/ ) then
  mkdir -p data
endif
if ( ! -f data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z ) then
  wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z 
endif
