# Defensive ML + Maya

## Requirements
1. Install PARSEC 3.0 benchmark and modify the path to parsecmgmt in Scripts/Launch.sh, from line 184.
2. Follow instructions of Maya-README.md to set up the Maya framework in your system.
3. Change the --logdir paths in aml_experiment.sh, experiment.sh, and maya_experiment.sh
4. Install libtorch API and configure TORCHDIR in Makefile accordingly.

## Instructions
1. Run ```wrapper.sh 0 [samplecount]``` to collect Maya traces for PARSEC benchmark runs. Modify the numbers if needed.
2. Run ```python DefenderGAN.py``` to train the ML defender.
3. Follow colldown.ipynb to get the jit-traced cpuscript_parsec_rnn2_64.pt. Copy it under Controller/ directory.
4. Run ```aml_wrapper.sh 0 [samplecount]``` to collect the traces with ml defender activated.
5. Run ```maya_wrapper.sh 0 [samplecount]``` to collect the traces with Maya defense activated.
6. Run ```MayaDataset.py [logs|aml_logs|maya_logs]``` to evaluate the defense results