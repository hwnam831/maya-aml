# Defensive ML + Maya

This repository integrates Defensive ML techniques into Maya.  
Most of the framework remain the same, except that we replace Maya's Gaussian Sinusoid planner with a Defensive ML defender.
You can do either of followings:
- Use the pre-trained defender to protect the system from power
- Train a new defender by collecting the power traces from your system

## Requirements
1. Install PARSEC 3.0 benchmark and modify PARSECDIR in Scripts/Launch.sh, line 184.
2. Follow instructions of Maya-README.md to set up the Maya framework in your system.
3. Install [PyTorch](https://pytorch.org/). Running the ML-related code requires a CUDA-capable GPU.
4. Install libtorch API and set TORCHDIR in Makefile to the directory where libtorch is installed.
5. ```make``` will build the system.


## Defending the system with the pre-trained defender
1. Running ```./defender_wrapper.sh 0 [samplecount]``` will turn on the Maya defense with Defensive ML, and collect the PARSEC benchmark traces while the defense is active. Samplecount should be larger than 100 for the next step. 
2. Running ```python MayaDataset.py --victimdir defender_logs``` will emulate the attack on the power side-channel, giving the CNN accuracy.

## Training a new defender

1. Run ```./wrapper.sh 0 [samplecount]``` to collect unprotected traces for PARSEC benchmark runs. Modify the numbers if needed.
2. Run ```python DefensiveML.py``` to train the ML defender.
3. Change name of the output pth file ```cp parsec_shaper_64_[...].pth best_shaper_64.pth```
4. Follow shaperToScript.ipynb to get the jit-traced cpuscript_parsec_shaper_64.pt. Copy it under Controller/ directory.  
    ```cp cpuscript_parsec_shaper_64.pt Controller/ssvFast3_Shaper_64.pt```  
    Or, you can use the provided ```Controller/ssvFast3_Shaper_64.pt ```
5. Run ```./defender_wrapper.sh 0 [samplecount]``` to collect the traces with the defender activated.
6. Run ```./maya_wrapper.sh 0 [samplecount]``` to collect the traces with Maya defense activated.
7. Run ```MayaDataset.py --victimdir [defender_logs|maya_logs]``` to evaluate the defense results for ML defense (defender\_logs) or Maya defense (maya\_logs)