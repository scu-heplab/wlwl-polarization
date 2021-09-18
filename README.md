# WlWl-Polarization
Measure WWjj polarization fraction 

Paper:xxxxxxxxxxxxxx

#### Notice: This code can only use the inference process, if you want to train your own model, please contact [zhangrao@scu.stu.edu.cn](mailto:zhangrao@scu.stu.edu.cn).

## Requirements
* Both Linux and Windows are supported.
* 64-bit Python3.6(or higher) installation.
* Tensorflow2.x(recommend 2.4), Numpy.
* One or more high-end NVIDIA GPUs(at least 4 GB of DRAM), NVIDIA drivers, CUDA 11.0 toolkit and cuDNN 8.0.

## Prepare dataset
* The dataset is stored in `./raw/`, the data structure is as follows:
```
One event for every 6 lines:
   1. first lepton 
   2. second lepton 
   3. first FB jet 
   4. second FB jet 
   5. MET 
   6. remaining jet 
Each line has the following five columns of elements:
   1.ParticleID  2.Px  3.Py  4.Pz  5.E
The format of an event in the dataset is as follows:
   ...
   -1  166.023   5.35817   10.784    166.459
   1   -36.1648  -64.1513  -28.9064  79.113
   7   -11.3233  -39.6316  -318.178  320.85
   7   -34.2795  22.0472   622.79    624.128
   0   -22.6711  52.8976   -422.567  426.468
   6   -49.9758  29.3283   274.517   294.098
   ...
```
ParticleID: 1 for electron, 2 for muon, 3 for tau, 4 for b-jet, 5 for normal jet, 0 for met, 6 for remaining jets, 7 for forward backward jet, signs represent electric charge.
* Use the command `python create_dataset YOUR_RAWDATA_PATH`
## Using pre-trained models
* Pre-training weights are placed in `./weights/`.

* 123
