# Pytorch_Analyzer

This tool for Pytorch can analyze your model. 

Information include **layer memory usage, max memory usage, cumulative memory usage and execution time.**

# Requirement

>pytorch 
> 
>matplotlib (python -m pip install -U matplotlib)

# Analysis
##  Print the analysis.

    No.    Layer_memory    Max_memory        Memory      Exec_time    Layer
    Initial---------------------------------------------------------------------------------------------------
    0          61.18 MB     118.39 MB      61.18 MB     3971.82 us    Input, label etc.                  
    Forward---------------------------------------------------------------------------------------------------
    1        6400.00 kB      87.13 MB      67.43 MB       97.75 us    Conv2d(3, 32, kernel_size=(3,      
    2        6400.00 kB      73.68 MB      73.68 MB       29.56 us    LeakyReLU(negative_slope=0.01)     
    3        6400.00 kB      79.93 MB      79.93 MB       52.21 us    BatchNorm2d(32, eps=1e-05, mom     
    4        4800.00 kB      84.62 MB      84.62 MB       34.57 us    MaxPool2d(kernel_size=2, strid     
    5        3200.00 kB     101.52 MB      87.74 MB       77.01 us    Conv2d(32, 64, kernel_size=(3,     
    6        3200.00 kB      90.87 MB      90.87 MB       25.99 us    LeakyReLU(negative_slope=0.01)     
    7        3200.00 kB      93.99 MB      93.99 MB       44.82 us    BatchNorm2d(64, eps=1e-05, mom     
    8        2400.00 kB      96.34 MB      96.34 MB       30.04 us    MaxPool2d(kernel_size=2, strid     
    9        1600.00 kB     123.90 MB      97.90 MB       87.50 us    Conv2d(64, 128, kernel_size=(3     
    10       1600.00 kB      99.46 MB      99.46 MB       25.27 us    LeakyReLU(negative_slope=0.01)     
    11       1600.00 kB     101.02 MB     101.02 MB       43.15 us    BatchNorm2d(128, eps=1e-05, mo     
    12        800.00 kB     101.81 MB     101.81 MB       67.95 us    Conv2d(128, 64, kernel_size=(1     
    13        800.00 kB     102.59 MB     102.59 MB       25.27 us    LeakyReLU(negative_slope=0.01)     
    14        800.00 kB     103.37 MB     103.37 MB       42.92 us    BatchNorm2d(64, eps=1e-05, mom     
    15       1600.00 kB     130.93 MB     104.93 MB       75.82 us    Conv2d(64, 128, kernel_size=(3     
    16       1600.00 kB     106.49 MB     106.49 MB       25.03 us    LeakyReLU(negative_slope=0.01)     
    17       1600.00 kB     108.06 MB     108.06 MB       42.20 us    BatchNorm2d(128, eps=1e-05, mo     
    18       1200.00 kB     109.23 MB     109.23 MB       29.80 us    MaxPool2d(kernel_size=2, strid     
    19       3125.00 kB     112.28 MB     112.28 MB       72.24 us    Conv2d(128, 1000, kernel_size=     
    20       3125.00 kB     115.33 MB     115.33 MB       24.80 us    LeakyReLU(negative_slope=0.01)     
    21       3125.00 kB     118.38 MB     118.38 MB       43.39 us    BatchNorm2d(1000, eps=1e-05, m     
    22          0.00 kB     118.38 MB     118.38 MB       18.60 us    Flatten()                          
    23          2.00 kB     118.39 MB     118.39 MB       66.52 us    Linear(in_features=16000, out_     
    24          2.00 kB     118.39 MB     118.39 MB       25.27 us    LeakyReLU(negative_slope=0.01)     

## Plot the result

* Every layer memory usage

![png](/img/layer.png)

* Total memory usage (cumulative memory usage)

![png](/img/cumulative.png)

* Every layer execution time

![png](/img/exec_time.png)

# How to use

1. Import Pytorch_Analyzer in your code

``` python
from pytorch_analyzer import Pytorch_Analyzer
```

2. Construct Pytorch_Analyzer and input your model.

``` python
analyzer = Pytorch_Analyzer(Your_model)
```

3. Before training/inference, please **initialize analyzer.**

``` python
analyzer.initial()
outputs = Your_model(inputs)
```

4. Print the analysis.

``` python
analyzer.analysis()
```

5. Plot the analysis

``` python
analyzer.analysis_plot()
```

# Reference

<https://pytorch.org/docs/stable/cuda.html> \
<https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html>
