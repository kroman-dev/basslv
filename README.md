# basslv

This repository is a fork of https://github.com/igudav/Bass-Local-Volatility

Differences:
1. Added stub to solve fixed-point equation for arbitrary tensors.
2. Completely redesigned the project architecture.
3. Add MarketMarginal class (correctly work for synthetic call prices from Black and Heston)
4. The speed of solving the fixed point equation and path generation has been improved by interpolating the inner convolution; this also resolves the OOM (Out-of-Memory) issue during path generation.

TODO:
1) It remains unclear why the solution to the fixed point equation is shifted from 0.5. Because of this, we employ a manual fix that shifts the solution to ensure F(0) = 0.5.
2) Improve interpolation and extrapolation of fixed point equation solution
3) Greeks calculation

Structure:
```
├── notebooks                                      <- examples in jupyter notebook
├── unittest                                       <- some tests
├── examples                                       <- examples
└── basslv   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes the basslv a Python module
    │
    ├── core
    │   ├── __init__.py
    │   ├── bassLocalVolatility.py                 <- contains the complete algorithm including Monte Carlo sampling
    │   ├── fixedPointEquation.py                  <- contains the numerical algorithm for building the mapping function
    │   ├── genericMarginal.py                     <- abstract class
    │   ├── heatKernelConvolutionEngine.py         <- contains methods to calculate convolution
    │   ├── logNormalMarginal.py                   <- exact case 
    │   ├── marketMarginal.py                      <- build marginal from market call prices
    │   ├── projectTyping.py                       <- auxiliary typing
    │   ├── solutionFixedPointEquation.py          <- realization of SolutionInterpolator  
    │   └── solutionInterpolator.py                <- abstract class - solution of fixed point equation
    │   
    └── visualVerification  
         ├── __init__.py
         └── visualVerification.py                 <- contains methods for visual verification
```

**References:**
1) Antoine Conze and Henry-Labordere, A new fast local volatility model: \
    https://www.risk.net/media/download/1079736/download
2) Antoine Conze and Henry-Labordere, Bass Construction with Multi-Marginals:  
    Lightspeed Computation in a New Local Volatility Model: \
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3853085

Note. The references in the code are linked to the first article
