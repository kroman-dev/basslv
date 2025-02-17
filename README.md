# Bass Local Volatility

Python implementation (incomplete) of the paper [Bass Construction with Multi-Marginals: Lightspeed Computation in a New Local Volatility Model
](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3853085).
Only the basic local volatility model is implemented. 
Tested with analytical marginals from the Black-Scholes model (see [demo](demo.ipynb).
You can find some logs of the development process in [dev notebook](dev.ipynb) with debugging plots.
Issues and pull requests are welcome.

Current issues:
1. Fixed-point iteration (equation (2) in the paper) is divergent now for tenors > 10. Even for Black-Scholes marginals the solution is visibly inaccurate (see [demo](demo.ipynb)).
2. Interpolations and inverse CDFs definetely can be improved.
