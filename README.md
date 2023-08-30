# example-neutrino

Spey plugin for the likelihood presented in [arXiv:2007.08526](https://arxiv.org/pdf/2007.08526.pdf)
where the likelihood has been implemented in the following form:

$$
\mathcal{L}(\mu, \theta) = \left[\prod_{i\in{\rm channels}}\prod_{j_i\in {\rm bins}}
        {\rm Poiss}(n^j|(\mu n_s^j + n_b^j)(1 + \theta^j\sigma_b^j))\right] \cdot
        \prod_{k\in{\rm nuis}}\mathcal{N}(\theta^k | 0, 1)
$$

Here $n$ stands for the data, $n_s$ and $n_b$ are the signal and background yields, $\sigma_b$
are the background uncertainties. $\mu$ is the parameter of interest and $\theta$ are the nuisance parameters.

This plug-in can be installed from GitHub with `pip`

```
python -m pip install --upgrade "git+https://github.com/SpeysideHEP/example-neutrino"
```

or from the locally cloned repository

```
python -m pip install --upgrade .
```

command and can be used with ``spey.get_backend("example.neutrino")`` function.

```python
import numpy as np
import spey

pdf_wrapper = spey.get_backend("example.neutrino")
stat_model = pdf_wrapper(
    signal_yields=np.array([12, 15]),
    background_yields=np.array([50.0, 48.0]),
    data=np.array([38, 47]),
    absolute_uncertainties=np.array([11, 25]),
)
stat_model.likelihood()
# 7.42425771274118
```
