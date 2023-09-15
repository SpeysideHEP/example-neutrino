import numpy as np
import spey
from pytest import approx


def test_readme_example():
    pdf_wrapper = spey.get_backend("example.neutrino")
    stat_model = pdf_wrapper(
        signal_yields=np.array([12, 15]),
        background_yields=np.array([50.0, 48.0]),
        data=np.array([38, 47]),
        absolute_uncertainties=np.array([11, 25]),
    )
    assert stat_model.likelihood() == approx(7.42425771274118)
