from setuptools import setup

setup(
    entry_points={
        "spey.backend.plugins": [
            "example.neutrino = example_neutrino:NeutrinoExperiment"
        ]
    }
)
