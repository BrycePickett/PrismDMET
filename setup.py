from setuptools import setup

setup(
    name="prismdmet",
    version="1.4.0",
    description="PrismDMET: Density Matrix Embedding Theory for ab initio quantum chemistry",
    author="Bryce Pickett",
    package_dir={"prismdmet": "src"},
    packages=["prismdmet", "prismdmet.solvers", "prismdmet.qmmm"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pyscf",
    ],
)
