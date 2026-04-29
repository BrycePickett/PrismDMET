from setuptools import setup, find_namespace_packages

setup(
    name="prismdmet",
    version="1.3.0",
    description="PrismDMET: Advanced Density Matrix Embedding Theory for Materials",
    author="PrismDMET Team",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pyscf"
    ],
)
