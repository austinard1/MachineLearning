"""Package Setup"""

import setuptools

setuptools.setup(
    name="optimization",
    description="Optimization problems and fitness functions",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "yellowbrick", "mlxtend", "networkx", "mlrose-hiive"],
    entry_points={"console_scripts": ["optimization=optimization.optimization:main"]},
)