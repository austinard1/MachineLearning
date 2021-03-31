"""Package Setup"""

import setuptools

setuptools.setup(
    name="unsupervised",
    description="Unsupervised learning",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "yellowbrick", "mlxtend"],
    entry_points={"console_scripts": ["unsupervised=unsupervised.unsupervised:main"]},
)