"""Package Setup"""

import setuptools

setuptools.setup(
    name="supervised",
    description="Supervised Learning with conctraceptive data set",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "yellowbrick", "mlxtend"],
    entry_points={"console_scripts": ["bank_loan=supervised.bank_loan:main", "smart_grid=supervised.smart_grid:main"]},
)