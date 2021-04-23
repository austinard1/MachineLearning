"""Package Setup"""

import setuptools

setuptools.setup(
    name="reinforcement",
    description="MDP and Reinforcement learning",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "yellowbrick", "mlxtend"],
    entry_points={"console_scripts": ["reinforcement=reinforcement.reinforcement:main"]},
)