from setuptools import setup, find_packages

setup(
    name="interest-heterogeneity-network-formation",
    version="1.0.0",
    description="Interest Heterogeneity in Network Formation on Large Online Platforms",
    author="Shalemsuma Sumanthiran",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
    ],
)
