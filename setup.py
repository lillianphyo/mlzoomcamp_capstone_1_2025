from setuptools import setup, find_packages

setup(
    name="ml_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "prefect",
        "python-dotenv",
        "matplotlib",
        "seaborn",
        "flask",
        "joblib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    python_requires=">=3.7",
    include_package_data=True,
)