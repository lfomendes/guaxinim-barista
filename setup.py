"""
Setup script for the Guaxinim package.
Contains package metadata and dependencies.
"""

from setuptools import setup, find_packages

setup(
    name="guaxinim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "openai",
        "python-dotenv",
    ],
    entry_points={
        'console_scripts': [
            'guaxinim=guaxinim.ui.home:main',
        ],
    },
)
