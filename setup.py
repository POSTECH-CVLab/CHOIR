"""Setup script for CHOIR package."""

from setuptools import setup, find_packages


setup(
    name="choir",
    packages=find_packages(include=["src*"]),
    include_package_data=True,
    version="1.0.0",
    description="Stable and Consistent Prediction of 3D Characteristic Orientation via Invariant Residual Learning",
    author="Seungwook Kim, Chunghyun Park, Jaesik Park, Minsu Cho",
    url="https://github.com/chrockey/CHOIR",
    python_requires=">=3.10",
)
