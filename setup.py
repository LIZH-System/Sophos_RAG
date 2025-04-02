"""
Setup script for Sophos RAG.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="sophos_rag",
    version="0.1.0",
    author="LIZH-System",
    author_email="your.email@example.com",
    description="A Retrieval-Augmented Generation (RAG) system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LIZH-System/Sophos_RAG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sophos_rag=sophos_rag.cli:main",
        ],
    },
)

if __name__ == "__main__":
    setup() 