from setuptools import setup, find_packages

setup(
    name="breath-removal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    install_requires=[
        "torch>=2.5.1",
        "torchaudio>=2.5.1",
        "librosa>=0.10.2",
        "numpy>=1.26.4",
        "intervaltree>=3.1.0",
        "matplotlib>=3.0.0",
        "soundfile>=0.12.1",
        "click>=8.1.7",
        "requests>=2.25.1",  
        "tqdm>=4.65.0",      
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="l.liniewicz@gmail.com",
    description="A tool for detecting and removing breath sounds from audio",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/breath-removal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)