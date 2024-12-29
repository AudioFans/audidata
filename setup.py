from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="audidata", # Replace with your own username
    version="0.0.1",
    author="Qiuqiang Kong",
    author_email="qiuqiangkong@gmail.com",
    description="Audio dataloaders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AudioFans/audidata",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "torch", "torchaudio", "pretty_midi", "pandas"],
    python_requires='>=3.5',
)