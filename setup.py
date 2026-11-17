from setuptools import setup, find_packages

setup(
    name="catbridge",
    version="1.0",
    packages=find_packages(),

    # Metadata
    author="Bowen Yang",
    author_email="by172@georgetown.edu",
    description="CAT Bridge (Compounds And Transcripts Bridge) is a comprehensive longitudinal multi-omics analysis tool",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important!
    url="https://github.com/Bowen999/CAT-Bridge",  # Optional
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ],

    install_requires=[
        # List your package's dependencies here
    ],
)
