from setuptools import setup, find_packages

setup(
    name="catbridge",
    version="1.1.0",
    packages=find_packages(),
    python_requires=">=3.9",

    # Metadata
    author="Bowen Yang",
    author_email="by172@georgetown.edu",
    description="CAT Bridge (Compounds And Transcripts Bridge) is a comprehensive longitudinal multi-omics analysis tool",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important!
    url="https://github.com/Bowen999/CAT-Bridge",  # Optional
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License"
    ],

    # Third-party libraries imported by the package (checked against the modules):
    #   data_io     pandas (+ openpyxl for .xlsx, xlrd for .xls)
    #   preprocess  numpy, scipy, scikit-learn, pandas
    #   metrics     statsmodels, causal-ccm, scikit-learn, fastdtw, scipy;
    #               scikit-fuzzy (clustering) and tslearn (ts_clustering) are imported lazily
    #   foldchange  pydeseq2 (imported lazily in fc_comp)
    #   scoring     numpy, pandas, scipy, scikit-learn
    #   plotting    matplotlib, seaborn, networkx, scikit-learn; adjustText imported lazily
    install_requires=[
        # pandas 3.0: pydeseq2 0.5.2 raises UFuncTypeError inside DeseqDataSet (tested), so
        # the fold change with a design file does not work there.
        "pandas>=1.5,<3",
        "numpy",
        "scipy",
        # scikit-learn >= 1.6: tslearn 0.6.3 (TimeSeriesKMeans, used by ts_clustering) raises
        # a `force_all_finite` TypeError (tested).
        "scikit-learn>=1.2,<1.6",
        "statsmodels",
        "causal-ccm",
        "fastdtw",
        "scikit-fuzzy",
        "tslearn",
        "pydeseq2",
        "matplotlib",
        "seaborn",
        "networkx",
        "adjustText",
        "openpyxl",   # pandas.read_excel for .xlsx (read_upload)
        "xlrd",       # pandas.read_excel for .xls (read_upload)
    ],
    extras_require={
        # catbridge.llm.Yuanfang only. It uses the legacy `openai.ChatCompletion` interface,
        # which was removed in openai 1.0, hence the upper bound.
        'llm': ['openai<1.0'],
    },
)
