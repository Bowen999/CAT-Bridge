from setuptools import setup, find_packages

setup(
    name="catbridge",
    version="2.0.0",
    packages=find_packages(),
    python_requires=">=3.9",

    # Metadata
    author="Bowen Yang",
    author_email="by172@georgetown.edu",
    description="CAT Bridge (Compounds And Transcripts Bridge) is a comprehensive longitudinal multi-omics analysis tool",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',  # This is important!
    url="https://github.com/Bowen999/CAT-Bridge",  # Optional
    project_urls={
        "Homepage": "https://catbridge.work",
        "Source": "https://github.com/Bowen999/CAT-Bridge",
    },
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License"
    ],

    # Third-party libraries imported by the package (checked against the modules):
    #   data_io     pandas (+ openpyxl for .xlsx, xlrd for .xls)
    #   preprocess  numpy, scipy, scikit-learn, pandas
    #   metrics     numpy, scipy, scikit-learn (kmeans_plusplus in ts_clustering);
    #               statsmodels (granger_list only); scikit-fuzzy (clustering) imported lazily.
    #               The metrics are computed in numpy and no longer need causal-ccm (CCM),
    #               fastdtw (DTW) or tslearn (ts_clustering) (CHANGELOG 2.0.0, "Speed and
    #               input checks"); the test suite compares against them when installed.
    #   foldchange  pydeseq2 (imported lazily in fc_comp)
    #   scoring     numpy, pandas, scipy, scikit-learn
    #   plotting    matplotlib, seaborn, networkx, scikit-learn; adjustText imported lazily
    #   (scipy, scikit-learn, matplotlib, seaborn and networkx are imported on first use)
    install_requires=[
        # pandas 3.0: pydeseq2 0.5.2 raises a TypeError/UFuncTypeError inside DeseqDataSet
        # (tested), so the fold change with a design file does not work there; fc_comp then
        # raises a RuntimeError that says so.
        "pandas>=1.5,<3",
        "numpy",
        "scipy",
        # An earlier development version pinned scikit-learn<1.6 because of tslearn 0.6.3 (force_all_finite TypeError);
        # tslearn is no longer used.
        "scikit-learn>=1.2",
        "statsmodels",
        "scikit-fuzzy",
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
