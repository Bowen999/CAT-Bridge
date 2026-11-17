"""
Package Name: CAT Bridge (Compounds And Trancrips Bridge)
Author: Bowen Yang
email: by8@ualberta
Homepage: 
Version: 2.0.0
Description: CAT Bridge (Compounds And Transcripts Bridge) is a robust tool built with the goal of uncovering biosynthetic mechanisms in multi-omics data, such as identifying genes potentially involved in compound synthesis by incorporating metabolomics and transcriptomics data. 

For more detailed information on specific functions or classes, use the help() function on them. For example:
help(your_package_name.your_function_name)

This module is a thin facade: the implementation lives in the sibling modules
(data_io, preprocess, metrics, foldchange, scoring, plotting, llm, pipeline)
and every public name is re-exported here so that
``from catbridge import catbridge as cat; cat.compute_corr(...)`` keeps working.


#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
"""

from .data_io import *
from .preprocess import *
from .metrics import *
from .foldchange import *
from .scoring import *
from .plotting import *
from .llm import *
from .pipeline import *
