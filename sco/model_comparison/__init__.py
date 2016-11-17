####################################################################################################
# model_comparison/__init__.py
# The model comparison module of the standard cortical observer
# by William F. Broderick

"""

"""
from .core import (create_model_dataframe, create_setup_dict, create_images_dict,
                   create_brain_dict, visualize_model_comparison, _create_plot_df)
from compare_with_Kay2013 import compare_with_Kay2013
