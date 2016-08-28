####################################################################################################
# stimulus/__init__.py
# The image module of the standard cortical observer.
# By Noah C. Benson

'''
The sco.stimulus module of the standard cortical observer library is responsible for processing the
stimulus shown to the subject. Although this project was written for the purpose of the analysis of
visual stimuli, there is no reason this must be the only kind of stimuli used.

The stimulus module defines an abstract base class, StimulusBase, whose abstract methods define the
interface for the module. When constructing an SCO pipeline, a StimulusBase object is required,
and one may be obtained from the StimulusImage class or from a custom class that overloads the
StimulusBase class.
'''

from ..core import calc_chain
from .core  import (import_stimulus_images, calc_normalized_stimulus_images,
                    calc_gabor_filters, calc_filtered_images)

stimulus_chain = (('import_stimulus',          import_stimulus_images),
                  ('calc_normalized_stimulus', calc_normalized_stimulus_images),
                  ('calc_filters',             calc_gabor_filters),
                  ('calc_filtered_images',     calc_filtered_images))

calc_stimulus = calc_chain(stimulus_chain)

