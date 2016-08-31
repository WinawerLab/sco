####################################################################################################
# __main__.py
# The main function, if sco is invoked directly as command.
# By Noah C. Benson

import os, sys, math
import pysistence

from neuropythy.util import CommandLineParser
from sco import (calc_sco, export_predicted_responses)

main_parser = CommandLineParser(
    [('h', 'help',      'help',              False),
     ('p', 'deg2px',    'pixels_per_degree', '24'),
     ('e', 'max-eccen', 'max_eccentricity',  '20'),
     ('o', 'output',    'output_dir',        '.')])

sco_help = \
'''
Usage: sco <subject> <image0000> <image0001>...
Runs the Standard Cortical Observer prediction routine and exports a series of
MGZ files: prediction_0000.mgz, prediction_0001.mgz, etc.
The following options may be given:
  * -h|--help prints this message.
  * -p|--deg2px=<value> sets the pixels per degree in the input images
    (default: 24).
  * -e|--max-eccen=<value> sets the maximum eccentricity modeled in the output
    (default: 20).
  * -o|--output=<directory> sets the output directory (default: .).
'''

def main(argv):
    (args, opts) = main_parser(argv)
    if opts['help']:
        print sco_help
        return 1
    if len(args) < 2: raise ValueError('Syntax: sco <subject> <images...>')
    opts['pixels_per_degree'] = float(opts['pixels_per_degree'])
    opts['max_eccentricity']  = float(opts['max_eccentricity'])
    # Arg 0 is a subject; arg 1 is a directory of images
    imfiles = []
    for f in args[1:]:
        if os.path.isdir(f):
            for ff in os.listdir(f):
                if ff[-4:] in ['.png', '.jpg', '.gif']:
                    imfiles.append(ff)
        else:
            imfiles.append(f)
    r = calc_sco(opts, subject=args[0], stimulus_image_filenames=imfiles)
    export_predicted_responses(r, export_path=opts['output_dir'])
    return 0

# Run the main function
sys.exit(main(sys.argv[1:]))
