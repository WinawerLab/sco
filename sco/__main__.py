####################################################################################################
# __main__.py
# The main function, if sco is invoked directly as command.
# By Noah C. Benson

import os, sys, math, tempfile, tarfile
import pysistence

from neuropythy.util import CommandLineParser
from sco import (build_model)

main_parser = CommandLineParser(
    [('h', 'help',                           'help',                           False),
     ('C', 'create-directories',             'create_directories',             False),
     ('u', 'use-spatial-gabors',             'use_spatial_gabors',             False),
     
     ('w', 'aperture-edge-width',            'aperture_edge_width',            None),
     ('r', 'aperture-radius',                'aperture_radius',                None),
     ('b', 'background',                     'background',                     '0.5'),
     ('n', 'compressive-constants-by-label', 'compressive_constants_by_label', 'sco.impl.benson17.compressive_constants_by_label_Kay2013'),
     ('c', 'contrast-constants-by-label',    'contrast_constants_by_label',    'sco.impl.benson17.contrast_constants_by_label_Kay2013'),
     ('f', 'cpd-sensitivity-function',       'cpd_sensitivity_function',       'sco.impl.benson17.cpd_sensitivity'),
     ('R', 'divisive-exponents-by-label',    'divisive_exponents_by_label',    'sco.impl.benson17.divisive_exponents_by_label_Kay2013'),
     ('d', 'divisive-normalization-schema',  'divisive_normalization_schema',  'Heeger1992'),
     ('g', 'gabor-orientations',             'gabor_orientations',             '8'),
     ('k', 'gains-by-label',                 'gains_by_label',                 'sco.impl.benson17.gains_by_label_Benson2017'),
     ('g', 'gamma',                          'gamma',                          None),
     ('i', 'import-filter',                  'import_filter',                  None),
     ('e', 'max-eccentricity',               'max_eccentricity',               '12'),
     ('M', 'measurements-filename',          'measurements_filename',          None),
     ('m', 'modality',                       'modality',                       'surface'),
     ('X', 'normalized-pixels-per-degree',   'normalized_pixels_per_degree',   '6.4'),
     ('o', 'output-directory',               'output_directory',               '.'),
     ('P', 'output-prefix',                  'output_prefix',                  ''),
     ('S', 'output-suffix',                  'output_suffix',                  ''),
     ('N', 'pRF-n-radii',                    'pRF_n_radii',                    '3'),
     ('Q', 'pRF-sigma-offsets-by-label',     'pRF_sigma_offsets_by_label',     'sco.impl.benson17.pRF_sigma_offsets_by_label_Wandell2015'),
     ('q', 'pRF-sigma-slopes-by-label',      'pRF_sigma_slopes_by_label',      'sco.impl.benson17.pRF_sigma_slopes_by_label_Wandell2015'),
     ('s', 'saturation-constants-by-label',  'saturation_constants_by_label',  'sco.impl.benson17.saturation_constants_by_label_Kay2013')])

sco_help = \
'''
Usage: sco <subject> <pixels-per-degree> <image0000> <image0001>...
Runs the Standard Cortical Observer prediction routine and exports a prediction
file, predictions.mgz.

The following options may be given:
  * -h|--help prints this message.
  ...
'''

def _check_extract(arg, subq=False):
    img_formats = ['.png', '.jpg', '.gif']
    if len(arg) > 7 and arg[-7:] == '.tar.gz' or len(arg) > 4 and arg[-4:] == '.tgz':
        tdir = tempfile.mkdtemp()
        tf = tarfile.TarFile(name=arg, mode='r')
        tf.extractall(path=tdir)
        l = [os.path.join(tdir, f) for f in os.listdir(tdir) if f[0] != '.']
        if subq:
            # Return just the subject path
            if 'surf' in l and 'mri' in l: return tdir
            elif len(l) == 0:
                raise ValueError('Extracted tar-file but found no subject directory!')
            else:
                return os.path.join(tdir, l[0])
        else:
            # Return a list of image paths
            return [fl
                    for fl0 in l
                    for fl in ([fl0] if os.path.isfile(fl0) else
                               [os.path.join(fl0, fff) for fff in os.listdir(fl0)])
                    if len(fl) > 4 and fl[-4:].lower() in img_formats]
    elif subq:
        return arg
    else:
        if os.path.isdir(arg):
            return [os.path.join(arg, ff)
                    for ff in os.listdir(arg)
                    if ff[-4:].lower() in img_formats]
        else:
            return [arg] if len(arg) > 4 and arg[-4:].lower() in img_formats else []

def _parse_label_arg(arg):
    s = ''.join(arg.split()) # eliminate whitespace
    if s[0] == '{' and s.endswith('}'):
        els = s[1:-1].split(',')
        return {int(k):float(v) for el in els for (k,v) in [el.split(':')]}
    elif s[0] == '[' and s.endswith(']'):
        els = s[1:-1].split(',')
        return {(k + 1):float(v) for (k,v) in enumerate(els)}
    else:
        return arg # probably is a variable name...
def _parse_gamma_arg(g):
    if g is None: return None
    s = ''.join(g.split()) # eliminate whitespace
    if s[0] != '[' or s[-1] != ']':
        raise ValueError('gamma argument must be specified as a matrix or vector (in []s)')
    s = s[1:-1]
    rows = s.split(';')
    if len(rows) == 1:
        # we have a vector...
        return np.asarray([float(k) for k in s.split(',')])
    else:
        return np.asarray([[float(c) for c in row.split(',')] for row in rows])
        
def main(argv):
    (args, opts) = main_parser(argv)
    if opts['help']:
        print sco_help
        return 1
    if len(args) < 3: raise ValueError('Syntax: sco <subject> <pixels-per-degree> <images...>')
    # Arg 0 is a subject; arg 1 is a pixels-per-degree value, rest are images
    sub = _check_extract(args[0], True)
    d2p = float(args[1])
    imfiles = [a for arg in args[2:] for a in _check_extract(arg)]

    # simple processing for arguments
    opts['aperture_edge_width'] = None if opts['aperture_edge_width'] is None else \
                                  float(opts['aperture_edge_width'])
    opts['aperture_radius'] = None if opts['aperture_radius'] is None else \
                              float(opts['aperture_radius'])
    opts['background'] = float(opts['background'])
    opts['gabor_orientations'] = int(opts['gabor_orientations'])
    opts['max_eccentricity'] = float(opts['max_eccentricity'])
    opts['normalized_pixels_per_degree'] = float(opts['normalized_pixels_per_degree'])
    opts['pRF_n_radii'] = float(opts['pRF_n_radii'])

    # These take more than a little processing
    opts['gamma']                          = _parse_gamma_arg(opts['gamma'])
    opts['pRF_sigma_offsets_by_label']     = _parse_label_arg(opts['pRF_sigma_offsets_by_label'])
    opts['pRF_sigma_slopes_by_label']      = _parse_label_arg(opts['pRF_sigma_slopes_by_label'])
    opts['divisive_exponents_by_label']    = _parse_label_arg(opts['divisive_exponents_by_label'])
    opts['compressive_constants_by_label'] = _parse_label_arg(opts['compressive_constants_by_label'])
    opts['contrast_constants_by_label']    = _parse_label_arg(opts['contrast_constants_by_label'])
    opts['saturation_constants_by_label']  = _parse_label_arg(opts['saturation_constants_by_label'])
    opts['gains_by_label']                 = _parse_label_arg(opts['gains_by_label'])

    # Might be two measurements files...
    mmfl = opts['measurements_filename']
    if mmfl is not None and opts['modality'].lower() == 'surface':
        mmfl = mmfl.split(':')
        if len(mmfl) != 2:
            raise ValueError('surface modality requires 2 :-separated measurement filenames')
        opts['measurements_filename'] = tuple(mmfl)

    mdl = build_model('benson17')
    r = mdl(opts, subject=sub, stimulus=imfiles, pixels_per_degree=d2p)
    r['exported_files']
    return 0

# Run the main function
sys.exit(main(sys.argv[1:]))
