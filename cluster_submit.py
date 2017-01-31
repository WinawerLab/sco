# cluster_submit.py
#
# by William F. Broderick
#
# This little script will help you submit the SCO model to the cluster, saving the outputs for
# later inspection (since they can take a while to run). Note that this was created for use with
# the NYU mercer HPC cluster and may need to be modified for use with others.
#
# Most important is the TEMPLATE_SUBMISSION_SCRIPT global variable at the top of the file. This bit
# of text should be modified to create the basic style of submission script that you need, as it
# will be filled in, turned into a .sh file, and submitted.

TEMPLATE_SUBMISSION_SCRIPT = """
#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=7:00:00
#PBS -l mem=12GB
#PBS -N {job_name}
#PBS -M {username}
#PBS -j oe
#PBS -m ae
#PBS -o {output_dir}/scripts
 
module purge
module load anaconda/2.3.0
 
cd $HOME/sco
python2.7 cluster_submit.py {model_name} {image_dir} {output_dir}
 
# leave a blank line at the end
"""

import glob
import os
import subprocess
from sco.util import metamers

SCO_KWARGS = {'full': {},
              'dumb_V1': {'Kay2013_SOC_constant': 0, 'Kay2013_output_nonlinearity': 1},
              'intermediate_1': {'Kay2013_SOC_constant': .25, 'Kay2013_output_nonlinearity': .8},
              'intermediate_2': {'Kay2013_SOC_constant': .5, 'Kay2013_output_nonlinearity': .6},
              'intermediate_3': {'Kay2013_SOC_constant': .75, 'Kay2013_output_nonlinearity': .4}}


def main(model_names, image_dir, output_dir='~/SCO_metamer_data', job_name="SCO", username="",
         submission_command="qsub {script}", script_path=None):
    """Run the given variants of the models on the passed-in image directory
    
    Because I can't come up with a good way to pass a dictionary around on the command line, the
    user is limited to the supported variants of the model (given in SCO_KWARGS), though this can
    always be modified to add more. I think json could actually be used to accomplish this
    (http://stackoverflow.com/questions/18006161/how-to-pass-dictionary-as-command-line-argument-to-python-script),
    but I'm not sure it's worth the work.
    
    Arguments
    ==============

    model_names: string or list. The model(s) to run. See the keys of SCO_KWARGS in this script for 
                 supported values and what they mean.

    image_dir: string. The path to the directory containing the images to run the model on.

    output_dir: string, optional. Path to the directory to place outputs in

    job_name: string, optional. Name on the cluster of this/these jobs

    username: string, optional. If you want email updates on your job, enter your 
              'username@domain.something' here

    submission_command: string, optional. What command should be used to submit the script that 
                        this function will create. Must contain {script}.

    script_path: string, optional. If you do not want to use the TEMPLATE_SUBMISSION_SCRIPT, set
                 script_path to the path to an example script. If script_path is None (default),
                 will use the TEMPLATE_SUBMISSION_SCRIPT. If you have multiple scripts for different
                 model_names, include {model_name} in your `script_path`
    """
    if isinstance(model_names, basestring):
        model_names = [model_names]
    output_dir = _check_path(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/scripts'):
        os.makedirs(output_dir+'/scripts')
    for name in model_names:
        assert name in SCO_KWARGS.keys(), "Don't know how to run model %s!" % name
        if script_path is None:
            script_file = "%s/scripts/SCO_%s.sh" % (output_dir, name)
            with open(script_file, 'w') as f:
                f.write(TEMPLATE_SUBMISSION_SCRIPT.format(job_name=job_name, username=username,
                                                          model_name=name, image_dir=image_dir,
                                                          output_dir=output_dir))
        else:
            script_file = script_path.format(model_name=model_name)
        print("Submitting model %s" % name)
        subprocess.call(submission_command.format(script=script_file).split(' '))

def _check_path(path_name):
    path_name = os.path.expanduser(path_name)
    if path_name[-1] == '/':
        path_name = path_name[:-1]
    return path_name
    
if __name__ == '__main__':
    import sys
    model_name = sys.argv[1]
    img_dir = _check_path(sys.argv[2])
    output_dir = _check_path(sys.argv[3])
    images = glob.glob(os.path.expanduser('%s/*png' % img_dir))
    assert model_name in SCO_KWARGS.keys(), "Don't know how to run model %s!" % model_name
    metamers.main(images, output_dir, model_name, subject_path='~/Freesurfer_subjects',
                  **SCO_KWARGS[model_name])
