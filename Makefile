SHELL := /bin/bash

STIMULI_IDX := $(shell seq 69 73)
VOXEL_IDX := $(shell seq 0 2)

KNK_PATH=/home/billbrod/Documents/Kendrick-socmodel/code/
STIMULI_PATH=stimuli.mat
SUBJ=test-sub
SUBJ_DIR=/Volumes/server/Freesurfer_subjects

# make sure matlab is in your path, which it may not be by default if you're on Mac.

# add way to get stimuli.mat from Kendrick's website, load, and resave
# just the images with matlab

test :
	echo $(STIMULI_IDX)

soc_model_params.csv :
	python2.7 model_comparison_script.py $(STIMULI_PATH) $(SUBJ) $@ $(STIMULI_IDX) -v $(VOXEL_IDX) -s $(SUBJ_DIR)

MATLAB_soc_model_param.csv : soc_model_params.csv 
	echo $(STIMULI_IDX) > sco/model_comparison/stim.txt
	echo $(VOXEL_IDX) > sco/model_comparison/voxel.txt
	matlab -nodesktop -nodisplay -r "cd $(pwd)/sco/model_comparison; compareWithKay2013($(KNK_PATH), $(STIMULI_PATH), 'stim.txt', 'voxel.txt', $<, '$(pwd)/soc_model_params_image_names.mat', $@)";
	rm stim.txt voxel.txt
