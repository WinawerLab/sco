SHELL := /bin/bash

# for all full stimuli, use the following:
STIMULI_IDX_full = $(shell seq 69 224)

# for testing:
# STIMULI_IDX_full := $(shell seq 69 73)

# for all sweep stimuli, use the following (and honestly, this is
# small enough that you don't need to use something else for testing)
STIMULI_IDX_sweep = $(shell seq 0 33)

VOXEL_IDX = $(shell seq 0 3)

# these defaults are probably fine
BRODATZ_METAMER_IDX = $(shell seq 1 15)
FREEMAN2013_METAMER_IDX = $(shell seq 1 15)
IMAGENET_METAMER_IDX = $(shell seq 1 20)
NUM_SCALES=4
NUM_ORIENTATIONS=4
SIZE_NEIGHBORHOOD=7
METAMER_SEEDS = $(shell seq 1 5)

# these you probably need to change
KNK_PATH=~/matlab/git/knkutils/
SUBJ=test-sub
SUBJ_DIR=/Volumes/server/Freesurfer_subjects
TEXTURE_SYNTH_PATH=~/matlab/git/textureSynth/
PYR_TOOLS_PATH=~/matlab/git/matlabPyrTools-master


# make sure matlab is in your path, which it may not be by default if you're on Mac.

# for our stimuli, we use the pictures from Kay2013, which Kendrick
# provides on his website.
Kay2013_comparison :
	mkdir $@

Kay2013_comparison/full_stimuli.mat : Kay2013_comparison
	wget -q http://kendrickkay.net/socmodel/stimuli.mat -O ./Kay2013_comparison/full_stimuli.mat
        # we need to do this to get the stimuli.mat into the format we want
	matlab -nodesktop -nodisplay -r "load('$@','images'); save('$@','images'); quit"

Kay2013_comparison/sweep_stimuli.mat : Kay2013_comparison/full_stimuli.mat
	python2.7 pRF_check.py $< Kay2013_comparison/{}_stimuli.mat

Metamer_images :
	mkdir $@

# primarily using Freeman 2013 stimuli now, so this might not be relevant.
Metamer_images/Original_Brodatz_seed_images : Metamer_images
	wget -q http://multibandtexture.recherche.usherbrooke.ca/images/Original_Brodatz.zip -O ./Original_Brodatz.zip
	unzip ./Original_Brodatz.zip -d Metamer_images
	mv Metamer_images/Original\ Brodatz Metamer_images/Original_Brodatz_seed_images
	rm Original_Brodatz.zip

Metamer_images/Brodatz_metamers : Metamer_images/Original_Brodatz_seed_images/
	mkdir $@
	matlab -nodesktop -nodisplay -r "createMetamers('$(TEXTURE_SYNTH_PATH)', '$(PYR_TOOLS_PATH)', '$(KNK_PATH)','$</*.gif', '$@', [$(BRODATZ_METAMER_IDX)], $(NUM_SCALES), $(NUM_ORIENTATIONS), $(SIZE_NEIGHBORHOOD), 20, 5); quit;"

Metamer_images/Freeman2013_seed_images : Metamer_images
	wget -q https://files.osf.io/v1/resources/9bxut/providers/osfstorage/589e2fa06c613b01f7b350a2/?zip= -O ./Freeman2013.zip
	jar xvf ./Freeman2013.zip
	mkdir $@
	mv tex*im*png $@
	rm Freeman2013.zip

Metamer_images/Freeman2013_metamers : Metamer_images/Freeman2013_seed_images/
	mkdir $@
	matlab -nodesktop -nodisplay -r "createMetamers('$(TEXTURE_SYNTH_PATH)', '$(PYR_TOOLS_PATH)', '$(KNK_PATH)','$</tex-320x320-im*-smp1.png', '$@', [$(FREEMAN2013_METAMER_IDX)], $(NUM_SCALES), $(NUM_ORIENTATIONS), $(SIZE_NEIGHBORHOOD), 20, [$(METAMER_SEEDS)]); quit;"

Metamer_images/ImageNet_seed_images : Metamer_images
	wget -q https://files.osf.io/v1/resources/9bxut/providers/osfstorage/589e2f98594d9001fa121288/?zip= -O ./ImageNet.zip
	jar xvf ./ImageNet.zip
	mkdir $@
	mv im*png $@
	mv im*JPEG $@
	rm ImageNet.zip

Metamer_images/ImageNet_metamers : Metamer_images/ImageNet_seed_images/
	mkdir $@
	matlab -nodesktop -nodisplay -r "createMetamers('$(TEXTURE_SYNTH_PATH)', '$(PYR_TOOLS_PATH)', '$(KNK_PATH)','$</im*', '$@', [$(IMAGENET_METAMER_IDX)], $(NUM_SCALES), $(NUM_ORIENTATIONS), $(SIZE_NEIGHBORHOOD), 20, [$(METAMER_SEEDS)]); quit;"

# this will also create soc_model_params_%_image_names.mat in the same call
Kay2013_comparison/soc_model_params_%.csv : Kay2013_comparison/%_stimuli.mat
	python2.7 model_comparison_script.py $< $(SUBJ) $@ $(STIMULI_IDX_$*) -v $(VOXEL_IDX) -s $(SUBJ_DIR)

Kay2013_comparison/MATLAB_soc_model_params_%.csv : Kay2013_comparison/soc_model_params_%.csv
        # we increment the stimuli index and not the voxel index,
        # because the voxel indices refer to a column in the
        # dataframe/table, while the stimuli indices will actually be
        # used to grab something from an array in matlab
	matlab -nodesktop -nodisplay -r "cd $(shell pwd)/sco/model_comparison; compareWithKay2013('$(KNK_PATH)', '$(shell pwd)/Kay2013_comparison/$*_stimuli.mat', [$(STIMULI_IDX_$*)]+1, [$(VOXEL_IDX)], '$(shell pwd)/$<', '$(shell pwd)/Kay2013_comparison/soc_model_params_$*_image_names.mat', '$(shell pwd)/$@'); quit;"

.PHONY : %_images
# this will create several images, with names based on the default options in sco/model_comparison/core.py
%_images : Kay2013_comparison/MATLAB_soc_model_params_%.csv Kay2013_comparison/%_stimuli.mat Kay2013_comparison/soc_model_params_%.csv
	python2.7 sco/model_comparison/core.py $* $< Kay2013_comparison/soc_model_params_$*_image_names.mat sco/model_comparison/stimuliNames.mat Kay2013_comparison/$*_stimuli.mat $(STIMULI_IDX_$*)
	mv SCO_model_comparison*.svg Kay2013_comparison/

.PHONY : %clean
%clean : 
	-rm Kay2013_comparison/soc_model_params_$*.csv
	-rm Kay2013_comparison/soc_model_params_$*_image_names.mat
	-rm Kay2013_comparison/MATLAB_soc_model_params_$*.csv
	-rm Kay2013_comparison/soc_model_params_$*_results_dict.pkl

# this way these won't be deleted as unnecessary intermediates. These
# take a while to make, so we don't want to do that.
.PRECIOUS : Kay2013_comparison/soc_model_params_%.csv Kay2013_comparison/MATLAB_soc_model_params_%.csv Kay2013_comparison/full_stimuli.mat
