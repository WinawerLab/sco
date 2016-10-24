STIMULI_IDX=100
KNK_PATH=/home/billbrod/Documents/Kendrick-socmodel/code/
STIMULI_PATH=stimuli.mat
VOXEL_IDX=1

# add way to get stimuli.mat from Kendrick's website, load, and resave
# just the images with matlab

soc_model_params.csv :
	python compare_with_Kay2013.py $(STIMULI_PATH) test-sub $@ $(STIMULI_IDX)

MATLAB_soc_model_param.csv : soc_model_params.csv
	matlab -r 'compareWithKay2013(${KNK_PATH}, ${STIMULI_PATH}, ${STIMULI_IDX}, ${VOXEL_IDX}, $<, $@)';
