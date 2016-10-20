% compare_with_Kay2013.m
% 
% this matlab script is the complement of compare_with_Kay2013.py
% and will be run after it to produce results to compare to the
% outputs of the python code.
%
% Assumes you used the default gabor filter options in the python
% code (8 orientations, etc) -- unclear about cycles per image.
% 
% currently, this is the naive way of implementing this, we don't
% take advantage of parallelizing in any way, since this is only
% for small tests.
% 
% By William F. Broderick

function compareWithKay2013(knkutilsPath, stimuliPath, stimuliIdx, voxelIdx, modelPath)
% arguments:
% 
% knkutilsPath: string, path to your knkutils folder, which we'll
% add to the matlab path
% 
% stimuliPath: string, path to stimuli.mat which we'll grab the
% input images from
% 
% stimuliIdx: vector, subset of images to use for these models
% 
% voxelIdx: vector, subset of voxels to generate predictions
% for. There's no reason to do this for every voxel.
% 
% modelPath: string, path to the python dataframe / matlab table
% containing all the model parameters
    modelTable = readtable(modelPath);
    addpath(genpath(knkutilsPath));
    load(stimuliPath, 'images');
    stimuli = images(stimuliIdx);
    clear images;
    
    for imgNum=stimuliIdx
        eval(sprintf('modelTable.MATLAB_predicted_responses_image_%d=nan(height(modelTable),1);',imgNum));
    end
    
    [stimuli, normAlreadyDoneFlag] = preprocessStimuli(stimuli, modelTable);
    
    for vox=voxelIdx
        if ~normAlreadyDoneFlag
            stimuliTmp = divisiveNormalization(stimuli, modelTable, voxelIdx);
        else
            stimuliTmp = stimuli;
        end
        modelTable = generateVoxelPredictions(stimuliTmp, modelTable, vox, stimuliIdx);
    end
    writetable(modelTable,'Kay2013.csv');
end

function [stimuli, normAlreadyDoneFlag] = preprocessStimuli(stimuli, modelTable)
% Preprocess the stimuli for use by all the voxels: resize, make sure
% that all values lay between 0 and 254, then rescale to being between
% 0 and 1 and then mean-center (so they lie between -.5 and .5). Then
% pad stimuli with zeros to reduce edge effects. We then move onto the
% more specialized preprocessing: applying the gabor filters (note
% that this assumes you used the defaults in the python code!) and, if
% we can, divisive normalization (we apply divisive normalization here
% if all voxels have the same divisive normalization parameters)

    % resize the stimuli to 150 x 150 to reduce computational time.
    % use single-format to save memory.
    temp = zeros(150,150,size(stimuli,2),'single');
    for p=1:size(stimuli,2)
        stim = stimuli{:, p};
        if length(size(stim)) == 2
            temp(:,:,p) = imresize(single(stim),[150 150],'cubic');
        else
            % most of the stimuli are made up of multiple frames, but they're all
            % variants on the same thing. For these, we just grab the
            % first frame and ignore the rest because we just want to
            % get a sense of what the model is doing.
            temp(:,:,p) = imresize(single(stim(:,:,1)),[150 150],'cubic');
        end
    end
    stimuli = temp;
    clear temp;

    % ensure that all values are between 0 and 254.
    % rescale values to the range [0,1].
    % subtract off the background luminance (0.5).
    % after these steps, all pixel values will be in the 
    % range [-.5,.5] with the background corresponding to 0.
    stimuli(stimuli < 0) = 0;
    stimuli(stimuli > 254) = 254;
    stimuli = stimuli/254 - 0.5;

    % pad the stimuli with zeros (to reduce edge effects).
    % the new resolution is 180 x 180 (15-pixel padding on each side).
    stimuli = placematrix(zeros(180,180,size(stimuli,3),'single'),stimuli);
   
    % apply Gabor filters to the stimuli.  filters occur at different positions, 
    % orientations, and phases.  there are several parameters that govern the
    % design of the filters:
    %   the number of cycles per image is 37.5*(180/150)
    %   the spatial frequency bandwidth of the filters is 1 octave
    %   the separation of adjacent filters is 1 std dev of the Gaussian envelopes
    %     (this results in a 90 x 90 grid of positions)
    %   filters occur at 8 orientations
    %   filters occur at 2 phases (between 0 and pi)
    %   the Gaussian envelopes are thresholded at .01
    %   filters are scaled to have an equivalent Michelson contrast of 1
    %   the dot-product between each filter and each stimulus is computed
    % after this step, stimulus is images x phases*orientations*positions.
    stimuli = applymultiscalegaborfilters(reshape(stimuli,180*180,[])', ...
                                          37.5*(180/150),-1,1,8,2,.01,2,0);
    
    % compute the square root of the sum of the squares of the outputs of 
    % quadrature-phase filter pairs (this is the standard complex-cell energy model).
    % after this step, stimulus is images x orientations*positions.
    stimuli = sqrt(blob(stimuli.^2,2,2));

    if range(modelTable.Kay2013_normalization_r)==0 && range(modelTable.Kay2013_normalization_s)==0
        % if they all have the same r and s, we can do this here.
        
        normAlreadyDoneFlag = true;
        
        % We call divisive normalization, using parameter values
        % from the first voxel (since they're all the same)
        stimuli = divisiveNormalization(stimuli, modelTable, 1);
    else
        normAlreadyDoneFlag = false;
    end
    
end

function stimuli =  divisiveNormalization(stimuli, modelTable, voxelIdx)
% compute the population term in the divisive-normalization equation.  
% this term is simply the average across the complex-cell outputs 
% at each position (averaging across orientation).
    stimuliPOP = blob(stimuli,2,8)/8;

    % repeat the population term for each of the orientations
    stimuliPOP = upsamplematrix(stimuliPOP,8,2,[],'nearest');
    
    % We only do this if all voxels have the same r and s
    r = modelTable.Kay2013_normalization_r(voxelIdx);
    s = modelTable.Kay2013_normalization_s(voxelIdx);
    
    % apply divisive normalization to the complex-cell outputs.  there are two parameters
    % that influence this operation: an exponent term (r) and a semi-saturation term (s).  
    % the parameter values specified here were determined through a separate fitting 
    % procedure (see paper for details).  for the purposes of this script, we will 
    % simply hard-code the parameter values here and not worry about attempting to fit 
    % the parameters.
    stimuli = stimuli.^r ./ (s.^r + stimuliPOP.^r);
    clear stimuliPOP;

    % sum across orientation.  after this step, stimuli is images x positions.
    stimuli = blob(stimuli,2,8);

end

function modelTable = generateVoxelPredictions(stimuli, modelTable, voxelIdx, stimuliIdx)
% The parameters are [R C S G N C] where
%   R is the row index of the center of the 2D Gaussian (pRF_pixel_centers_image_###_dim0)
%   C is the column index of the center of the 2D Gaussian (pRF_pixel_centers_image_###_dim1)
%   S is the standard deviation of the 2D Gaussian (pRF_pixel_sizes)
%   G is a gain parameter (Kay2013_response_gain)
%   N is the exponent of the power-law nonlinearity (Kay2013_output_nonlinearity)
%   C is a parameter that controls the strength of second-order contrast (Kay2013_SOC_constant)

    % resolution of the pre-processed stimuli; this is taken from
    % Kendrick's socmodel_example, because after the gabor filters
    % are applied, your stimuli are all 90x90
    res = 90;
    % issue a dummy call to makegaussian2d.m to pre-compute xx and yy.
    % these variables are re-used to achieve faster computation.
    [~,xx,yy] = makegaussian2d(res,2,2,2,2);
    
    socfun = @(dd,wts,c) bsxfun(@minus,dd,c*(dd*wts)).^2 * wts;
    gaufun = @(pp) vflatten(makegaussian2d(res,pp(1),pp(2),pp(3),pp(3),xx,yy,0,0)/(2*pi*pp(3)^2));
    modelfun = @(pp,dd) pp(4)*(socfun(dd,gaufun(pp),restrictrange(pp(6),0,1)).^pp(5));
    
    for idx=1:length(stimuliIdx)
        params = [floor(eval(sprintf('modelTable.pRF_pixel_centers_image_%d_dim0(%d)', stimuliIdx(idx), voxelIdx))),
                  floor(eval(sprintf('modelTable.pRF_pixel_centers_image_%d_dim1(%d)', stimuliIdx(idx), voxelIdx))),
                  eval(sprintf('modelTable.pRF_pixel_sizes_image_%d(%d)', stimuliIdx(idx), voxelIdx)),
                  modelTable.Kay2013_response_gain(voxelIdx),
                  modelTable.Kay2013_output_nonlinearity(voxelIdx),
                  modelTable.Kay2013_SOC_constant(voxelIdx)];
        modelfit = modelfun(params, stimuli(idx,:));
        eval(sprintf('modelTable.MATLAB_predicted_responses_image_%d(%d) = %d;', stimuliIdx(idx),voxelIdx,modelfit));
    end

end