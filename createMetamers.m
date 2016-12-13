% createMetamers.m
%
% by William F. Broderick
%
% creates the V1 and V2 metamers used to test the sco model. Note
% that these are spectrally matched noise (V1) and the outputs of
% Eero's texture model (V2).
% 
% based off of example1.m in textureSynth
% 
% this requires both Eero's textureSynth and matlabPyrTools

function createMetamers(textureSynthPath, pyrToolsPath, imgDir, outputDir, imgIdx, numScales, numOrientations, sizeNeighborhood, Niter, numVersions)
% arguments:
% 
% textureSynthPath: string, path to your textureSynth folder, which
% we'll add to the matlab path
%
% pyrToolsPath: string, path to your matlabPyrTools folder, which
% we'll add to the matlab path
% 
% imgDir: template string to the images you want are contained. Should
% be something like the following: /path/to/imgs/*.jpg, so that we
% only find the images of the right type (we will call dir(imgDir)).
% 
% outputDir: string, directory to save these metamers (and the
% original image used) in.
% 
% imgIdx: vector, specifying which images from imgDir to use.
% 
% numScales: integer, the number of scales to use in filters for
% generating metamers. recommended value 4.
% 
% numOrientations: integer, the number of orientations in the
% filters for generating metamers. recommended value 4
% 
% sizeNeighborhood: integer, spatial neighborhood for generating metamers
% (must be an odd number!). recommended value 7
% 
% for details on these three, see the textureSynth library and
% the paper: "A Parametric Texture Model based on Joint Statistics of
% Complex Wavelet Coefficients".  J Portilla and E P Simoncelli. Int'l
% Journal of Computer Vision, vol.40(1), pp. 49-71, Dec 2000.
% 
% Niter: integer, how many iterations the textureSynthesis
% algorithm should go through. recommended 10 to 50.
% 
% numVersions: integer, number of variants of each metamer to make
% (will do this with a different seed).

    addpath(genpath(textureSynthPath));
    addpath(genpath(pyrToolsPath));
    
    if isempty(find(imgDir=='*'))
        error('imgDir must contain a wildcard in order for this to work! see docstring for example')
    end
    imgPaths = dir(imgDir);
    lastSlash = find(imgDir=='/');
    lastSlash = lastSlash(end);
    imgDir = imgDir(1:lastSlash);
    
    if ~strcmp(outputDir(end), '/')
        outputDir = [outputDir '/'];
    end
    
    for ii=imgIdx
        % our input image must be a double
        imgName = imgPaths(ii).name;
        baseImage = double(imread([imgDir, imgName]));
        
        lastPeriod = find(imgName=='.');
        imgName = imgName(1:lastPeriod-1);
        
        imwrite(baseImage/255, [outputDir imgName '.png']);

        params = textureAnalysis(baseImage, numScales, numOrientations, sizeNeighborhood);
        
        % metamer size must be some multiple of
        % 2^(numScales+2). for now, we try to make the size of the
        % metamer image about the same as the size of the input image
        metamerSize = 2^(numScales + 2);
        metamerSize = ceil(size(baseImage, 1) / metamerSize) * metamerSize;
        
        seed = -1;
        for jj=1:numVersions
            seed = seed + 1;
            % with these inputs, textureSynthesis uses white noise
            % as the seed image
            V1Metamer = textureSynthesis(params, [metamerSize metamerSize seed], Niter, [1 0 0 0]');
            V2Metamer = textureSynthesis(params, [metamerSize metamerSize seed], Niter);
        
            % imwrite requires them to lie between 0 and 1, so we scale
            % them down
            imwrite(V1Metamer/255, [outputDir 'V1Met-' imgName '-' num2str(seed) '.png']);
            imwrite(V2Metamer/255, [outputDir 'V2Met-' imgName '-' num2str(seed) '.png']);
        end
    end
    
    close all;
    
    
end
