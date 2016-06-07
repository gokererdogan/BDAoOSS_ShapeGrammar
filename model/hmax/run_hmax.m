clear variables
addpath('./hmaxMatlab')

saveFolder = './results/';

% read images
image_files = dir('stimuli/*.png');
n = numel(image_files);
image_names = cell(1, n);
images = cell(1, n);

for i = 1:n
    img_filename = image_files(i).name;
    sg = strsplit(img_filename, '.');
    image_names{i} = sg{1};
    images{i} = double(imread(sprintf('stimuli/%s', img_filename)));
end
        
%% Initialize S1 gabor filters and C1 parameters
fprintf('initializing S1 gabor filters\n');
orientations = [90 -45 0 45]; % 4 orientations for gabor filters
RFsizes      = 7:2:39;        % receptive field sizes
div          = 4:-.05:3.2;    % tuning parameters for the filters' "tightness"
[filterSizes,filters,c1OL,~] = initGabor(orientations,RFsizes,div);

fprintf('initializing C1 parameters\n')
c1Scale = 1:2:18; % defining 8 scale bands
c1Space = 8:2:22; % defining spatial pooling range for each scale band

%% Load the universal patch set.
fprintf('Loading the universal patch set\n')
load('universal_patch_set.mat','patches','patchSizes');

nPatchSizes = size(patchSizes,2);


%% For each patch calculate responses
fprintf('calculating unit responses\n');    

[c2,c1,bestBands,bestLocations,s2,s1] = extractC2forCell...
    (filters,filterSizes,c1Space,c1Scale,c1OL,patches,images,nPatchSizes,patchSizes(1:3,:));

%% reformat c1 activations

% we want each cell to contain activations for one band, and these
% activations to be in a matrix of size no_features x no_images
n_bands = numel(c1{1});
c1n = cell(1, n_bands);
for i = 1:n_bands
    c1n{i} = zeros(numel(c1{1}{i}), n);
    for j = 1:n
        z = c1{j}{i};
        c1n{i}(:, j) = z(:);
    end
end

c1 = c1n;
%% Save the output
save([saveFolder 'activations.mat'], 'image_names', 'images', 'c1', 'c2','bestBands','bestLocations');
% save([saveFolder 'activations.mat'],'s1', 's2', 'c1', 'c2','bestBands','bestLocations');
