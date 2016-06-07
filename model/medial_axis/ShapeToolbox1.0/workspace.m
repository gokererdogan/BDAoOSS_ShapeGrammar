%%
img = imread('o1_t1_cs_d1.png');
img = rgb2gray(img);
imwrite(img, 'o1_t1_cs_d1.png');
%

%%
clear variables
clear global
s = image2shape('o1.png');
s = resample_shape(s, 200);
% [sk,~,~,dl] = mapskeleton(s, 'gradientdescent');
[sk,~,~,dl] = mapskeleton(s);

globalVars = who('global');
for iVar = 1:numel(globalVars)
  eval(sprintf('global %s', globalVars{iVar}));  % [EDITED]
end
% 1.6068
save o1.mat
%%
clear variables
clear global
s = image2shape('o1_t1_cs_d1.png');
s = resample_shape(s, 200);
% [sk,~,~,dl] = mapskeleton(s, 'gradientdescent');
[sk,~,~,dl] = mapskeleton(s);

globalVars = who('global');
for iVar = 1:numel(globalVars)
  eval(sprintf('global %s', globalVars{iVar}));  % [EDITED]
end
% 2.1843
save o1_t1_cs_d1.mat

%%
clear variables
clear global
% load o1.mat
% load o1_t1_cs_d1.mat sk current_coribs mean_rib_lengths
load o1_t1_cs_d1.mat
load o1.mat sk current_coribs mean_rib_lengths

description_length(sk)
%%
draw_shape(s1)
draw_skeleton(sk1)
