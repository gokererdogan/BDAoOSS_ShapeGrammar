clear variables
clear global
imgs = dir('stimuli/*.png');
n = numel(imgs);

for i = 1:n
    i
    img_filename = imgs(i).name;
    segs = strsplit(img_filename, '.');
    obj_name = segs{1};
    
    s = image2shape(sprintf('stimuli/%s.png', obj_name));
    s = resample_shape(s, 200);
    % sk = mapskeleton(s);
    sk = mapskeleton(s, 'gradientdescent');

    globalVars = who('global');
    for iVar = 1:numel(globalVars)
        eval(sprintf('global %s', globalVars{iVar}));  % [EDITED]
    end

    mat_name = sprintf('%s.mat', obj_name);
    save(mat_name);

    clear variables
    clear global
    
    imgs = dir('stimuli/*.png');
    n = numel(imgs);
end