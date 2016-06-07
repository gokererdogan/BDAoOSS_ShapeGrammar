clear variables
imgs = dir('stimuli/*.png');
n = numel(imgs);

for i = 1:n
    img_filename = imgs(i).name;
    img = imread(sprintf('stimuli/%s', img_filename));
    img = imresize(img, [250, 250]);
    imwrite(img, sprintf('stimuli/%s', img_filename));
end
