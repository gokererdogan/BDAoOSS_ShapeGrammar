s = image2shape('stimuli/o2.png');
s = resample_shape(s, 200);
sk = mapskeleton(s, 'gradientdescent');