clear variables
clear global

objects = {'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10'};
variations = {'t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_rp_d1', 't2_rp_d2', 't2_mf_d1', 't2_mf_d2'};

n_objects = numel(objects);
n_variations = numel(variations);

% load c1 and c2 activations
load('results/activations.mat', 'c1', 'c2', 'image_names');

% number of c1 bands
n_bands = numel(c1);
% number of c2 patch sizes
n_patch_sizes = numel(c2);

fid = fopen('HMAX_ModelSimilarities.csv', 'w');
fprintf(fid, 'Target,Comparison');

for i = 1:n_bands
    fprintf(fid, ',c1_%d', i);
end
fprintf(fid, ',c1');

for i = 1:n_patch_sizes
    fprintf(fid, ',c2_%d', i);
end
fprintf(fid, ',c2\n');

for o = 1:n_objects
    object = objects{o};
    object_id = find(cellfun(@(s) strcmp(s, object), image_names));
    object

    for v = 1:n_variations
        comparison = sprintf('%s_%s', object, variations{v});
        comparison_id = find(cellfun(@(s) strcmp(s, comparison), image_names));

        fprintf(fid, '%s,%s', object, comparison);

        tot_d = 0.0;
        for b = 1:n_bands
            d = sum((c1{b}(:, object_id) - c1{b}(:, comparison_id)).^2);
            fprintf(fid, ',%f', d);
            tot_d = tot_d + d;
        end
        fprintf(fid, ',%f', tot_d);


        tot_d = 0.0;
        for p = 1:n_patch_sizes
            d = sum((c2{p}(:, object_id) - c2{p}(:, comparison_id)).^2);
            fprintf(fid, ',%f', d);
            tot_d = tot_d + d;
        end
        fprintf(fid, ',%f\n', tot_d);
    end
end
fclose(fid);
