clear variables
clear global

objects = {'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10'};

n_objects = numel(objects);

fid = fopen('MA_ModelSimilarities.csv', 'w');
fprintf(fid, 'Target,Comparison,LogP(SK_c|S_t),LogP(S_t|SK_c),LogP(SK_c),LogP(SK_t|S_c),LogP(S_c|SK_t),LogP(SK_t),LogP(SK_t|S_t),LogP(SK_c|S_c)\n');

for o = 1:n_objects
    object = objects{o};
    object

    variations = dir(sprintf('skeletons/%s_*.mat', object));
    n_variations = numel(variations);
    for v = 1:n_variations
        variation_mat = variations(v).name;
        sg = strsplit(variation_mat, '.');
        variation = sg{1};
        variation

        % load shape and skeleton for object
        load(sprintf('skeletons/%s.mat', object));

        [dl_tot,~,dl_ll, dl_sk] = description_length(sk);
        log_sktarget_starget = -dl_tot;
        
        % load skeleton for variation
        load(sprintf('skeletons/%s.mat', variation), 'sk', 'current_coribs', 'mean_rib_lengths');

        [dl_tot,~,dl_ll, dl_sk] = description_length(sk);
        log_skcomp_starget = -dl_tot;
        log_starget_skcomp = -dl_ll;
        log_skcomp = -dl_sk;

        % load shape and skeleton for variation
        load(sprintf('skeletons/%s.mat', variation));
        [dl_tot,~,dl_ll, dl_sk] = description_length(sk);
        log_skcomp_scomp = -dl_tot;
                
        % load skeleton for object
        load(sprintf('skeletons/%s.mat', object), 'sk', 'current_coribs', 'mean_rib_lengths');

        [dl_tot,~,dl_ll, dl_sk] = description_length(sk);
        log_sktarget_scomp = -dl_tot;
        log_scomp_sktarget = -dl_ll;
        log_sktarget = -dl_sk;

        fprintf(fid, '%s,%s,%f,%f,%f,%f,%f,%f,%f,%f\n', object, variation, log_skcomp_starget, log_starget_skcomp, log_skcomp, log_sktarget_scomp, log_scomp_sktarget, log_sktarget, log_sktarget_starget, log_skcomp_scomp);
    end
end
fclose(fid);
