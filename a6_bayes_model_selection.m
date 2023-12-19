% -*- coding: utf-8 -*-
% Authors: Wei Xu <wxu@stu.pku.edu.cn>
% License: Simplified BSD
%
% This script contains Bayesian Model Selection.

load('../data/ctg/Bayes/data-ctg.mat');
res = permute(data, [3, 4, 1, 2]); % res: (501, 501, 21, 5)
times = [0, 799]; % ms
tp = floor(times/2+100);
res = res(tp(1):tp(2), tp(1):tp(2), :, :);
res = imresize(res, [200, 200], 'Method', 'bicubic');
s = size(res, 1);

fname_res = '../data/ctg/Bayes/bayes.mat';
for dim = 1:5
    fname_para = sprintf('../data/ctg/Bayes/para%d.mat', dim);
    if ~exist(fname_para, "file")

        tmp_res = res;

        mean_res = squeeze(mean(tmp_res, 3)); % (200, 200, 5)
        mean_res_dim = squeeze(mean_res(:, :, dim)); % 200, 200

        % Sustained
        fprintf('DIM %d\n', dim);
        cnt = 1;
        para_i = 0;
        para_j = 0;
        para_corr = 0;
        for i = 2:s - 1
            for j = i + 2:s - 1
                M1 = zeros(s, s);
                M1(i:j, i:j) = 1;
                para_i(cnt) = i;
                para_j(cnt) = j;
                a = mean_res_dim(:);
                b = M1(:);
                tmp_corr = corrcoef(a, b);
                para_corr(cnt) = tmp_corr(2);
                cnt = cnt + 1;
            end
        end
        [~, idx] = max(para_corr);
        M1_i = para_i(idx);
        M1_j = para_j(idx);
        fprintf('       M1 start at %.0fms, end at %.0fms \n', ...
            (times(2) - times(1))/(length(res) - 1)*(M1_i - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M1_j - 1)+times(1));

        % Chained
        cnt = 1;
        para_i = 0;
        para_j = 0;
        para_k = 0;
        para_corr = 0;
        for i = 2:s - 1
            for j = i:s - 1
                for k = 1:floor((j - i)/3)
                    M2 = zeros(s, s);
                    M2(i:j, i:j) = 1 - (triu(ones(j-i+1, j-i+1), k) + ...
                        tril(ones(j-i+1, j-i+1), -k));
                    para_i(cnt) = i;
                    para_j(cnt) = j;
                    para_k(cnt) = k;
                    a = mean_res_dim(:);
                    b = M2(:);
                    tmp_corr = corrcoef(a, b);
                    para_corr(cnt) = tmp_corr(2);
                    cnt = cnt + 1;
                end
            end
        end
        [~, idx] = max(para_corr);
        M2_i = para_i(idx);
        M2_j = para_j(idx);
        M2_k = para_k(idx);
        fprintf('       M2 start at %.0fms, end at %.0fms %.0f \n', ...
            (times(2) - times(1))/(length(res) - 1)*(M2_i - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M2_j - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M2_k - 1)+times(1));

        % Reactivated
        cnt = 1;
        para_i = 0;
        para_j = 0;
        para_k = 0;
        para_corr = 0;
        for i = 2:s - 2
            for j = i:s - 2
                for k = 1:floor((j - i)/3)
                    M3 = zeros(s, s);
                    M3(i:j, i:j) = 1 - (triu(ones(j-i+1, j-i+1), k) + ...
                        tril(ones(j-i+1, j-i+1), -k));
                    M3(j-k:j, i:i+k) = 1;
                    M3(i:i+k, j-k:j) = 1;
                    para_i(cnt) = i;
                    para_j(cnt) = j;
                    para_k(cnt) = k;
                    a = mean_res_dim(:);
                    b = M3(:);
                    tmp_corr = corrcoef(a, b);
                    para_corr(cnt) = tmp_corr(2);
                    cnt = cnt + 1;
                end
            end
        end
        [~, idx] = max(para_corr);
        M3_i = para_i(idx);
        M3_j = para_j(idx);
        M3_k = para_k(idx);
        fprintf('       M3 start at %.0fms, end at %.0fms %.0f \n', ...
            (times(2) - times(1))/(length(res) - 1)*(M3_i - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M3_j - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M3_k - 1)+times(1));

        % Oscillating
        cnt = 1;
        para_i = 0;
        para_j = 0;
        para_k = 0;
        para_corr = 0;
        i = M1_i;
        j = M1_j;
        for k = 2:10
            M4 = zeros(s, s);
            M4(i:j, i:j) = chk(k, j-i+1);
            para_i(cnt) = i;
            para_j(cnt) = j;
            para_k(cnt) = k;
            a = mean_res_dim(:);
            b = M4(:);
            tmp_corr = corrcoef(a, b);
            para_corr(cnt) = tmp_corr(2);
            cnt = cnt + 1;
        end

        [val, idx] = max(para_corr);
        M4_i = para_i(idx);
        M4_j = para_j(idx);
        M4_k = para_k(idx);
        fprintf('       M4 start at %.0fms, end at %.0fms %.0f \n', ...
            (times(2) - times(1))/(length(res) - 1)*(M4_i - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M4_j - 1)+times(1), ...
            (times(2) - times(1))/(length(res) - 1)*(M4_k - 1)+times(1));

        save(fname_para, 'M1_*', 'M2_*', 'M3_*', 'M4_*');
    else
        load(fname_para);
    end

    M1 = zeros(s, s);
    i = M1_i;
    j = M1_j;
    M1(i:j, i:j) = 1;

    M2 = zeros(s, s);
    i = M2_i;
    j = M2_j;
    k = M2_k;
    M2(i:j, i:j) = 1 - (triu(ones(j-i+1, j-i+1), k) + ...
        tril(ones(j-i+1, j-i+1), -k));

    M3 = zeros(s, s);
    i = M3_i;
    j = M3_j;
    k = M3_k;
    M3(i:j, i:j) = 1 - (triu(ones(j-i+1, j-i+1), k) + ...
        tril(ones(j-i+1, j-i+1), -k));
    M3(j-k:j, i:i+k) = 1;
    M3(i:i+k, j-k:j) = 1;

    M4 = zeros(s, s);
    i = M4_i;
    j = M4_j;
    k = M4_k;
    M4(i:j, i:j) = chk(k, j-i+1);

    for i = 1:size(data, 1)
        Y = res(:, :, i, dim);
        log_evidence(1, i) = lev_GLM(Y(:), M1(:));
        log_evidence(2, i) = lev_GLM(Y(:), M2(:));
        log_evidence(3, i) = lev_GLM(Y(:), M3(:));
        log_evidence(4, i) = lev_GLM(Y(:), M4(:));
    end

    options.verbose = false;
    [p1, o1] = VBA_groupBMC(log_evidence, options);
    set(o1.options.handles.hf, 'name', 'Group BMS: Y');

    fprintf('Statistics of true model: pxp = %04.3f (Ef = %04.3f)\n', ...
        o1.pxp(1), o1.Ef(1));

    ress{dim} = {p1, o1};
end

save(fname_res, 'ress');

load(fname_res)
for i = 1:5
    for j = 1:4
        [~, pval(i, j), ~, stats] = ttest(ress{i}{1}.r(j, :), 0.25, 'Tail', 'right');
        tval(i, j) = stats.tstat;
        if pval(i, j) >= 0.05
            psig{i, j} = 'n.s.';
        elseif pval(i, j) > 0.01
            psig{i, j} = '*';
        elseif pval(i, j) > 0.001
            psig{i, j} = '**';
        elseif pval(i, j) > 0.0001
            psig{i, j} = '***';
        else
            psig{i, j} = '****';
        end
    end
end

function res = chk(n, sz)
    res = rem(bsxfun(@plus, 1:n, (1:n).'), 2);
    res = imresize(res*2-1, [sz, sz], 'Method', 'bicubic');
end
