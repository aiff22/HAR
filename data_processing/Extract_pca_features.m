function pca_features = Extract_pca_features (data, k)
 
    mu = mean(data);
    data_norm = bsxfun(@minus, data, mu);

    stat_sigma = std(data_norm);
    data_norm = bsxfun(@rdivide, data_norm, stat_sigma);

    [m, n] = size(data_norm);

    sigma = data_norm'*data_norm/m;
    [U, S, V] = svd(sigma);

    pca_features = data_norm*U(:, 1:k);

end
