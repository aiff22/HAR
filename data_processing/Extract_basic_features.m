function feature_vector = Extract_basic_features (x, y, z)

n = size(x, 1);

% 6 features

feature_vector = [mean(x), mean(y), mean(z), std(x), std(y), std(z)];

% 9 features

feature_vector = [feature_vector, mean(abs(x - mean(x))), mean(abs(y - mean(y))), mean(abs(z - mean(z)))];

% 10 features

feature_vector = [feature_vector, mean(sqrt(x.^2 + y.^2 + z.^2))];

% 40 features

feature_vector = [feature_vector, hist(x, 10)/n, hist(y, 10)/n, hist(z, 10)/n];

end
