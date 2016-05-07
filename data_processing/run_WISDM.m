%% WISDM data generation

'generating data using WISDM dataset'

%% set parameters

segment_length = 200;
test_user_ids = 27:36;
use_pca_features = false;
pca_k = 26;

strcat('segment lenght: ', num2str(segment_length))

%% open file and read raw data

'loading the data'

fid = fopen('datasets/wisdm_raw_data.txt');
A = textscan(fid, '%d%s%d%f%f%f', 'delimiter', ',');
fclose(fid);

%% extract data about activity type, x- , y- , and z-acceleration

user_id = A{1};

arr_activities = A{2};
arr_x = A{4};
arr_y = A{5};
arr_z = A{6};

%% find array indexes when user activity type changes

arr_shift = arr_activities;
arr_shift(end + 1) = arr_activities(1);
arr_shift(1) = [];

idx_shuffled = [1; find(~strcmp(cellstr(arr_activities), cellstr(arr_shift))) + 1];
idx_shuffled(end) = size(arr_activities, 1);

%% form the dataset

'generating new dataset'

sample_size = size(arr_activities, 1);

x = [];
y = [];
z = [];

basic_features = [];
idx_training = [];

for i = 1:size(idx_shuffled, 1) - 1
    
    k = idx_shuffled(i);
        
    while k + segment_length < idx_shuffled(i + 1) 
        
        idx_training = [idx_training; [k:(k + segment_length - 1)]];
        
         x_add = arr_x(k:(k + segment_length - 1))';
         y_add = arr_y(k:(k + segment_length - 1))';
         z_add = arr_z(k:(k + segment_length - 1))';

         basic_features = [basic_features; Extract_basic_features(x_add, y_add, z_add)];
         k = k + segment_length;
        
    end
    
       
end

x = arr_x(idx_training);
y = arr_y(idx_training);
z = arr_z(idx_training);

raw_answers = arr_activities(idx_training(:, 1));
user_data = user_id(idx_training(:, 1));

'dataset was generated'

%% generate pca features

if use_pca_features
   
    'generationg pca features'
    
    basic_features = Extract_pca_features([x, y, z], pca_k);

end

%% transorm answers into numeric labels

answ = [];

for i = 1 : length(raw_answers)
    if strcmp(raw_answers(i), 'Jogging')
        answ = [answ; 1];
    elseif strcmp(raw_answers(i), 'Walking')
        answ = [answ; 2];
    elseif strcmp(raw_answers(i), 'Upstairs')
        answ = [answ; 3];
    elseif strcmp(raw_answers(i), 'Downstairs')
        answ = [answ; 4];
    elseif strcmp(raw_answers(i), 'Sitting')
        answ = [answ; 5];
    elseif strcmp(raw_answers(i), 'Standing')
        answ = [answ; 6];
    end
end

%% transform answers into vectors

answ_vector = [];

for i = 1 : length(answ)
    vect = zeros(6, 1)';
    vect(answ(i)) = 1;
    answ_vector = [answ_vector; vect];
end


%% split data into training and testing sets

'splitting data into training and test sets'

idx_common = ismember(user_data, test_user_ids);
test_indexes = find(idx_common);

% select training data

x_test = x(test_indexes, :);
y_test = y(test_indexes, :);
z_test = z(test_indexes, :);

answ_test = answ(test_indexes, :);
answ_vector_test = answ_vector(test_indexes, :);
basic_features_test = basic_features(test_indexes, :);

% write training data to file

'writing test data to file'

dlmwrite(strcat('wisdm_data/answers_test_', num2str(segment_length), '.csv'), answ_test, 'delimiter', ',')
dlmwrite(strcat('wisdm_data/answers_vectors_test_', num2str(segment_length), '.csv'), answ_vector_test, 'delimiter', ',')
dlmwrite(strcat('wisdm_data/data_x_test_', num2str(segment_length), '.csv'), x_test, 'delimiter', ',', 'precision', 4)
dlmwrite(strcat('wisdm_data/data_y_test_', num2str(segment_length), '.csv'), y_test, 'delimiter', ',', 'precision', 4)
dlmwrite(strcat('wisdm_data/data_z_test_', num2str(segment_length), '.csv'), z_test, 'delimiter', ',', 'precision', 4)
dlmwrite(strcat('wisdm_data/basic_features_test_', num2str(segment_length), '.csv'), basic_features_test, 'delimiter', ',', 'precision', 4);

% delete testing data from training dataset

x(test_indexes, :) = [];
y(test_indexes, :) = [];
z(test_indexes, :) = [];

answ(test_indexes, :) = [];
answ_vector(test_indexes, :) = [];
basic_features(test_indexes, :) = [];

%% chuffle training data

[m, n] = size(y);
idx_shuffled = randperm(m);

x = x(idx_shuffled, :);
y = y(idx_shuffled, :);
z = z(idx_shuffled, :);

answ = answ(idx_shuffled, :);
answ_vector = answ_vector(idx_shuffled, :);
basic_features = basic_features(idx_shuffled, :);


%% write training data to file

'writing training data to file'

dlmwrite(strcat('wisdm_data/answers_', num2str(segment_length), '.csv'), answ, 'delimiter', ',')
dlmwrite(strcat('wisdm_data/answers_vectors_', num2str(segment_length), '.csv'), answ_vector, 'delimiter', ',')
dlmwrite(strcat('wisdm_data/data_x_', num2str(segment_length), '.csv'), x, 'delimiter', ',', 'precision', 4)
dlmwrite(strcat('wisdm_data/data_y_', num2str(segment_length), '.csv'), y, 'delimiter', ',', 'precision', 4)
dlmwrite(strcat('wisdm_data/data_z_', num2str(segment_length), '.csv'), z, 'delimiter', ',', 'precision', 4)
dlmwrite(strcat('wisdm_data/basic_features_', num2str(segment_length), '.csv'), basic_features, 'delimiter', ',', 'precision', 4);

'training and test data was generated'
