%% UCI data generation

'generating data using UCI dataset'

%% set parameters

segment_length = 128;
test_user_ids = [2, 4, 9, 10, 12, 13, 18, 20, 24];

strcat('segment lenght: ', num2str(segment_length))

%% open file and read labels

'loading labels'

fid = fopen('datasets/uci_raw_data/labels.txt');
A = textscan(fid, '%d%d%d%d%d', 'delimiter', ' ');
fclose(fid);

%% extract data about experiments

exp_id = A{1};
usr_id = A{2};
act_id = A{3};
act_be = A{4};
act_en = A{5};

idx_files = [exp_id, usr_id];
idx_files = unique(idx_files, 'rows');

%% form the dataset

'generating new dataset'

% training data

x = []; gyro_x = [];
y = []; gyro_y = [];
z = []; gyro_z = [];
answers_raw = [];

% testing data

test_x = []; test_gyro_x = [];
test_y = []; test_gyro_y = [];
test_z = []; test_gyro_z = [];
test_answers_raw = [];

for i = 1:size(idx_files, 1)
    
    nexp = num2str(idx_files(i, 1));
    if idx_files(i, 1) < 10
       nexp = strcat('0', nexp); 
    end
    
    nusr = num2str(idx_files(i, 2));
    if idx_files(i, 2) < 10
       nusr = strcat('0', nusr); 
    end
    
    fid = fopen(strcat('datasets/uci_raw_data/acc_exp', nexp, '_user', nusr, '.txt'));
    acc_data = textscan(fid, '%f%f%f', 'delimiter', ' ');
    fclose(fid);
    
    fid = fopen(strcat('datasets/uci_raw_data/gyro_exp', nexp, '_user', nusr, '.txt'));
    gyro_data = textscan(fid, '%f%f%f', 'delimiter', ' ');
    fclose(fid);

    data_x = acc_data{1};
    data_y = acc_data{2};
    data_z = acc_data{3};

    gdata_x = gyro_data{1};
    gdata_y = gyro_data{2};
    gdata_z = gyro_data{3};
   
    exp_idx = find(exp_id == idx_files(i, 1));
    exp_data = [act_id(exp_idx), act_be(exp_idx), act_en(exp_idx)];

    if length(find(test_user_ids == idx_files(i, 2))) == 0
        
        for j = 1:size(exp_data, 1)
            
            if exp_data(j, 1) < 7
                k = exp_data(j, 2);
                while k + segment_size <= exp_data(j, 3)
                   x = [x; data_x(k : k + segment_size - 1)'];
                   y = [y; data_y(k : k + segment_size - 1)'];
                   z = [z; data_z(k : k + segment_size - 1)'];

                   gyro_x = [gyro_x; gdata_x(k : k + segment_size - 1)'];
                   gyro_y = [gyro_y; gdata_y(k : k + segment_size - 1)'];
                   gyro_z = [gyro_z; gdata_z(k : k + segment_size - 1)'];

                   answers_raw = [answers_raw; exp_data(j, 1)];
                   k = k + segment_size;
                   
                end
            end
            
	    end
    else
        
        for j = 1:size(exp_data, 1)
            
            if exp_data(j, 1) < 7
                k = exp_data(j, 2);
                while k + segment_size <= exp_data(j, 3)
                   test_x = [test_x; data_x(k : k + segment_size - 1)'];
                   test_y = [test_y; data_y(k : k + segment_size - 1)'];
                   test_z = [test_z; data_z(k : k + segment_size - 1)'];

                   test_gyro_x = [test_gyro_x; gdata_x(k : k + segment_size - 1)'];
                   test_gyro_y = [test_gyro_y; gdata_y(k : k + segment_size - 1)'];
                   test_gyro_z = [test_gyro_z; gdata_z(k : k + segment_size - 1)'];

                   test_answers_raw = [test_answers_raw; exp_data(j, 1)];
                   k = k + segment_size;
                end
            end
            
        end
    end
end

'data was generated'

%% transform answers into vectors

answer_vector = [];

for i = 1 : length(answers_raw)
    vect = zeros(6, 1)';
    vect(answers_raw(i)) = 1;
    answer_vector = [answer_vector; vect];
end

test_answ_vector = [];

for i = 1 : length(test_answers_raw)
    vect = zeros(6, 1)';
    vect(test_answers_raw(i)) = 1;
    test_answ_vector = [test_answ_vector; vect];
end


%% write data to file

'writing data to file'

all_acc_data = [x, y, z];
all_gyro_data = [gyro_x, gyro_y, gyro_z];
all_data = [x, y, z, gyro_x, gyro_y, gyro_z];

% dlmwrite('uci_data/data_acc.csv', all_acc_data, 'delimiter', ',', 'precision', 4)
% dlmwrite('uci_data/data_gyro.csv', all_gyro_data, 'delimiter', ',', 'precision', 4)
dlmwrite('uci_data/all_data.csv', all_data, 'delimiter', ',', 'precision', 4)
dlmwrite('uci_data/answers.csv', answer_vector, 'delimiter', ',')

all_test_acc_data = [test_x, test_y, test_z];
all_test_gyro_data = [test_gyro_x, test_gyro_y, test_gyro_z];
all_test_data = [test_x, test_y, test_z, test_gyro_x, test_gyro_y, test_gyro_z];

% dlmwrite('uci_data/data_acc_test.csv', all_test_acc_data, 'delimiter', ',', 'precision', 4)
% dlmwrite('uci_data/data_gyro_test.csv', all_test_gyro_data, 'delimiter', ',', 'precision', 4)
dlmwrite('uci_data/all_data_test.csv', all_test_data, 'delimiter', ',', 'precision', 4)
dlmwrite('uci_data/answers_test.csv', test_answ_vector, 'delimiter', ',')

'training and test data was generated'
