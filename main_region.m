clc; clear; close all;  

rdm_centroid = true;
training_sample = 1000;

subcar_num = 114;
folder_name = 'csi_data/case4';
region = ["inside"; "outside";]; % case4

pos_num = 16;
pos = [];
for i = 1:pos_num
    curr_pos = "p" + num2str(i);
    pos = [pos; curr_pos];
end

color = ['b', 'g', 'r', 'y', 'm', 'c'];
region_num = length(region);

data = cell(region_num, 1);
label = cell(region_num, 1);
sample_num = zeros(region_num, 1);

for i = 1:region_num
    for j = 1:pos_num
        [curr_data, curr_label, curr_sample_num] = load_and_preprocessing_data(folder_name, region(i), pos(j), i);
        data{i,1} = [data{i,1}; curr_data];
        label{i,1} = [label{i,1}; curr_label];
        sample_num(i,1) = sample_num(i,1) + curr_sample_num;
    end
end 

% Initial Cluster Centroid Positions
cent_data = zeros(region_num, subcar_num);
if rdm_centroid == true
    idx = randi([1, min(sample_num)], training_sample, 1);
else
    idx = 1:training_sample;
end
for i = 1:region_num
    cent_data(i,:) = mean(data{i,1}(idx, :), 1);
    data{i,1}(idx, :) = [];
    label{i,1}(idx) = [];
    sample_num(i,1) = sample_num(i,1) - training_sample;
end

% Plot Cluster Centroid
figure;
hold on
for i = 1:region_num
    plot(1:subcar_num, cent_data(i,:), color(i));   
end

% Clustering
all_data = [];
all_label = [];
for i = 1:region_num
    all_data = [all_data; data{i}];
    all_label = [all_label; label{i}];
end

%[pred, C] = kmeans(all_data, region_num, 'Distance', 'correlation', 'Start', cent_data);
[pred, C] = kmeans(all_data, region_num, 'Distance', 'correlation');

class_data = cell(region_num, 1);
for i = 1:region_num
    class_data{i} = all_data(find(pred == i), :);
    figure; plot(1:subcar_num, class_data{i}, color(i));
end

% Review Performance
pred_class = zeros(region_num, 1);
error_num = 0;
start_idx = 1;
end_idx = sample_num(1,1);
for i = 1:region_num
    
    curr_pred_class = mode(pred(start_idx:end_idx, :));
    if ismember(curr_pred_class, pred_class)
        error_num = error_num + length(find(pred(start_idx:end_idx, :) == curr_pred_class));
    else
        error_num = error_num + length(find(pred(start_idx:end_idx, :) ~= curr_pred_class));
    end
    pred_class(i) = curr_pred_class;
    
    start_idx = start_idx + sample_num(i,1);
    if i ~= region_num
        end_idx = sum(sample_num(1:i+1,1));
    end
    
end
            
rate = 1 - (error_num / sum(sample_num));

fprintf('Folder name: %s\n', folder_name);
fprintf('Total sample: %d\n', sum(sample_num));
fprintf('Error sample: %d\n', error_num);
fprintf('Acc: %f\n', rate);

% =========================================================================
function [output_data, output_label, output_number] = load_and_preprocessing_data(folder_name, region, pos, label)

    data = csvread([folder_name '/'] + region + '/' +  pos + ['/amp.csv']);
    data = replace_inf(data);
    data = remove_zero(data);
    data = remove_pulse(data);
       
    mean_data = mean(data, 2);
    output_data = data - mean_data;
    
    [sample_num, ~] = size(data);
    output_label = ones(sample_num, 1) * label;
    
    [output_number, ~] = size(data);
    
end

% =========================================================================
function [data] = replace_inf(data)
    
    [sample_num, ~] = size(data);
    
    inf_idx = find(isinf(data));
    
    if ~isempty(inf_idx)
        for i=1:length(inf_idx)
            data(inf_idx(i)) = (data(inf_idx(i)-sample_num) + data(inf_idx(i)+sample_num)) / 2;
        end
    end
      
end

% =========================================================================
function [data] = remove_zero(data)
    
    [sample_num, ~] = size(data);
    rm_idx = [];
    
    for i = 1:sample_num
        if ~isempty(find(data(i,:)==0))
            rm_idx = [rm_idx; i];
        end
    end
      
    data(rm_idx,:) = [];
    
end

% =========================================================================
function [data] = remove_pulse(data)
    
    [sample_num, ~] = size(data);
    rm_idx = [];
    
    for i = 1:sample_num
        slope = abs(data(i,2:end) - data(i,1:end-1));
        if length(find(slope == 0)) > 20
            rm_idx = [rm_idx; i];
        end              
    end
    
    data(rm_idx,:) = [];
    
end