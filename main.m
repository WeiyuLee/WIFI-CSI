clc; clear; close all;  

subcar_num = 114;
folder_name = 'csi_data/case3/near_chip_diff_2';
%pos = ["p1"; "p2"; "p3"; "p4"; "p5"; "p6"]; % case1
%pos = ["no activity"; "people walking";]; % case2
pos = ["DUT_A"; "DUT_B";]; % case3
color = ['b', 'g', 'r', 'y', 'm', 'c'];

pos_num = length(pos);

data = cell(pos_num, 1);
label = cell(pos_num, 1);
sample_num = zeros(pos_num, 1);

for i = 1:pos_num
    [data{i,1}, label{i,1}, sample_num(i,1)] = load_and_preprocessing_data(folder_name, pos(i), i);
end 

% Initial Cluster Centroid Positions
cent_data = zeros(pos_num, subcar_num);
for i = 1:pos_num
    cent_data(i,:) = mean(data{i,1}(1:100, :), 1);
    data{i,1} = data{i,1}(101:end, :);
    label{i,1} = label{i,1}(101:end);
    sample_num(i,1) = sample_num(i,1) - 100;
end

% Plot Cluster Centroid
figure;
hold on
for i = 1:pos_num
    plot(1:subcar_num, cent_data(i,:), color(i));   
end

% Clustering
all_data = [];
all_label = [];
for i = 1:pos_num
    all_data = [all_data; data{i}];
    all_label = [all_label; label{i}];
end

[pred, C] = kmeans(all_data, pos_num, 'Distance', 'correlation', 'Start', cent_data);

class_data = cell(pos_num, 1);
for i = 1:pos_num
    class_data{i} = all_data(find(pred == i), :);
    figure; plot(1:subcar_num, class_data{i}, color(i));
end

% Review Performance
pred_class = zeros(pos_num, 1);
error_num = 0;
start_idx = 1;
end_idx = sample_num(1,1);
for i = 1:pos_num
    
    curr_pred_class = mode(pred(start_idx:end_idx, :));
    if ismember(curr_pred_class, pred_class)
        error_num = error_num + length(find(pred(start_idx:end_idx, :) == curr_pred_class));
    else
        error_num = error_num + length(find(pred(start_idx:end_idx, :) ~= curr_pred_class));
    end
    pred_class(i) = curr_pred_class;
    
    start_idx = start_idx + sample_num(i,1);
    if i ~= pos_num
        end_idx = sum(sample_num(1:i+1,1));
    end
    
end
            
rate = 1 - (error_num / sum(sample_num));

fprintf('Folder name: %s\n', folder_name);
fprintf('Total sample: %d\n', sum(sample_num));
fprintf('Error sample: %d\n', error_num);
fprintf('Acc: %f\n', rate);

% =========================================================================
function [output_data, output_label, output_number] = load_and_preprocessing_data(folder_name, pos, label)

    data = csvread([folder_name '/'] + pos + ['/amp.csv']);
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