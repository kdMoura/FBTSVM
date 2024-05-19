function ftsvm_struct = bin_iFBTSVM_prequential(trdata,trlabel,ini_size,batch_size,Parameter,kobj, output_file_path)
addpath('/Randfeat_releasever');


if isfield(Parameter,'kernel_name')==1    
    traindata1= rf_featurize(kobj, double(trdata(1:ini_size, :)));
    %traindata11=  rf_featurize(kobj, double(trdata(ini_size+1,:)));
else
    traindata1=   trdata(1:ini_size, :);
    %traindata11=  trdata(ini_size+1,:);
end
 
%% Initial training
%Split the data into first training and the rest
%traindata1=   trdata(1:ini_size, :); %trdata(1:ceil(length(trdata)*ini_size),:);
trainlabel1=  trlabel(1:ini_size); %trlabel(1:ceil(length(trlabel)*ini_size));


data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency

%Initial training
[ftsvm_struct,data,label] = offline_bintrain(traindata1,trainlabel1,Parameter);
%Clear initial traindata from memory
clearvars traindata1 trainlabel1  


%% Incremental training
%Get the rest of the data
traindata11=  trdata(ini_size+1:end,:); %trdata(ceil(length(trdata)*ini_size)+1:end,:);
trainlabel11= trlabel(ini_size+1:end); %trlabel(ceil(length(trlabel)*ini_size)+1:end);



integerTest=(batch_size==floor(batch_size));
inc_dsize=size(trainlabel11,1);

if integerTest==1 %batch_size is not a percentage
    bats=batch_size;
    warning('-> The last batch size fits the data size.');
else
    bats=batch_size*inc_dsize;
    bats=round(bats);
    warning('-> The last batch size fits the data size.');
    
end

score=[];
p=1;

% Open the file for writing (creating if it doesn't exist, overwriting if it does)
fileID = fopen(output_file_path, 'w');
fclose(fileID);
%fileID = fopen(output_file_path, 'a');
% Define column names
columnNames = {'pred', 'proba_class0', 'proba_class1', 'label'};

% Create an empty table with column names
variableTypes = {'int32', 'double', 'double', 'int32'};

% Create an empty table with column names and variable types
T = table('Size',[0,length(columnNames)], 'VariableNames',columnNames, 'VariableTypes',variableTypes);

while p<inc_dsize
    if p >= 400
        disp("here")
    end
    disp('---------------------------------------------')
    fprintf('p = %d p+bats-1= %d and inc_dsize: %d\n', p, p+bats-1, inc_dsize);
    trainlabel_chunk =  trainlabel11(p:p+bats-1);

    if isfield(Parameter,'kernel_name')==1    
        traindata_chunk= rf_featurize(kobj, double(traindata11(p:p+bats-1,:)));
    else
        traindata_chunk=   traindata11(p:p+bats-1,:);
    end
   
  %teste 
    [acc2,outclass2,time2, fp2, fn2]= bin_classify(ftsvm_struct,traindata_chunk,trainlabel_chunk,kobj, 0);
    predictions = outclass2;
    labels_ = trainlabel11(p:p+bats-1);
    proba_class0 = zeros(size(labels_));
    % Find the indices where fp2 has higher absolute values than fn2
    %fp2_abs_higher = abs(fp2) > abs(fn2);
    % Initialize proba_class1 with zeros
    %proba_class1 = zeros(size(fp2));
    
    % Assign values from fp2 or fn2 based on the comparison
    %proba_class1(fp2_abs_higher) = fp2(fp2_abs_higher);  % Select values from fp2 where abs(fp2) > abs(fn2)
    %proba_class1(~fp2_abs_higher) = fn2(~fp2_abs_higher);  % Select values from fn2 where abs(fp2) <= abs(fn2)

   proba_class1 = fn2-fp2; 

  
  %train
  [ftsvm_struct,data,label]=inc_bintrain(traindata_chunk,trainlabel_chunk,Parameter,ftsvm_struct,data,label);

  [ftsvm_struct,data,label,score]=forget_bin(ftsvm_struct,data,label,score);
  
  
  
  
  
 p = p + bats;
 if (p + bats - 1) > inc_dsize && p < inc_dsize
    bats = rem(inc_dsize, bats);
 end

  
  
  

    % Open the file for writing
    %fileID = fopen('mat_output.csv', 'a');
    
    % Concatenate the data into a matrix
    %data = [pred(:), proba_class0(:), proba_class1(:), label(:)];
    %output_data = [predictions(:), proba_class0(:), proba_class1(:), labels_(:)];
    
    % Write the data to the file
    %printf(fileID, '%.4f,%.4f,%.4f,%.4f\n', output_data.');
   
    % Specify format for each array
    % format_pred = '%.4f';  % Example format: six decimal places
    % format_proba_class0 = '%.4f';
    % format_proba_class1 = '%.4f';
    % format_labels = '%.4f';
    % 
    % % Convert each array to the specified format
    % formatted_predictions =  num2str(predictions, format_pred);
    % formatted_proba_class0 = num2str(proba_class0, format_proba_class0);
    % formatted_proba_class1 = num2str(proba_class1, format_proba_class1);
    % formatted_labels =       num2str(labels_, format_labels);
    % 
    % size_predictions = size(formatted_predictions, 1);
    % size_proba_class0 = size(formatted_proba_class0, 1);
    % size_proba_class1 = size(formatted_proba_class1, 1);
    % size_labels = size(formatted_labels, 1);
    % 
    % disp(size_predictions);
    % disp(size_proba_class0);
    % disp(size_proba_class1);
    % disp(size_labels);
    
    % Concatenate the formatted arrays
    %output_data = [formatted_predictions(:), formatted_proba_class0(:), formatted_proba_class1(:), formatted_labels(:)];
    %output_data = [formatted_predictions(:), formatted_proba_class1(:), formatted_labels(:)];
    %fprintf(fileID, '%s,%s,%s,%s\n', output_data{:});
    predictions(predictions == -1) = 0;
    labels_(labels_ == -1) = 0;
    newTable = table(predictions, proba_class0, proba_class1, labels_, ...
    'VariableNames', T.Properties.VariableNames);
    T = [T; newTable];
    
end

%fclose(fileID);
writetable(T, output_file_path);