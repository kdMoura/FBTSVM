function ftsvm_struct = bin_iFBTSVM_prequential(trdata,trlabel,ini_size,batch_size,Parameter,kobj)
addpath('/Randfeat_releasever');


if isfield(Parameter,'kernel_name')==1    
trdata= rf_featurize(kobj, double(trdata));
    
end
 
%% Initial training
%Split the data into first training and the rest
traindata1=trdata(1:ceil(length(trdata)*ini_size),:);
trainlabel1=trlabel(1:ceil(length(trlabel)*ini_size));


data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency

%Initial training
[ftsvm_struct,data,label] = offline_bintrain(traindata1,trainlabel1,Parameter);
%Clear initial traindata from memory
clearvars traindata1 trainlabel1  


%% Incremental training
%Get the rest of the data
traindata11=trdata(ceil(length(trdata)*ini_size)+1:end,:);
trainlabel11=trlabel(ceil(length(trlabel)*ini_size)+1:end);



integerTest=(batch_size==floor(batch_size));
inc_dsize=size(trainlabel11,1);

if integerTest==1 %batch_size is not a percentage
    bats=batch_size-1;
    warning('-> The last batch size fits the data size.');
else
    bats=batch_size*inc_dsize;
    bats=round(bats);
    warning('-> The last batch size fits the data size.');
    
end

score=[];
p=1;

% Open the file for writing (creating if it doesn't exist, overwriting if it does)
fileID = fopen('/home/livia/FilesK/exp_results/fbtsvm_output.csv', 'w');
fclose(fileID);
fileID = fopen('/home/livia/FilesK/exp_results/fbtsvm_output.csv', 'a');
while p<inc_dsize
   
  %teste 
    [acc2,outclass2,time2, fp2, fn2]= bin_classify(ftsvm_struct,traindata11(p:p+bats,:),trainlabel11(p:p+bats),kobj, 0);
    predictions = outclass2;
    labels_ = trainlabel11(p:p+bats);
    proba_class0 = zeros(size(labels_));
    % Find the indices where fp2 has higher absolute values than fn2
    fp2_abs_higher = abs(fp2) > abs(fn2);
    % Initialize proba_class1 with zeros
    proba_class1 = zeros(size(fp2));
    
    % Assign values from fp2 or fn2 based on the comparison
    proba_class1(fp2_abs_higher) = fp2(fp2_abs_higher);  % Select values from fp2 where abs(fp2) > abs(fn2)
    proba_class1(~fp2_abs_higher) = fn2(~fp2_abs_higher);  % Select values from fn2 where abs(fp2) <= abs(fn2)


  
  %train
  [ftsvm_struct,data,label]=inc_bintrain(traindata11(p:p+bats,:),trainlabel11(p:p+bats),Parameter,ftsvm_struct,data,label);

  [ftsvm_struct,data,label,score]=forget_bin(ftsvm_struct,data,label,score);
  
  
  
  
  
  p=p+bats;
  if (p+bats)>inc_dsize && p<inc_dsize
     resto=rem(inc_dsize,bats)-1;
     
     if resto==-1
         bats=bats-1; 
     else
     bats=resto;
     end
  end

  
  
  

    % Open the file for writing
    %fileID = fopen('mat_output.csv', 'a');
    
    % Concatenate the data into a matrix
    %data = [pred(:), proba_class0(:), proba_class1(:), label(:)];
    output_data = [predictions(:), proba_class0(:), proba_class1(:), labels_(:)];
    
    % Write the data to the file
    fprintf(fileID, '%f,%f,%f,%f\n', output_data.');
    
    % Close the file
    
end

fclose(fileID);