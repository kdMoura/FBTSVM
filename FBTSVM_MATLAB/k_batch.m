%% Fold 3 breakhis PFTAS dataset example - Single training
%86
addpath(genpath(pwd))
%% Load dataset
addpath('/Randfeat_releasever');
%file_number = '0';

for file_number = ['0', '1', '2', '3', '4']

    load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/batch/sgpds_signets_tr__n%s_g12_ir1_iu300-881.mat', file_number));
    %load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/labels_n%s_strConfig1_pTrain1.mat',file_number));
    % %Train dataset
    traindata=data;
    trainlabel=target';
    trainlabel(trainlabel == 0) = -1;
    clearvars data target ref_idxs q_idxs ref_users q_users q_type
    
    
    
    Parameter.CC = 8; %C1=C3
    Parameter.CC2=8;
    Parameter.CR = 2; %C2=C4
    Parameter.CR2=2;
    Parameter.eps=0.0001; %epsilon to avoid inverse matrix calculation error
    Parameter.maxeva=500; %maximum of function evaluations to each train/update the model
    %fuzzy
    Parameter.u=0.01; %fuzzy parameter
    Parameter.epsilon=1.00e-05; %fuzzy epsilon
    %forgetting
    Parameter.repetitions=5;
    Parameter.phi=1.00e-05;
    %Kernel approximation parameters
    %if you want to use linear kernel, do not create the parameters in the
    %structure (comment the Paramater. lines below).
    Parameter.kernel_name='rbf';
    Parameter.kernel_param=2.00e-03;
    Parameter.feat_dimensionality=2048;
    Parameter.Napp=4096;
    Parameter.options=[];
    kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality,Parameter.Napp, Parameter.options);
    
    
    
    
   
    %Train model (initial training + incremental training)
    
    if isfield(Parameter,'kernel_name')==1    
        traindata= rf_featurize(kobj, double(traindata));
        %traindata11=  rf_featurize(kobj, double(trdata(ini_size+1,:)));
    end
    

    o_data=[]; %Pre-initialize for efficiency
    o_label=[]; %Pre-initialize for efficiency
    
    
    [model,o_data,o_label] = offline_bintrain(traindata,trainlabel,Parameter);
  


    %% Classify model
     % %Test dataset
    
    load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/batch/sgpds_signets_ts__n%s_r12_q10_sk1__iu0-300.mat', file_number));
    testdata=data;
    testlabel=target';
    testlabel(trainlabel == 0) = -1;
    [acc2,outclass2,time2, fp2, fn2]= bin_classify(model,testdata,testlabel,kobj);

    predictions = outclass2;
    labels_ = testlabel;
    proba_class0 = zeros(size(labels_));
    % Find the indices where fp2 has higher absolute values than fn2
    %fp2_abs_higher = abs(fp2) > abs(fn2);
    % Initialize proba_class1 with zeros
    proba_class1 = fn2-fp2 %zeros(size(fp2));
    
    % Assign values from fp2 or fn2 based on the comparison
    %proba_class1(fp2_abs_higher) = fp2(fp2_abs_higher);  % Select values from fp2 where abs(fp2) > abs(fn2)
    %proba_class1(~fp2_abs_higher) = fn2(~fp2_abs_higher);  % Select values from fn2 where abs(fp2) <= abs(fn2)

    filename = sprintf('pred#model_fbtsvm_sgpds_signets_tr__n%s_g12_ir1_iu300-881#sgpds_signets_ts__n%s_r12_q10_sk1__iu0-300.csv',file_number,file_number);
    output_file_path=sprintf('/home/livia/FilesK/exp_results/fbtsvm/batch/%s',filename);
    

    columnNames = {'pred','proba_class0','proba_class1','label','ref_idxs','q_idxs','ref_users','q_users','q_type'};
    variableTypes = {'int32', 'double', 'double',       'int32', 'int32',  'int32', 'int32',     'int32',  'int32'};
    % Create an empty table with column names and variable types
    T = table('Size',[0,length(columnNames)], 'VariableNames',columnNames, 'VariableTypes',variableTypes);
    predictions(predictions == -1) = 0;
    labels_(labels_ == -1) = 0;
    ref_idxs = ref_idxs';
    q_idxs = q_idxs'; 
    ref_users = ref_users'; 
    q_users = q_users';
    q_type=q_type';
    newTable = table(predictions, proba_class0, proba_class1, labels_, ref_idxs,q_idxs,ref_users,q_users,q_type, ...
    'VariableNames', T.Properties.VariableNames);
    T = [T; newTable];
   writetable(T, output_file_path, 'Delimiter', '\t');
end





