%% Fold 3 breakhis PFTAS dataset example - Single training
%86
addpath(genpath(pwd))
%% Load dataset

load('/home/livia/FilesK/databases/dissimilarities/matlab/tune/sgpds_signets_tr__n5_g12_ir1_iu571-881.mat')
% %Train dataset
traindata=data;
trainlabel=target';
trainlabel(trainlabel == 0) = -1;
clearvars data target ref_idxs q_idxs ref_users q_users q_type

 % %Test dataset
load('/home/livia/FilesK/databases/dissimilarities/matlab/tune/sgpds_signets_ts__n5_r10_q1_sk1__iu0-300.mat')
testdata=data;
testlabel=target';
testlabel(testlabel == 0) = -1;
clearvars data target %ref_idxs q_idxs ref_users q_users q_type





Parameter = struct();
Parameter.CC = 8; %C1=C3
Parameter.CC2=8;
Parameter.CR = 2; %C2=C4
Parameter.CR2=2;
Parameter.eps=0.001; %epsilon to avoid inverse matrix calculation error
Parameter.maxeva=500; %maximum of function evaluations to each train/update the model
%fuzzy
Parameter.u=0.01; %fuzzy parameter
Parameter.epsilon=1e-10; %fuzzy epsilon
%forgetting
Parameter.repetitions=3;
Parameter.phi=0.00001;
%Kernel approximation parameters
%if you want to use linear kernel, do not create the parameters in the
%structure (comment the Paramater. lines below).
Parameter.kernel_name='rbf';
Parameter.kernel_param=0.2;
Parameter.feat_dimensionality=2048;
Parameter.Napp=300;%4096;

Parameter.options=[];
kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality,Parameter.Napp, Parameter.options);


output_file_path='/home/livia/FilesK/exp_results/fbtsvm/fbtsvm_sig0.csv'
%Train model (initial training + incremental training

if isfield(Parameter,'kernel_name')==1    
    traindata= rf_featurize(kobj, double(traindata));
end
[ftsvm_struct,data,label] = offline_bintrain(traindata,trainlabel,Parameter);

%% Classify model



[acc2,predictions,time2, fp2, fn2]= bin_classify(ftsvm_struct,testdata,testlabel,kobj);

proba_class0 = zeros(size(testlabel));
fp2_abs_higher = abs(fp2) > abs(fn2); % Find the indices where fp2 has higher absolute values than fn2
proba_class1 = zeros(size(fp2));% Initialize proba_class1 with zeros
proba_class1(fp2_abs_higher) = fp2(fp2_abs_higher);  % Select values from fp2 where abs(fp2) > abs(fn2)
proba_class1(~fp2_abs_higher) = fn2(~fp2_abs_higher);  % Select values from fn2 where abs(fp2) <= abs(fn2)

% Writing results
columnNames = {'pred', 'proba_class0', 'proba_class1', 'label', 'ref_idxs', 'q_idxs', 'ref_users', 'q_users', 'q_type'};
variableTypes = {'int32', 'double', 'double', 'int32', 'int64', 'int64', 'int64', 'int64', 'int64'};
T = table('Size',[0,length(columnNames)], 'VariableNames',columnNames, 'VariableTypes',variableTypes);
ref_idxs  = ref_idxs';
q_idxs    = q_idxs';
ref_users = ref_users';
q_users   = q_users';
q_type    = q_type';
newTable = table(predictions, proba_class0, proba_class1, testlabel, ref_idxs, q_idxs, ref_users, q_users, q_type, ...
    'VariableNames', T.Properties.VariableNames);
T = [T; newTable];


output_file_path = '/home/livia/FilesK/exp_results/fbtsvm/fbtsvm_tune_n5_1.csv';
writetable(T, output_file_path);
save('/home/livia/FilesK/exp_results/fbtsvm/fbtsvm_tune_n5_1_parameter.mat', '-struct', 'Parameter');



