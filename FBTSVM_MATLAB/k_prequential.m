%% Fold 3 breakhis PFTAS dataset example - Single training
%86
addpath(genpath(pwd))
%% Load dataset

%file_number = '0';

for file_number = ['0', '1', '2', '3', '4']

    load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/stream/data_n%s_strConfig1_pTrain1.mat', file_number));
    load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/stream/labels_n%s_strConfig1_pTrain1.mat',file_number));
    % %Train dataset
    traindata=data;
    trainlabel=labels';
    trainlabel(trainlabel == 0) = -1;
    clearvars breakhis_traindata breakhis_trainlabel
    
    %Experiment 121: C=10.00, cr=2.00, u=0.01, epsilon=1.00e-05, rep=3, phi=1.00e-05, kernel_param=2.00e-03, Napp=256, fn=55  Accuracy
    
%Experiment 100: C=8.00, cr=4.00, u=0.01, epsilon=1.00e-05, rep=5, phi=1.00e-05, kernel_param=2.00e-03, Napp=4096, fn=53 Accuracy : 77.9000 (7011/9000)
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
    
    
    ini_size=76692; %Initial training size
    batch_size=100; %int or percentage
    
    output_file_path=sprintf('/home/livia/FilesK/exp_results/fbtsvm/stream/fbtsvm_sig%s.csv',file_number)
    %Train model (initial training + incremental training)
    model=bin_iFBTSVM_prequential(traindata,trainlabel,ini_size,batch_size,Parameter,kobj, output_file_path)
    
    
    %% Classify model
     % %Test dataset
    %load('breakhis_testlabel');
     %load('breakhis_testdata');
    %[acc2,outclass2,time2, fp2, fn2]= bin_classify(model,breakhis_testdata,breakhis_testlabel,kobj);
end