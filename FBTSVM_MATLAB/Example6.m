%% Border dataset - Multiclass DAG version - Create, update the model and classify at each batch iteration
addpath(genpath(pwd))

%% Output
%Model - Classifier

%cl_output - Classification output to each batch iteration
%it contains the accuracy, time to classify and the distance to +1 and -1
%classifier

%Note - for this version the approximate kernel is loaded to both
%training and testing data in the memory.

%% Read datasets

%Training dataset

load('bo_traindata.mat');
load('bo_trainlabel.mat');
%Testing dataset
load('bo_testlabel.mat');
load('bo_testdata.mat');


%% new set of parameters
Parameter.CC = 8; %C1=C3
Parameter.CC2=8;
Parameter.CR = 2; %C2=C4
Parameter.CR2=2;
Parameter.eps=0.0000001; %epsilon to avoid inverse matrix calculation error
Parameter.maxeva=500; %maximum of function evaluations to each train/update the model
Parameter.u=0.01; %fuzzy parameter
Parameter.epsilon=1e-10; %fuzzy epsilon
Parameter.repetitions=5;
Parameter.phi=0.00001;
Parameter.sliv=true;

%Kernel approximation parameters
%if you want to use linear kernel, do not create the parameters in the
%structure (comment the Paramater. lines below).
 Parameter.kernel_name='rbf';
 Parameter.kernel_param=0.4;
 Parameter.feat_dimensionality=2;
 Parameter.Napp=150;
 Parameter.options=[];
kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality,Parameter.Napp, Parameter.options);
ini_size=0.015; %percentage of the data
batch_size=100; %int or percentage


%% Initial training

%If using kernel approximation
traindata_ini=traindata(1:ceil(length(traindata)*ini_size),:);
trainlabel_ini=trainlabel(1:ceil(length(trainlabel)*ini_size));

model=create_fbtsvm_model(traindata_ini,trainlabel_ini,Parameter,kobj)

%% Update model
%get the rest of the data
traindata_up=traindata(ceil(length(traindata)*ini_size)+1:end,:);
trainlabel_up=trainlabel(ceil(length(trainlabel)*ini_size)+1:end);

[model,cl_output]=update_classify_fbtsvm_model(model,traindata_up,trainlabel_up,batch_size,Parameter,kobj,testdata,testlabel)



