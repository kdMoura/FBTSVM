%% Fold 3 breakhis PFTAS dataset example - Single training
%86
addpath(genpath(pwd))
%% Load dataset
addpath('/Randfeat_releasever');
%file_number = '0';


ccs = [2, 4, 8, 10];
crs = [2, 4, 8, 10];
us = [0.01];
epsilons = [1e-5];
repetitions = [3, 5];
phis = [0.00001];
kernel_params = [0.002] %v
Napps = [256, 512, 1024, 2048, 4096];


for file_number = ['5', '6', '7']

   

    load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/tune/sgpds_signets_tr__n%s_g12_ir1_iu571-881.mat', file_number));
    %load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/labels_n%s_strConfig1_pTrain1.mat',file_number));
    % %Train dataset
    traindata=data;
    trainlabel=target';
    trainlabel(trainlabel == 0) = -1;
    clearvars data target ref_idxs q_idxs ref_users q_users q_type

    load(sprintf('/home/livia/FilesK/databases/dissimilarities/matlab/tune/sgpds_signets_ts__n%s_r10_q1_sk1__iu0-300.mat', file_number));
    testdata=data;
    testlabel=target';
    testlabel(testlabel == 0) = -1;
    ref_idxs = ref_idxs';
    q_idxs = q_idxs'; 
    ref_users = ref_users'; 
    q_users = q_users';
    q_type=q_type';
    exp=0;
    for C = ccs
    for cr = crs
        for u = us
            for epsilon = epsilons
                for rep = repetitions
                    for phi = phis
                        for kernel_param = kernel_params
                            for Napp = Napps
                                try
                                exp = exp + 1;
                                Parameter = struct();
                                Parameter.CC = C;
                                Parameter.CC2 = C;
                                Parameter.CR = cr;
                                Parameter.CR2 = cr;
                                Parameter.eps = 0.0001;
                                Parameter.maxeva = 500;
                                Parameter.u = u;
                                Parameter.epsilon = epsilon;
                                Parameter.repetitions = rep;
                                Parameter.phi = phi;
                                Parameter.kernel_name = 'rbf';
                                Parameter.kernel_param = kernel_param;
                                Parameter.feat_dimensionality = 2048;
                                Parameter.Napp = Napp;
                                Parameter.options = [];

                                % Print experiment combination
                                disp('---------------');
                                fprintf('Experiment %d: C=%.2f, cr=%.2f, u=%.2f, epsilon=%.2e, rep=%d, phi=%.2e, kernel_param=%.2e, Napp=%d, fn=%d\n', ...
                                    exp, C, cr, u, epsilon, rep, phi, kernel_param, Napp, file_number);
    
    
                                kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality,Parameter.Napp, Parameter.options);
    
    
    
    
   
                                %Train model (initial training + incremental training)
                                
                                if isfield(Parameter,'kernel_name')==1    
                                    traindata_k= rf_featurize(kobj, double(traindata));
                                    %traindata11=  rf_featurize(kobj, double(trdata(ini_size+1,:)));
                                end
                                
                            
                                o_data=[]; %Pre-initialize for efficiency
                                o_label=[]; %Pre-initialize for efficiency
                                
                                
                                [model,o_data,o_label] = offline_bintrain(traindata_k,trainlabel,Parameter);
  


                                %% Classify model
                                 % %Test dataset
                                
                               
                                [acc2,outclass2,time2, fp2, fn2]= bin_classify(model,testdata,testlabel,kobj);
                            
                                predictions = outclass2;
                                labels_ = testlabel;
                                proba_class0 = zeros(size(labels_));
                                % Find the indices where fp2 has higher absolute values than fn2
                                %fp2_abs_higher = abs(fp2) > abs(fn2);
                                % Initialize proba_class1 with zeros
                                %proba_class1 = zeros(size(fp2));
                                
                                % Assign values from fp2 or fn2 based on the comparison
                                %proba_class1(fp2_abs_higher) = fp2(fp2_abs_higher);  % Select values from fp2 where abs(fp2) > abs(fn2)
                                %proba_class1(~fp2_abs_higher) = fn2(~fp2_abs_higher);  % Select values from fn2 where abs(fp2) <= abs(fn2)

                                proba_class1 = fn2-fp2; %d1-d2 I think fn is actually fp %fp2-fn2%fn2-fp2;
                            
                                filename = sprintf('pred%d#model_fbtsvm_sgpds_signets_tr__n%s_g12_ir1_iu571-881#sgpds_signets_ts__n%s_r10_q1_sk1__iu0-300.csv',exp,file_number,file_number);
                                output_file_path=sprintf('/home/livia/FilesK/exp_results/fbtsvm/tune/%s',filename);
                                %/home/livia/FilesK/databases/dissimilarities/tune/sgpds_signets/sgpds_signets_ts__n5_r10_q1_sk1__iu0-300.npz
                            
                                columnNames = {'pred','proba_class0','proba_class1','label','ref_idxs','q_idxs','ref_users','q_users','q_type'};
                                variableTypes = {'int32', 'double', 'double',       'int32', 'int32',  'int32', 'int32',     'int32',  'int32'};
                                
                                % Create an empty table with column names and variable types
                                T = table('Size',[0,length(columnNames)], 'VariableNames',columnNames, 'VariableTypes',variableTypes);
                                predictions(predictions == -1) = 0;
                                labels_(labels_ == -1) = 0;
                                
                                newTable = table(predictions, proba_class0, proba_class1, labels_, ref_idxs,q_idxs,ref_users,q_users,q_type, ...
                                'VariableNames', T.Properties.VariableNames);
                                T = [T; newTable];
                                writetable(T, output_file_path, 'Delimiter', '\t');

                                
                                clearvars acc2 filename fn2 fp2 fp2_abs_higher kobj labels_ newTable T o_data o_label outclass2
                                clearvars output_file_path Parameter predictions proba_class0 proba_class1 time2 model
                                


                                catch err
                                    fprintf('Error occurred in experiment %d: %s\n', exp, err.message);
                                    % Increment experiment count
                                    
                               
                                    continue; % Continue to the next experiment
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    end
    clearvars data target ref_idxs q_idxs ref_users q_users q_type
    clearvars traindata trainlabel testdata testlabel 
end





