addpath(genpath(pwd))

% Load dataset
load('/home/livia/FilesK/databases/dissimilarities/matlab/tune/sgpds_signets_tr__n5_g12_ir1_iu571-881.mat')
traindata=data;
trainlabel=target';
trainlabel(trainlabel == 0) = -1;
clearvars data target

load('/home/livia/FilesK/databases/dissimilarities/matlab/tune/sgpds_signets_ts__n5_r10_q1_sk1__iu0-300.mat')
testdata=data;
testlabel=target';
testlabel(testlabel == 0) = -1;
clearvars data target
ref_idxs  = ref_idxs';
q_idxs    = q_idxs';
ref_users = ref_users';
q_users   = q_users';
q_type    = q_type';

% Define parameter values
ccs = [2];
crs = [4];
us = [0.01];
epsilons = [1e-10];
repetitions = [3];
phis = [0.00001];
kernel_params = [0.002] %v
Napps = [256];

output_folder = '/home/livia/FilesK/exp_results/fbtsvm/';

% Loop over parameter combinations
experiment_count = 1;
for kn = [1, 2, 3, 4, 5]
for C = ccs
    for cr = crs
        for u = us
            for epsilon = epsilons
                for rep = repetitions
                    for phi = phis
                        for kernel_param = kernel_params
                            for Napp = Napps
                                try
                                Parameter = struct();
                                Parameter.CC = C;
                                Parameter.CC2 = C;
                                Parameter.CR = cr;
                                Parameter.CR2 = cr;
                                Parameter.eps = 0.001;
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
                                fprintf('Experiment %d: C=%.2f, cr=%.2f, u=%.2f, epsilon=%.2e, rep=%d, phi=%.2e, kernel_param=%.2e, Napp=%d\n', ...
                                    experiment_count, C, cr, u, epsilon, rep, phi, kernel_param, Napp);


                                kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality, Parameter.Napp, Parameter.options);

                                if isfield(Parameter,'kernel_name') == 1    
                                    traindata_k = rf_featurize(kobj, double(traindata));
                                else
                                    traindata_k = traindata
                                end

                                [ftsvm_struct, ~, ~] = offline_bintrain(traindata_k, trainlabel, Parameter);

                                [acc2, predictions, ~, fp2, fn2] = bin_classify(ftsvm_struct, testdata, testlabel, kobj);

                                proba_class0 = zeros(size(testlabel));
                                fp2_abs_higher = abs(fp2) > abs(fn2);
                                proba_class1 = zeros(size(fp2));
                                proba_class1(fp2_abs_higher) = fp2(fp2_abs_higher);
                                proba_class1(~fp2_abs_higher) = fn2(~fp2_abs_higher);

                                % Writing results
                                columnNames = {'pred', 'proba_class0', 'proba_class1', 'label', 'ref_idxs', 'q_idxs', 'ref_users', 'q_users', 'q_type'};
                                variableTypes = {'int32', 'double', 'double', 'int32', 'int64', 'int64', 'int64', 'int64', 'int64'};
                                T = table('Size',[0,length(columnNames)], 'VariableNames',columnNames, 'VariableTypes',variableTypes);

                                newTable = table(predictions, proba_class0, proba_class1, testlabel, ref_idxs, q_idxs, ref_users, q_users, q_type, ...
                                    'VariableNames', T.Properties.VariableNames);
                                T = newTable;

                                output_file_path = fullfile(output_folder, ['experiment_', num2str(experiment_count), '.csv']);
                                writetable(T, output_file_path);
                                save(fullfile(output_folder, ['experiment_', num2str(experiment_count), '_parameter.mat']), '-struct', 'Parameter');

                                experiment_count = experiment_count + 1;
                                catch err
                                    fprintf('Error occurred in experiment %d: %s\n', experiment_count, err.message);
                                    % Increment experiment count
                                    experiment_count = experiment_count + 1;
                                    clearvars ftsvm_struct Parameter traindata_k kobj
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
end
