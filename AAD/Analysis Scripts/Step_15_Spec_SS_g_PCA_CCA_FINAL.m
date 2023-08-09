%% About

% Here, we do the following:

% 1. Read the out structure containing neural data
% 2. Do leave-one-trial-out cross validation and save the models

%% Prepare the workspace

clc;
clear all;

%% Change to current directory and add path to functions

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

% Add path to functions
addpath(genpath("../../APAN 2020 Dataset/AnalysisScripts/Functions/"));
cust_mkdir("./Results");

%% Location of the datasets

neuralDatapath = "../Neural Data - LIJ/";
neuralDatapathStimuli = "../Stimuli/";

% Load stimuli struct
load(neuralDatapathStimuli + "stimuliStructLite.mat", "stimuliStruct");

%% Get a set of subject IDs from the folder

subjectIDs = [174, 204, 182];
subjectPLs = ["LIJ", "CU", "LIJ"];

%% Some parameters for spectrogram reconstruction

silenceDurnBeforeStimulusOnset = 0.5;               % [in seconds]
skipDurnPostStimuliOnset = 4;                       % [in seconds] NOTE: 3 s is ST, additional 1 s to allow for settling post MT introduction
silenceDurnAfterStimulusOffset = 0.5;               % [in seconds] % If this value = 1.5, then 0.5 s = post stim silence, 1.0 s = etches part of the trial.

fNeuralData = 100;                                        % Hz sampling rate of neural data

skipStartSamples = (silenceDurnBeforeStimulusOnset + skipDurnPostStimuliOnset) * fNeuralData;
skipEndSamples = (silenceDurnAfterStimulusOffset) * fNeuralData;

% These two parameters below should be linked:

slidingNeuralWindowDurn = 0.5; % seconds
% lags = -100:0;     

%% Read subjects' out structures and train filters

talkerMode = "MT";

subjectCTR = 3;
    
thisSubjectID = subjectIDs(subjectCTR);
thisSubjectPl = subjectPLs(subjectCTR);

%% Load the structure containing neural data

readFileName = sprintf("out_%s_%d_%s_CMRamp_highgamma.mat", thisSubjectPl, thisSubjectID, talkerMode);
load(strjoin([neuralDatapath filesep thisSubjectPl '-' string(thisSubjectID) filesep 'processed' filesep 'Combined' filesep readFileName], ''), 'out');
thisSubjectOut = out;
clear out;

%% Load this subject's responsive electrodes

load("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/ResponsiveElectrodes.mat", 'thisSubjectResponsiveElectrodes');

%%

% Eliminate trials that are too short (probably not much use if trial lengths are > 40 s)
% Filter out only those trials whose total length exceeds silenceDurnBeforeStimulusOnset +
% skipDurnPostStimuliOnset + silenceDurnAfterStimulusOffset + slidingNeuralWindowDurn [To ensures that from the trials that are left, 
% we have atleast some data to construct stimulus]

dataDurn = [thisSubjectOut.duration];
thisSubjectOut = thisSubjectOut(dataDurn > silenceDurnBeforeStimulusOnset + skipDurnPostStimuliOnset + silenceDurnAfterStimulusOffset + slidingNeuralWindowDurn);

% Prepare data to pass to specReconsLOOClassifier

toPassNeuralResponse = {};

toPassStimuliSpecAttendLags = {};
toPassStimuliSpecUnAttendLags = {};

toPassStimuliTrajAttendLags = {};
toPassStimuliTrajUnAttendLags = {};

toPassGroupInformation = {};        % This holds a character vector for each trial indicating attendDirn and attendGender

% Iterate through trials and add to the cell arrays

for trialCTR = 1:1:length(thisSubjectOut)

    % Get trial details
    trialName = thisSubjectOut(trialCTR).name;
    trialNameSplit = split(trialName, '_');
    trialNo = str2double(trialNameSplit(3)); % PTN

    % Load the attended and un-attended trajectories

    % Note: here, trialNo = PTN
    
    toPassNeuralResponse{end+1} = thisSubjectOut(trialCTR).resp(thisSubjectResponsiveElectrodes, skipStartSamples + 1:end - skipEndSamples);
    toPassStimuliSpecAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(trialNo)).Speech.Original.Individual.Binaural.Conv1.LagsPCA(:, skipStartSamples + 1:end - skipEndSamples);
    toPassStimuliSpecUnAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(trialNo)).Speech.Original.Individual.Binaural.Conv2.LagsPCA(:, skipStartSamples + 1:end - skipEndSamples);
    
    
    toPassStimuliTrajAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(trialNo)).Trajectory.Original.Individual.Conv1.Trajectory_100_Hz_Lags_PCA(:, skipStartSamples + 1:end - skipEndSamples);
    toPassStimuliTrajUnAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(trialNo)).Trajectory.Original.Individual.Conv2.Trajectory_100_Hz_Lags_PCA(:, skipStartSamples + 1:end - skipEndSamples);
    
    toPassGroupInformation{end+1} = "PTN_" + string(trialNo);

end

[toPassNeuralResponse, ~, ~, ~, ~] = normaliseDataAndTraj(toPassNeuralResponse, {}, {});

%% CCA analysis

%%%%%%%%%%%%%%%%%%%%%%%%%%
% For CCA
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Transpose the matrices inside the cell arrays
toPassNeuralResponse = cellfun(@transpose, toPassNeuralResponse, 'UniformOutput', false);


toPassStimuliSpecAttendLags = cellfun(@transpose, toPassStimuliSpecAttendLags, 'UniformOutput', false);
toPassStimuliSpecUnAttendLags = cellfun(@transpose, toPassStimuliSpecUnAttendLags, 'UniformOutput', false);

toPassStimuliTrajAttendLags = cellfun(@transpose, toPassStimuliTrajAttendLags, 'UniformOutput', false);
toPassStimuliTrajUnAttendLags = cellfun(@transpose, toPassStimuliTrajUnAttendLags, 'UniformOutput', false);

% Append the lags to neural data
TMIN_Neural = -500/1000; % s
TMAX_Neural = 0; %s
toPassNeuralResponseLags = generateCellArrayWithLags(toPassNeuralResponse, TMIN_Neural, TMAX_Neural, fNeuralData);

% Append lags to stimuli
TMIN_Stim = -200/1000; % s
TMAX_Stim = 0; % s

% Apply PCA to Neural Data

fprintf("Performing PCA...\n");
toPassNeuralResponseLags = applyPCA(toPassNeuralResponseLags); fprintf("Finished PCA - 1.\n");

%% Save the results of PCA  - Final Spec Traj Combination

savestring1 = sprintf("Results/%s-%d/Step_15_Spec_SS_g_PCA_CCA_FINAL_%s_%d_SpecTrajCombo.mat", thisSubjectPl, thisSubjectID, thisSubjectPl, thisSubjectID);
save(savestring1, 'toPassNeuralResponseLags', 'toPassStimuliSpecAttendLags', 'toPassStimuliSpecUnAttendLags', 'toPassStimuliTrajAttendLags', 'toPassStimuliTrajUnAttendLags', 'toPassGroupInformation', 'TMIN_Neural', 'TMAX_Neural', 'TMIN_Stim', 'TMAX_Stim');

%% Performing CCA

g = struct;
shifts_seconds = 0; % (-500:100:1000)/1000;
shifts_samples = ceil(shifts_seconds * fNeuralData);

% Only Spec

[AA,BB,RR,iBest] = nt_cca_crossvalidate_3(toPassNeuralResponseLags, toPassStimuliSpecAttendLags, shifts_samples);
g.Spec.gA.AA = AA;
g.Spec.gA.BB = BB;
g.Spec.gA.RR = RR;
g.Spec.gA.iBest = iBest;

[AA,BB,RR,iBest] = nt_cca_crossvalidate_3(toPassNeuralResponseLags, toPassStimuliSpecUnAttendLags, shifts_samples);
g.Spec.gU.AA = AA;
g.Spec.gU.BB = BB;
g.Spec.gU.RR = RR;
g.Spec.gU.iBest = iBest;

% Only Traj

[AA,BB,RR,iBest] = nt_cca_crossvalidate_3(toPassNeuralResponseLags, toPassStimuliTrajAttendLags, shifts_samples);
g.Traj.gA.AA = AA;
g.Traj.gA.BB = BB;
g.Traj.gA.RR = RR;
g.Traj.gA.iBest = iBest;

[AA,BB,RR,iBest] = nt_cca_crossvalidate_3(toPassNeuralResponseLags, toPassStimuliTrajUnAttendLags, shifts_samples);
g.Traj.gU.AA = AA;
g.Traj.gU.BB = BB;
g.Traj.gU.RR = RR;
g.Traj.gU.iBest = iBest;

% Both Spec + Traj

[toPassStimuliBothAttendLags] = pairwiseCellArrayConcat(toPassStimuliSpecAttendLags, toPassStimuliTrajAttendLags);
[toPassStimuliBothUnAttendLags] = pairwiseCellArrayConcat(toPassStimuliSpecUnAttendLags, toPassStimuliTrajUnAttendLags);

[AA,BB,RR,iBest] = nt_cca_crossvalidate_3(toPassNeuralResponseLags, toPassStimuliBothAttendLags, shifts_samples);
g.Both.gA.AA = AA;
g.Both.gA.BB = BB;
g.Both.gA.RR = RR;
g.Both.gA.iBest = iBest;

[AA,BB,RR,iBest] = nt_cca_crossvalidate_3(toPassNeuralResponseLags, toPassStimuliBothUnAttendLags, shifts_samples);
g.Both.gU.AA = AA;
g.Both.gU.BB = BB;
g.Both.gU.RR = RR;
g.Both.gU.iBest = iBest;

% Add meta data

g.MetaData.TMIN_Neural = TMIN_Neural;
g.MetaData.TMAX_Neural = TMAX_Neural;

g.MetaData.TMIN_Stim = TMIN_Stim;
g.MetaData.TMAX_Stim = TMAX_Stim;

g.MetaData.shifts_seconds = shifts_seconds;
g.MetaData.shifts_samples = shifts_samples;

g.MetaData.fNeuralData = fNeuralData;

%% Save results to the subjects folder

save("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_SpecTrajCombo_SS_g_PCA_CCA_FINAL.mat", "g");

%% Function for ensuring sameSize

function [toReturn1, toReturn2] = sameSizeEnergy(yStimuliAttend, yStimuliUnAttend, fs)

    % Assumption: both the waveforms are monaural and have the same sampling rate.
    
    toReturn2 = zeros(size(yStimuliAttend));
    
    % Step 1: Normalise the RMS
    temp1 = 1/rms(yStimuliAttend) * yStimuliAttend;
    temp2 = 1/rms(yStimuliUnAttend) * yStimuliUnAttend;
    
    % Step 2: Add 3 s of silence to temp2 [New for this task!]
    temp2 = [zeros(fs * 3, 1); temp2];
    
    % Step 2: Make yStimuliUnAttend same size as yStimuliAttend
    if length(temp2) < length(temp1)                        % Short case -> Pad with 0
        toReturn2(1:length(temp2)) = temp2;
    else                                                    % Long case -> truncate
        toReturn2 = temp2(1:length(temp1));
    end
    
    % Step 3: Return
    toReturn1 = temp1;
    
end


function [toReturn1, toReturn2] = sameEnergy(yStimuliAttend, yStimuliUnAttend, fs)

    % Assumption: both the waveforms are monaural and have the same sampling rate.
    
    toReturn2 = zeros(size(yStimuliAttend));
    
    % Step 1: Normalise the RMS
    temp1 = 1/rms(yStimuliAttend) * yStimuliAttend;
    temp2 = 1/rms(yStimuliUnAttend) * yStimuliUnAttend;
    
    % Step 2: Add 3 s of silence to temp2 [New for this task!]
    % temp2 = [zeros(fs * 3, 1); temp2];
    
    % Step 2: Make yStimuliUnAttend same size as yStimuliAttend
%     if length(temp2) < length(temp1)                        % Short case -> Pad with 0
%         toReturn2(1:length(temp2)) = temp2;
%     else                                                    % Long case -> truncate
%         toReturn2 = temp2(1:length(temp1));
%     end
    
    % Step 3: Return
    toReturn1 = temp1;
    toReturn2 = temp2;
    
end


%% Function to upsample trajectory to 100 Hz

function upsampledTraj = upsampleTraj(inpTraj)
    
    % Assume that inpTraj is having fs = 10 Hz

    upsampledTraj = zeros(1, length(inpTraj) * 10);
    
    for ctr = 1:1:length(inpTraj)
        
        startIDX = (ctr - 1) * 10 + 1;
        endIDX = startIDX + 9;
        
        upsampledTraj(startIDX:endIDX) = inpTraj(ctr);
        
    end

end

%% Function to compute neural average

function newMatrix = neuralAverage(inpMatrix, groupSamples)

    % inpMatrix: (noChannels, oldTimeSteps)

    noChannels = size(inpMatrix, 1);
    oldTimeSteps = size(inpMatrix, 2);
    
    newTimeSteps = ceil(oldTimeSteps/groupSamples);
    
    newMatrix = zeros(noChannels, newTimeSteps);
    
    newMatrixColPTR = 1;
    
    for ptr = 1:groupSamples:oldTimeSteps
        
        endptr = ptr + groupSamples - 1;
        
        if endptr <= oldTimeSteps
            
            newMatrix(:, newMatrixColPTR) = mean(inpMatrix(:, ptr:endptr), 2);
            newMatrixColPTR = newMatrixColPTR + 1;
            
        else
            
            newMatrix(:, newMatrixColPTR) = mean(inpMatrix(:, ptr:end), 2);
            newMatrixColPTR = newMatrixColPTR + 1;
            
        end
        
    end
    
    
end

%% Function to apply the lags to cell array

function newCellArray = generateCellArrayWithLags(inputCellArray, TMIN, TMAX, fs)

    newCellArray = {};
    
    for trialCTR = 1:1:length(inputCellArray)
        
        [newCellArray{end + 1}, ~] = generateMatrixWithLags(inputCellArray{trialCTR}, TMIN, TMAX, fs);
        
    end

end

%% Function to append lagged columns to a channel matrix

function [newMatrix, useMask] = generateMatrixWithLags(inputMatrix, TMIN, TMAX, fs)

    noTimePoints = size(inputMatrix, 1);
    noChannels   = size(inputMatrix, 2);
    
    N_MIN = int32(TMIN*fs);
    N_MAX = int32(TMAX*fs);
    
    lags = N_MIN:1:N_MAX;
    
    newMatrix = zeros(noTimePoints, noChannels * length(lags));
    useMask = ones(noTimePoints, 1);
    
    for channelCTR = 1:1:noChannels
        
        thisChannelColumn = inputMatrix(:, channelCTR);
        [thisChannelLaggedMatrix, thisChannelMask] = generateLagColumns(thisChannelColumn, TMIN, TMAX, fs);
        newMatrix(:, ((channelCTR - 1)*length(lags) + 1):(channelCTR * length(lags))) = thisChannelLaggedMatrix;
        useMask = useMask.*thisChannelMask;
        
    end
    

end

%% Function to generate lagged columns

function [laggedMatrix, useMask] = generateLagColumns(inputColumn, TMIN, TMAX, fs)

    % NOTE: TMIN, TMAX in s
    % fs: Hz

    N_MIN = int32(TMIN*fs);
    N_MAX = int32(TMAX*fs);
    
    lags = N_MIN:1:N_MAX;

    laggedMatrix = zeros(numel(inputColumn), length(lags));
    useMask = ones(numel(inputColumn), 1);
    
    for lagCTR = 1:1:length(lags)
        
        [laggedMatrix(:, lagCTR), thisMask] = getLagColumn(inputColumn, lags(lagCTR));
        useMask = useMask.* thisMask;
        
    end
    
end

%% Function to yield a lagged column

function [laggedColumn, thisMask] = getLagColumn(inputColumn, lagSamples)

    laggedColumn = zeros(numel(inputColumn), 1);
    thisMask = ones(numel(inputColumn), 1);

    if lagSamples < 0 % TMIN = -100 ms => Advance 10 samples, TMIN = -10 ms => Advance 1 sample (if fs = 100 Hz)
        lagSamples = abs(lagSamples);
        laggedColumn(1:(numel(inputColumn) - lagSamples)) = inputColumn((lagSamples + 1):end);
        thisMask(((numel(inputColumn) - lagSamples) + 1):end) = 0;
    elseif lagSamples > 0 % TMAX = 400 ms => Delay 40 samples, TMAX = 10 ms => Delay 1 sample (if fs = 100 Hz)
        laggedColumn(lagSamples + 1:end) = inputColumn(1:end - lagSamples);
        thisMask(1:lagSamples) = 0;
    else
        laggedColumn = inputColumn;
    end
end

%% Centring a cell array

function newCellArray = centreCellArray(inputCellArray)

    % Centres along the first dimension

    newCellArray = {};
    temp = cat(1, inputCellArray{:});
    
    
    for ctr = 1:1:length(inputCellArray)
        newCellArray{end + 1} = inputCellArray{ctr} - mean(temp);
    end

end

%% Function to apply PCA and retain 95% of the variance

function newCellArray = applyPCA(inputCellArray)

    newCellArray = {};
    
    inputComponents = size(inputCellArray{1}, 2);
    
    [coeff, ~, ~, ~, explained, mu] = pca(cat(1, inputCellArray{:})); 
    
    keepComponents = find(cumsum(explained) > 95, 1);
    
    for ctr = 1:1:length(inputCellArray)
        
        thisArray = inputCellArray{ctr};
        thisArray = thisArray - mu;
        
        newCellArray{end + 1} = thisArray * coeff(:, 1:keepComponents);
        
    end
    
    fprintf("PCA: %d -> %d\n", inputComponents, keepComponents);

end

%% Function to apply PCA and retain 95% of the variance for spectrograms

function [newCellArray1, newCellArray2, coeff, mu, keepComponents] = applyPCASpectrograms(inputCellArray1, inputCellArray2)

    newCellArray1 = {};
    newCellArray2 = {};
    
    inputComponents = size(inputCellArray1{1}, 2);
    
    [coeff, ~, ~, ~, explained, mu] = pca(cat(1, inputCellArray1{:}, inputCellArray2{:})); 
    
    keepComponents = find(cumsum(explained) > 95, 1);
    
    for ctr = 1:1:length(inputCellArray1)
        
        thisArray = inputCellArray1{ctr};
        thisArray = thisArray - mu;
        
        newCellArray1{end + 1} = thisArray * coeff(:, 1:keepComponents);
        
    end
    
    for ctr = 1:1:length(inputCellArray2)
        
        thisArray = inputCellArray2{ctr};
        thisArray = thisArray - mu;
        
        newCellArray2{end + 1} = thisArray * coeff(:, 1:keepComponents);
        
    end
    
    fprintf("PCA: %d -> %d\n", inputComponents, keepComponents);

end

%% Function to apply PCA and retain 95% of the variance for spectrograms but using trained filters 

function [newCellArray1, newCellArray2] = applyPCASpectrogramsWithTrainedFilters(inputCellArray1, inputCellArray2, coeff, mu, keepComponents)

    newCellArray1 = {};
    newCellArray2 = {};
    
    inputComponents = size(inputCellArray1{1}, 2);
    
    % [coeff, ~, ~, ~, explained, mu] = pca(cat(1, inputCellArray1{:}, inputCellArray2{:})); 
    
    % keepComponents = find(cumsum(explained) > 95, 1);
    
    for ctr = 1:1:length(inputCellArray1)
        
        thisArray = inputCellArray1{ctr};
        thisArray = thisArray - mu;
        
        newCellArray1{end + 1} = thisArray * coeff(:, 1:keepComponents);
        
    end
    
    for ctr = 1:1:length(inputCellArray2)
        
        thisArray = inputCellArray2{ctr};
        thisArray = thisArray - mu;
        
        newCellArray2{end + 1} = thisArray * coeff(:, 1:keepComponents);
        
    end
    
    fprintf("PCA: %d -> %d\n", inputComponents, keepComponents);

end

%% Function that does pairwise concatenation of matrices from two cell arrays

function newCellArray = pairwiseCellArrayConcat(cellArray1, cellArray2)

    newCellArray = {};
    
    noTrials = length(cellArray1);
    
    for trialCTR = 1:1:noTrials
        
        
        
        if size(cellArray1{trialCTR}, 1) ~= size(cellArray2{trialCTR}, 1)
            toUseTimeSteps = min(size(cellArray1{trialCTR}, 1), size(cellArray2{trialCTR}, 1));
            newCellArray{end+1} = cat(2, cellArray1{trialCTR}(1:toUseTimeSteps, :), cellArray2{trialCTR}(1:toUseTimeSteps, :));
        else
            newCellArray{end+1} = cat(2, cellArray1{trialCTR}, cellArray2{trialCTR});
        end
        
            
        
    end

end