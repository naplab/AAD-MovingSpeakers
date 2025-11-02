%% About

% Here, we do the following:

% 1. Read the saved models
% 2. Extract window by window decoding accuracies and correlations

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

useSepStimuliVersion = "Original";

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

talkerMode = "MT"; % wheter ST or MT (shouldn't make much difference in results, but do a check)

for subjectCTR = 3 % 1:1:length(subjectIDs)

    thisSubjectID = subjectIDs(subjectCTR);
    thisSubjectPl = subjectPLs(subjectCTR);

%% Load PCA results - SpecTrajCombo

    neuralDataPathLocal = "./Results/";
    % neuralDataPathSSD = "/Volumes/SSD NTM/Work/Columbia/Research/Projects/Cognitive Hearing Aid - Spatial Decoding/Moving Speaker/Analysis Scripts/Results/";

    loadstring = sprintf("%s%s-%d/Step_15_Spec_SS_g_PCA_CCA_FINAL_%s_%d_SpecTrajCombo.mat", neuralDataPathLocal, thisSubjectPl, thisSubjectID, thisSubjectPl, thisSubjectID);
    load(loadstring, 'toPassNeuralResponseLags', 'toPassStimuliSpecAttendLags', 'toPassStimuliSpecUnAttendLags', 'toPassStimuliTrajAttendLags', 'toPassStimuliTrajUnAttendLags', 'toPassGroupInformation', 'TMIN_Neural', 'TMAX_Neural', 'TMIN_Stim', 'TMAX_Stim');

%% Load SEPARATED VERSION 

%     % Clear clean/ground truth stimuli stuff
% 
%     clear toPassStimuliSpecAttendLags;
%     clear toPassStimuliSpecUnAttendLags;
% 
%     clear toPassStimuliTrajAttendLags;
%     clear toPassStimuliTrajUnAttendLags;
% 
%     % Load the separated version and rename
% 
%     useSepStimuliVersion = "Cong_Final_GT";
% 
%     toPassStimuliSpecAttendLags = {};
%     toPassStimuliSpecUnAttendLags = {};
% 
%     toPassStimuliTrajAttendLags = {};
%     toPassStimuliTrajUnAttendLags = {};
% 
%     for trialCTR = 1:1:numel(toPassGroupInformation)
% 
%         PTN = toPassGroupInformation{trialCTR};
%         PTNSplit = split(PTN, "_");
%         PTN = str2double(PTNSplit(2));
% 
%         toPassStimuliSpecAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Speech.Separated.Individual.Binaural.Conv1.LagsPCA(:, skipStartSamples + 1:end - skipEndSamples)';
%         toPassStimuliSpecUnAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Speech.Separated.Individual.Binaural.Conv2.LagsPCA(:, skipStartSamples + 1:end - skipEndSamples)';
% 
%         toPassStimuliTrajAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Trajectory.Separated.Individual.Conv1.Trajectory_100_Hz_Lags_PCA(:, skipStartSamples + 1:end - skipEndSamples)';
%         toPassStimuliTrajUnAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Trajectory.Separated.Individual.Conv2.Trajectory_100_Hz_Lags_PCA(:, skipStartSamples + 1:end - skipEndSamples)';
% 
%     end

% loadstring = sprintf("%s/%s-%d/PCA_Seperated_Stim_Spec_%s.mat", neuralDatapathResults, thisSubjectPl, thisSubjectID, useSepStimuliVersion);
% load(loadstring, 'toPassStimulilSpecAttendSeparatedLags', 'toPassStimulilSpecUnAttendSeparatedLags');
% 
% toPassStimuliSpecAttendLags = toPassStimulilSpecAttendSeparatedLags;
% toPassStimuliSpecUnAttendLags = toPassStimulilSpecUnAttendSeparatedLags;
% 
% clear toPassStimulilSpecAttendSeparatedLags;
% clear toPassStimulilSpecUnAttendSeparatedLags;

%% Load SWAPPED VERSIONS

%     % Clear clean/ground truth stimuli stuff
% 
%     clear toPassStimuliSpecAttendLags;
%     clear toPassStimuliSpecUnAttendLags;
% 
%     clear toPassStimuliTrajAttendLags;
%     clear toPassStimuliTrajUnAttendLags;
% 
%     % Load the separated version and rename
% 
%     useSepStimuliVersion = "Cong_Final_GT";
% 
%     toPassStimuliSpecAttendLags = {};
%     toPassStimuliSpecUnAttendLags = {};
% 
%     toPassStimuliTrajAttendLags = {};
%     toPassStimuliTrajUnAttendLags = {};
% 
%     for trialCTR = 1:1:numel(toPassGroupInformation)
% 
%         PTN = toPassGroupInformation{trialCTR};
%         PTNSplit = split(PTN, "_");
%         PTN = str2double(PTNSplit(2));
% 
%         toPassStimuliSpecAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Speech.Separated.Individual.Binaural.Conv1.LagsPCA(:, skipStartSamples + 1:end - skipEndSamples)';
%         toPassStimuliSpecUnAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Speech.Separated.Individual.Binaural.Conv2.LagsPCA(:, skipStartSamples + 1:end - skipEndSamples)';
% 
%         toPassStimuliTrajAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Trajectory.Separated.Individual.Conv1.Trajectory_100_Hz_Lags_PCA(:, skipStartSamples + 1:end - skipEndSamples)';
%         toPassStimuliTrajUnAttendLags{end+1} = stimuliStruct.Phase_1.Multi.("PTN_" + string(PTN)).Trajectory.Separated.Individual.Conv2.Trajectory_100_Hz_Lags_PCA(:, skipStartSamples + 1:end - skipEndSamples)';
% 
%     end


%% Load the pre-trained weights

    load(neuralDataPathLocal + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_SpecTrajCombo_SS_g_PCA_CCA_FINAL.mat", "g");

    % Redundant but added to make things back-compatible
    g.Spec.MetaData = g.MetaData;
    g.Traj.MetaData = g.MetaData;
    g.Both.MetaData = g.MetaData;

    %% Window-by-window analysis - Spectrogram Only

    windowSizes = 0.5:0.5:40;   % NOTE: Here, window refers to the size of the spectrogram to be compared/correlated
    overlap = 50; % in percent

    % Make a results table + structure here
    tableResults = cell2table(cell(length(windowSizes), 4), 'VariableNames', {'windowSize', 'testMuMetric1_gA_1CC', 'testMuMetric2_gA_gU_1CC', 'testMuMetric4_gA_3CC_Voting'});
    structResults = struct;

    for windowSizeCTR = 1:1:length(windowSizes)

        thisWindowSize = windowSizes(windowSizeCTR);

        [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelations(toPassNeuralResponseLags, toPassStimuliSpecAttendLags, toPassStimuliSpecUnAttendLags, toPassGroupInformation, g.Spec, fNeuralData, thisWindowSize, overlap, slidingNeuralWindowDurn);

        % 1st CC

        [correctPred, incorrectPred] = metric1(corrAttEst_Att(:, 1), corrAttEst_UnAtt(:, 1), corrUnAttEst_Att(:, 1), corrUnAttEst_UnAtt(:, 1));
        metric1Accu = sum(correctPred>incorrectPred)/numel(correctPred)*100;
        [correctPred, incorrectPred] = metric2(corrAttEst_Att(:, 1), corrAttEst_UnAtt(:, 1), corrUnAttEst_Att(:, 1), corrUnAttEst_UnAtt(:, 1));
        metric2Accu = sum(correctPred>incorrectPred)/numel(correctPred)*100;

        % Voting CCs

        metric4Votes = metric4(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt);
        metric4Accu = sum(metric4Votes >=2 )/numel(metric4Votes) * 100;


        % Which windows correct, which windows incorrect? [Based on gA top 3 voting CCs policy]

        correctWindowIDXs = metric4Votes >= 2;

        correctWindowLabels = windowLabels(correctWindowIDXs);
        incorrectWindowLabels = windowLabels(~correctWindowIDXs);

        % Populating to structure
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).thisWindowSize = thisWindowSize;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrAttEst_Att = corrAttEst_Att;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrAttEst_UnAtt = corrAttEst_UnAtt;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrUnAttEst_Att = corrUnAttEst_Att;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrUnAttEst_UnAtt = corrUnAttEst_UnAtt;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).windowLabels = windowLabels;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric1Accu = metric1Accu;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric2Accu = metric2Accu;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric4Accu = metric4Accu;

        % Populating the table
        tableResults{windowSizeCTR, 'windowSize'} = num2cell(thisWindowSize);
        tableResults{windowSizeCTR, 'testMuMetric1_gA_1CC'} = num2cell(metric1Accu);
        tableResults{windowSizeCTR, 'testMuMetric2_gA_gU_1CC'} = num2cell(metric2Accu);
        tableResults{windowSizeCTR, 'testMuMetric4_gA_3CC_Voting'} = num2cell(metric4Accu);


    %         tableResults{windowSizeCTR, 'meanCorrAtt'} = num2cell(meanCorrAtt);
    %         tableResults{windowSizeCTR, 'meanCorrUnAtt'} = num2cell(meanCorrUnAtt);
    %         
        fprintf("Subject: %d, Window Size: %.2f s, Tot Windows = %d, Accuracies: %.1f - %.1f - %.1f \n", thisSubjectID, thisWindowSize, length(correctPred), metric1Accu, metric2Accu, metric4Accu);

        if thisWindowSize == 4
            PRINTWINDOWREPORT = 1;
        else
            PRINTWINDOWREPORT = 0;
        end

        if PRINTWINDOWREPORT == 1
            fprintf("In-Correct Predictions: \n");

            for winCTR = 1:1:length(incorrectWindowLabels)

                thisWindowLabel = incorrectWindowLabels(winCTR);
                thisWindowLabelSplit = split(thisWindowLabel, "_");
                thisPTN = str2double(thisWindowLabelSplit(2));
                thisWINIDX = str2double(thisWindowLabelSplit(3));

                thisWinStartTime = skipDurnPostStimuliOnset + (thisWINIDX - 1) * (1 - overlap/100) * thisWindowSize;
                thisWinEndTime = thisWinStartTime + thisWindowSize;

                fprintf("PTN: %2d, Window IDX: %5d, Window Start Time: %4.1f, Window End Time: %4.1f \n", thisPTN, thisWINIDX, thisWinStartTime, thisWinEndTime);

            end


            save(sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Spec_CCA_IncorrectWindows_4_s_%s.mat", useSepStimuliVersion), "incorrectWindowLabels");
            % save(sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Spec_CCA_IncorrectWindows_4_s_%s.mat", useSepStimuliVersion), "incorrectWindowLabels");

        else
            fprintf("Not printing window report.\n");
        end

    end

    % Save the table - locally + SSD
    writetable(tableResults, sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_SpecOnly_ClassResults_SS_WinByWin_%s_PCA_CCA.csv", useSepStimuliVersion));
    % writetable(tableResults, sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_SpecOnly_ClassResults_SS_WinByWin_%s_PCA_CCA.csv", useSepStimuliVersion));

    % Also save the struct - locally + SSD
    save(sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Spec_CCA_ResultsStruct_%s.mat", useSepStimuliVersion), "structResults");
    % save(sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Spec_CCA_ResultsStruct_%s.mat", useSepStimuliVersion), "structResults");

    %% Window-by-window analysis - Trajectory Only

    windowSizes = 0.5:0.5:40;   % NOTE: Here, window refers to the size of the spectrogram to be compared/correlated
    overlap = 50; % in percent

    % Make a results table here
    tableResults = cell2table(cell(length(windowSizes), 3), 'VariableNames', {'windowSize', 'testMuMetric1_gA_1CC', 'testMuMetric2_gA_gU_1CC'}); %, 'testMuMetric4_gA_3CC_Voting'});
    structResults = struct;

    for windowSizeCTR = 1:1:length(windowSizes)

        thisWindowSize = windowSizes(windowSizeCTR);

        [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelations(toPassNeuralResponseLags, toPassStimuliTrajAttendLags, toPassStimuliTrajUnAttendLags, toPassGroupInformation, g.Traj, fNeuralData, thisWindowSize, overlap, slidingNeuralWindowDurn);

        % 1st CC

        [correctPred, incorrectPred] = metric1(corrAttEst_Att(:, 1), corrAttEst_UnAtt(:, 1), corrUnAttEst_Att(:, 1), corrUnAttEst_UnAtt(:, 1));
        metric1Accu = sum(correctPred>incorrectPred)/numel(correctPred)*100;

        % Which windows correct, which windows incorrect? (Based on metric 1)

        correctWindowIDXs = correctPred > incorrectPred;

        correctWindowLabels = windowLabels(correctWindowIDXs);
        incorrectWindowLabels = windowLabels(~correctWindowIDXs);

        [correctPred, incorrectPred] = metric2(corrAttEst_Att(:, 1), corrAttEst_UnAtt(:, 1), corrUnAttEst_Att(:, 1), corrUnAttEst_UnAtt(:, 1));
        metric2Accu = sum(correctPred>incorrectPred)/numel(correctPred)*100;

        % Voting CCs

        %metric4Votes = metric4(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt);
        %metric4Accu = sum(metric4Votes >=2 )/numel(metric4Votes) * 100;

        % Populating to structure
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).thisWindowSize = thisWindowSize;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrAttEst_Att = corrAttEst_Att;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrAttEst_UnAtt = corrAttEst_UnAtt;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrUnAttEst_Att = corrUnAttEst_Att;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrUnAttEst_UnAtt = corrUnAttEst_UnAtt;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).windowLabels = windowLabels;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric1Accu = metric1Accu;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric2Accu = metric2Accu;
        % structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric4Accu = metric4Accu;




        tableResults{windowSizeCTR, 'windowSize'} = num2cell(thisWindowSize);
        tableResults{windowSizeCTR, 'testMuMetric1_gA_1CC'} = num2cell(metric1Accu);
        tableResults{windowSizeCTR, 'testMuMetric2_gA_gU_1CC'} = num2cell(metric2Accu);
        % tableResults{windowSizeCTR, 'testMuMetric4_gA_3CC_Voting'} = num2cell(metric4Accu);


    %         tableResults{windowSizeCTR, 'meanCorrAtt'} = num2cell(meanCorrAtt);
    %         tableResults{windowSizeCTR, 'meanCorrUnAtt'} = num2cell(meanCorrUnAtt);
    %         
        fprintf("Subject: %d, Window Size: %.2f s, Tot Windows = %d, Accuracies: %.1f - %.1f \n", thisSubjectID, thisWindowSize, length(correctPred), metric1Accu, metric2Accu);

        if thisWindowSize == 4
            PRINTWINDOWREPORT = 1;
        else
            PRINTWINDOWREPORT = 0;
        end

        if PRINTWINDOWREPORT == 1
            fprintf("In-Correct Predictions: \n");

            for winCTR = 1:1:length(incorrectWindowLabels)

                thisWindowLabel = incorrectWindowLabels(winCTR);
                thisWindowLabelSplit = split(thisWindowLabel, "_");
                thisPTN = str2double(thisWindowLabelSplit(2));
                thisWINIDX = str2double(thisWindowLabelSplit(3));

                thisWinStartTime = skipDurnPostStimuliOnset + (thisWINIDX - 1) * (1 - overlap/100) * thisWindowSize;
                thisWinEndTime = thisWinStartTime + thisWindowSize;

                fprintf("PTN: %2d, Window IDX: %5d, Window Start Time: %4.1f, Window End Time: %4.1f \n", thisPTN, thisWINIDX, thisWinStartTime, thisWinEndTime);

            end

            save(sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Traj_CCA_IncorrectWindows_4_s_%s.mat", useSepStimuliVersion), "incorrectWindowLabels");
            % save(sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Traj_CCA_IncorrectWindows_4_s_%s.mat", useSepStimuliVersion), "incorrectWindowLabels");

        else
            fprintf("Not printing window report.\n");
        end

    end

    writetable(tableResults, sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_TrajOnly_ClassResults_SS_WinByWin_%s_PCA_CCA.csv", useSepStimuliVersion));
    % writetable(tableResults, sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_TrajOnly_ClassResults_SS_WinByWin_%s_PCA_CCA.csv", useSepStimuliVersion));

    save(sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Traj_CCA_ResultsStruct_%s.mat", useSepStimuliVersion), "structResults");
    % save(sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Traj_CCA_ResultsStruct_%s.mat", useSepStimuliVersion), "structResults");


    %% Window-by-window analysis - Both Traj (1 PC) + Spec (30 PC)

    windowSizes = 0.5:0.5:40;   % NOTE: Here, window refers to the size of the spectrogram to be compared/correlated
    overlap = 50; % in percent

    % Make a results table here
    tableResults = cell2table(cell(length(windowSizes), 4), 'VariableNames', {'windowSize', 'testMuMetric1_gA_1CC', 'testMuMetric2_gA_gU_1CC', 'testMuMetric4_gA_3CC_Voting'});
    structResults = struct;

    % Concat features

    [toPassStimuliBothAttendLags] = pairwiseCellArrayConcat(toPassStimuliSpecAttendLags, toPassStimuliTrajAttendLags);
    [toPassStimuliBothUnAttendLags] = pairwiseCellArrayConcat(toPassStimuliSpecUnAttendLags, toPassStimuliTrajUnAttendLags);

    for windowSizeCTR = 1:1:length(windowSizes)

        thisWindowSize = windowSizes(windowSizeCTR);

        [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelations(toPassNeuralResponseLags, toPassStimuliBothAttendLags, toPassStimuliBothUnAttendLags, toPassGroupInformation, g.Both, fNeuralData, thisWindowSize, overlap, slidingNeuralWindowDurn);

        % 1st CC

        [correctPred, incorrectPred] = metric1(corrAttEst_Att(:, 1), corrAttEst_UnAtt(:, 1), corrUnAttEst_Att(:, 1), corrUnAttEst_UnAtt(:, 1));
        metric1Accu = sum(correctPred>incorrectPred)/numel(correctPred)*100;
        [correctPred, incorrectPred] = metric2(corrAttEst_Att(:, 1), corrAttEst_UnAtt(:, 1), corrUnAttEst_Att(:, 1), corrUnAttEst_UnAtt(:, 1));
        metric2Accu = sum(correctPred>incorrectPred)/numel(correctPred)*100;

        % Voting CCs

        metric4Votes = metric4(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt);
        metric4Accu = sum(metric4Votes >=2 )/numel(metric4Votes) * 100;

        % Populating to structure
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).thisWindowSize = thisWindowSize;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrAttEst_Att = corrAttEst_Att;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrAttEst_UnAtt = corrAttEst_UnAtt;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrUnAttEst_Att = corrUnAttEst_Att;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).corrUnAttEst_UnAtt = corrUnAttEst_UnAtt;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).windowLabels = windowLabels;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric1Accu = metric1Accu;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric2Accu = metric2Accu;
        structResults.("WinSizeCTR_" + string(windowSizeCTR)).metric4Accu = metric4Accu;


        % Which windows correct, which windows incorrect? [Based on gA top 3 voting CCs policy]

        correctWindowIDXs = metric4Votes >= 2;

        correctWindowLabels = windowLabels(correctWindowIDXs);
        incorrectWindowLabels = windowLabels(~correctWindowIDXs);

        tableResults{windowSizeCTR, 'windowSize'} = num2cell(thisWindowSize);
        tableResults{windowSizeCTR, 'testMuMetric1_gA_1CC'} = num2cell(metric1Accu);
        tableResults{windowSizeCTR, 'testMuMetric2_gA_gU_1CC'} = num2cell(metric2Accu);
        tableResults{windowSizeCTR, 'testMuMetric4_gA_3CC_Voting'} = num2cell(metric4Accu);


    %         tableResults{windowSizeCTR, 'meanCorrAtt'} = num2cell(meanCorrAtt);
    %         tableResults{windowSizeCTR, 'meanCorrUnAtt'} = num2cell(meanCorrUnAtt);
    %         
        fprintf("Subject: %d, Window Size: %.2f s, Tot Windows = %d, Accuracies: %.1f - %.1f - %.1f \n", thisSubjectID, thisWindowSize, length(correctPred), metric1Accu, metric2Accu, metric4Accu);

        if thisWindowSize == 4
            PRINTWINDOWREPORT = 1;
        else
            PRINTWINDOWREPORT = 0;
        end

        if PRINTWINDOWREPORT == 1
            fprintf("In-Correct Predictions: \n");

            for winCTR = 1:1:length(incorrectWindowLabels)

                thisWindowLabel = incorrectWindowLabels(winCTR);
                thisWindowLabelSplit = split(thisWindowLabel, "_");
                thisPTN = str2double(thisWindowLabelSplit(2));
                thisWINIDX = str2double(thisWindowLabelSplit(3));

                thisWinStartTime = skipDurnPostStimuliOnset + (thisWINIDX - 1) * (1 - overlap/100) * thisWindowSize;
                thisWinEndTime = thisWinStartTime + thisWindowSize;

                fprintf("PTN: %2d, Window IDX: %5d, Window Start Time: %4.1f, Window End Time: %4.1f \n", thisPTN, thisWINIDX, thisWinStartTime, thisWinEndTime);

            end


            save(sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Both_CCA_IncorrectWindows_4_s_%s.mat", useSepStimuliVersion), "incorrectWindowLabels");
            % save(sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Both_CCA_IncorrectWindows_4_s_%s.mat", useSepStimuliVersion), "incorrectWindowLabels");

        else
            fprintf("Not printing window report.\n");
        end

    end

    % writetable(tableResults, "Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_10_Both_ClassResults_SS_WinByWin_CleanMono_PCA_CCA.csv");

    writetable(tableResults, sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_Both_ClassResults_SS_WinByWin_%s_PCA_CCA.csv", useSepStimuliVersion));
    % writetable(tableResults, sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/Step_15_Both_ClassResults_SS_WinByWin_%s_PCA_CCA.csv", useSepStimuliVersion));

    save(sprintf("Results/" + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Both_CCA_ResultsStruct_%s.mat", useSepStimuliVersion), "structResults");
    % save(sprintf(neuralDatapathResults + filesep + thisSubjectPl + "-" + string(thisSubjectID) + "/" + "Step_15_Both_CCA_ResultsStruct_%s.mat", useSepStimuliVersion), "structResults");

end

%% Get correlations time series - spectrogram decoding method - Aug 31 2022 - For checking transitions etc.

windowSizes = 4; % [0.5, 1, 1.5, 2, 4];

correlationsTimeSeries = struct;

for windowSizeCTR = 1:1:length(windowSizes)
    
    thisWindowSize = windowSizes(windowSizeCTR);
    
    [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelationsTimeSeries(toPassNeuralResponseLags, toPassStimuliSpecAttendLags, toPassStimuliSpecUnAttendLags, toPassGroupInformation, g.Spec, fNeuralData, thisWindowSize, slidingNeuralWindowDurn); % Remove overlap
    
    windowLabelsPTN = zeros(1, length(windowLabels));

    for windowCTR = 1:1:length(windowLabels)
        thisWindowLabel = windowLabels(windowCTR);
        thisWindowLabelSplit = split(thisWindowLabel, "_");
        thisWindowPTN = str2double(thisWindowLabelSplit(2));
        windowLabelsPTN(windowCTR) = thisWindowPTN;
    end
    
    uniquePTNs = sort(unique(windowLabelsPTN));
    
    for PTNctr = 1:1:length(uniquePTNs)
        thisPTN = uniquePTNs(PTNctr);
        
        thisPTNStartIDX = find(windowLabelsPTN == thisPTN, 1);
        thisPTNEndIDX = find(windowLabelsPTN == thisPTN, 1, 'last');
        
        correlationsTimeSeries.Spec.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrAttEst_Att = corrAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :);
        correlationsTimeSeries.Spec.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrAttEst_UnAtt = corrAttEst_UnAtt(thisPTNStartIDX:thisPTNEndIDX, :);
        correlationsTimeSeries.Spec.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrUnAttEst_Att = corrUnAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :);
        correlationsTimeSeries.Spec.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrUnAttEst_UnAtt = corrUnAttEst_UnAtt(thisPTNStartIDX:thisPTNEndIDX, :);
        
        windowOffset = ((thisWindowSize * fNeuralData) - 1) * 1/fNeuralData * 1/2;
        
        correlationsTimeSeries.Spec.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).trialTime = generateTimePoints(corrAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :), fNeuralData, skipDurnPostStimuliOnset + windowOffset);
        
    end
    
    % fprintf("Test");
    
    
    
    
    % Populate the strcut
    % correlationsTimeSeries.Spec.
    
end

correlationsTimeSeries.Spec.windowSizes = windowSizes;

% Save to local machine
save(sprintf("%s%s-%d/Step_15_correlationsTimeSeries_%s.mat", neuralDataPathLocal, thisSubjectPl, thisSubjectID, useSepStimuliVersion), 'correlationsTimeSeries');

% Save to SSD
% save(sprintf("%s%s-%d/Step_15_correlationsTimeSeries_%s.mat", neuralDataPathSSD, thisSubjectPl, thisSubjectID, useSepStimuliVersion), 'correlationsTimeSeries');

%% Get correlations time series - both (spec + traj) decoding method - October 4 2022 - For sending out for evaluation, System ON case

% Concat features

[toPassStimuliBothAttendLags] = pairwiseCellArrayConcat(toPassStimuliSpecAttendLags, toPassStimuliTrajAttendLags);
[toPassStimuliBothUnAttendLags] = pairwiseCellArrayConcat(toPassStimuliSpecUnAttendLags, toPassStimuliTrajUnAttendLags);

windowSizes = 4; % [0.5, 1, 1.5, 2, 4];

correlationsTimeSeries = struct;

for windowSizeCTR = 1:1:length(windowSizes)
    
    thisWindowSize = windowSizes(windowSizeCTR);
    
    [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelationsTimeSeries(toPassNeuralResponseLags, toPassStimuliBothAttendLags, toPassStimuliBothUnAttendLags, toPassGroupInformation, g.Both, fNeuralData, thisWindowSize, slidingNeuralWindowDurn); % Remove overlap
    
    windowLabelsPTN = zeros(1, length(windowLabels));

    for windowCTR = 1:1:length(windowLabels)
        thisWindowLabel = windowLabels(windowCTR);
        thisWindowLabelSplit = split(thisWindowLabel, "_");
        thisWindowPTN = str2double(thisWindowLabelSplit(2));
        windowLabelsPTN(windowCTR) = thisWindowPTN;
    end
    
    uniquePTNs = sort(unique(windowLabelsPTN));
    
    for PTNctr = 1:1:length(uniquePTNs)
        thisPTN = uniquePTNs(PTNctr);
        
        thisPTNStartIDX = find(windowLabelsPTN == thisPTN, 1);
        thisPTNEndIDX = find(windowLabelsPTN == thisPTN, 1, 'last');
        
        correlationsTimeSeries.Both.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrAttEst_Att = corrAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :);
        correlationsTimeSeries.Both.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrAttEst_UnAtt = corrAttEst_UnAtt(thisPTNStartIDX:thisPTNEndIDX, :);
        correlationsTimeSeries.Both.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrUnAttEst_Att = corrUnAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :);
        correlationsTimeSeries.Both.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).corrUnAttEst_UnAtt = corrUnAttEst_UnAtt(thisPTNStartIDX:thisPTNEndIDX, :);
        
        windowOffset1 = ((thisWindowSize * fNeuralData) - 1) * 1/fNeuralData * 1/2;
        windowOffset2 = (thisWindowSize + abs(TMIN_Neural));
        
        correlationsTimeSeries.Both.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).trialTime = generateTimePoints(corrAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :), fNeuralData, skipDurnPostStimuliOnset + windowOffset1);
        correlationsTimeSeries.Both.("PTN_" + string(thisPTN)).("WinCTR_" + string(windowSizeCTR)).decodingTime = generateTimePoints(corrAttEst_Att(thisPTNStartIDX:thisPTNEndIDX, :), fNeuralData, skipDurnPostStimuliOnset + windowOffset2);
        
    end
    
    % fprintf("Test");
    
    % Populate the strcut
    % correlationsTimeSeries.Spec.
    
end

correlationsTimeSeries.Both.windowSizes = windowSizes;

% Save to local machine
save(sprintf("%s%s-%d/Step_15_correlationsTimeSeries_Both_%s.mat", neuralDataPathLocal, thisSubjectPl, thisSubjectID, useSepStimuliVersion), 'correlationsTimeSeries');

% Save to SSD
% save(sprintf("%s%s-%d/Step_15_correlationsTimeSeries_Both_%s.mat", neuralDataPathSSD, thisSubjectPl, thisSubjectID, useSepStimuliVersion), 'correlationsTimeSeries');

%%
% Metric 1, Vishal's

function [correctPred, incorrectPred] = metric1(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt)

    correctPred = corrAttEst_Att;
    incorrectPred = corrAttEst_UnAtt;

end

% Metric 2, Nima's

function [correctPred, incorrectPred] = metric2(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt)

    correctPred = corrAttEst_Att - corrUnAttEst_Att;
    incorrectPred = corrAttEst_UnAtt - corrUnAttEst_UnAtt;

end

% Metric 3, Vishal's Second

function [correctPred, incorrectPred] = metric3(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt)

    correctPred = corrAttEst_Att + corrUnAttEst_UnAtt;
    incorrectPred = corrAttEst_UnAtt + corrUnAttEst_Att;

end

% Metric 4, Vishal's CCA voting on three

function votedCCs = metric4(corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt)

    temp = corrAttEst_Att - corrAttEst_UnAtt;
    temp = temp > 0;
    votedCCs = sum(temp, 2);

end

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

function [newCellArray1, newCellArray2] = applyPCASpectrograms(inputCellArray1, inputCellArray2)

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

%% Function that returns correlations for all windows

function [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelations(toPassNeuralResponse, toPassStimuliDataAttend, toPassStimuliDataUnAttend, toPassGroupInformation, g, fNeuralData, thisWindowSize, overlap, slidingNeuralWindowDurn)

    corrAttEst_Att = [];
    corrAttEst_UnAtt = [];
    
    corrUnAttEst_Att = [];
    corrUnAttEst_UnAtt = [];
    
    windowLabels_fromAttLoop = [];
    windowLabels_fromUnAttLoop = [];

    % Processing preparation
    hopSize = (1 - overlap/100) * thisWindowSize * fNeuralData;
    noTrials = length(toPassNeuralResponse);
    Nmax = slidingNeuralWindowDurn * fNeuralData; % May not be relevant for CCA
    
    % Populate the testing correlations
    
    %%%%%%%%%%%%
    %    gA    %
    %%%%%%%%%%%%
    
    for trialCTR = 1:1:noTrials
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load accumulator - att
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        iBest = g.gA.iBest;
        AA = g.gA.AA{trialCTR}(:, :, iBest);
        BB = g.gA.BB{trialCTR}(:, :, iBest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        thisTrialNeuralResponse = toPassNeuralResponse{trialCTR};
        thisTrialStimuliDataAttend = toPassStimuliDataAttend{trialCTR};
        thisTrialStimuliDataUnAttned = toPassStimuliDataUnAttend{trialCTR};
        
        thisTrialGroupInformation = toPassGroupInformation{trialCTR};
        
        windowStartIDXs = [];
        windowEndIDXs = [];
        
        endflag = 0;
        samplePTR = 1;
        
        % Obtain window start and end idxs
        
        while endflag == 0
            
            thisWindowStartIDX = samplePTR;
            thisWindowEndIDX = samplePTR + thisWindowSize * fNeuralData - 1;
            
            if thisWindowEndIDX + Nmax  <= size(thisTrialNeuralResponse, 1) && thisWindowEndIDX <= size(thisTrialStimuliDataAttend, 1)
                
                windowStartIDXs = [windowStartIDXs, thisWindowStartIDX];
                windowEndIDXs   = [windowEndIDXs, thisWindowEndIDX];
                
                samplePTR = samplePTR + hopSize;
                
            else
                
                endflag = 1;
                
            end
            
        end
        
        % totWindows = totWindows + length(windowStartIDXs);
        
        for windowCTR = 1:1:length(windowStartIDXs)
            
            thisWindowStartIDX  = windowStartIDXs(windowCTR);
            thisWindowEndIDX    = windowEndIDXs(windowCTR);
            
            % thisWindowNeuralResponse = thisTrialNeuralResponse(:, thisWindowStartIDX:thisWindowEndIDX + Nmax); % check if this is correct for NAPLab Stim Recon package
            thisWindowNeuralResponse = thisTrialNeuralResponse(thisWindowStartIDX:thisWindowEndIDX, :); % For NAPLab Stim Recon package
            thisWindowStimuliDataAttend = thisTrialStimuliDataAttend(thisWindowStartIDX:thisWindowEndIDX, :);
            thisWindowStimuliDataUnAttend = thisTrialStimuliDataUnAttned(thisWindowStartIDX:thisWindowEndIDX, :);
            
            RR_Attend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            RR_UnAttend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataUnAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            
            if length(RR_Attend) >= 3
            
                corrAttEst_Att = [corrAttEst_Att;  RR_Attend(1:3)'];
                corrAttEst_UnAtt = [corrAttEst_UnAtt; RR_UnAttend(1:3)'];
                
            else
                
                corrAttEst_Att = [corrAttEst_Att;  RR_Attend];
                corrAttEst_UnAtt = [corrAttEst_UnAtt; RR_UnAttend];
                
            end
            
            windowLabels_fromAttLoop = [windowLabels_fromAttLoop, thisTrialGroupInformation + "_" + string(windowCTR)];
                        
        end
         
    end
    
    %%%%%%%%%%%%
    %    gU    %
    %%%%%%%%%%%%
    
    for trialCTR = 1:1:noTrials
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load accumulator - unAtt filters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        iBest = g.gU.iBest;
        AA = g.gU.AA{trialCTR}(:, :, iBest);
        BB = g.gU.BB{trialCTR}(:, :, iBest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        thisTrialNeuralResponse = toPassNeuralResponse{trialCTR};
        thisTrialStimuliDataAttend = toPassStimuliDataAttend{trialCTR};
        thisTrialStimuliDataUnAttned = toPassStimuliDataUnAttend{trialCTR};
        
        thisTrialGroupInformation = toPassGroupInformation{trialCTR};
        
        windowStartIDXs = [];
        windowEndIDXs = [];
        
        endflag = 0;
        samplePTR = 1;
        
        % Obtain window start and end idxs
        
        while endflag == 0
            
            thisWindowStartIDX = samplePTR;
            thisWindowEndIDX = samplePTR + thisWindowSize * fNeuralData - 1;
            
            if thisWindowEndIDX + Nmax  <= size(thisTrialNeuralResponse, 1) && thisWindowEndIDX <= size(thisTrialStimuliDataAttend, 1)
                
                windowStartIDXs = [windowStartIDXs, thisWindowStartIDX];
                windowEndIDXs   = [windowEndIDXs, thisWindowEndIDX];
                
                samplePTR = samplePTR + hopSize;
                
            else
                
                endflag = 1;
                
            end
            
        end
        
        % totWindows = totWindows + length(windowStartIDXs);
        
        for windowCTR = 1:1:length(windowStartIDXs)
            
            thisWindowStartIDX  = windowStartIDXs(windowCTR);
            thisWindowEndIDX    = windowEndIDXs(windowCTR);
            
            % thisWindowNeuralResponse = thisTrialNeuralResponse(:, thisWindowStartIDX:thisWindowEndIDX + Nmax); % check if this is correct for NAPLab Stim Recon package
            thisWindowNeuralResponse = thisTrialNeuralResponse(thisWindowStartIDX:thisWindowEndIDX, :); % For NAPLab Stim Recon package
            thisWindowStimuliDataAttend = thisTrialStimuliDataAttend(thisWindowStartIDX:thisWindowEndIDX, :);
            thisWindowStimuliDataUnAttend = thisTrialStimuliDataUnAttned(thisWindowStartIDX:thisWindowEndIDX, :);
            
            RR_Attend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            RR_UnAttend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataUnAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            
            
            if length(RR_Attend) >=3 
                
                corrUnAttEst_Att = [corrUnAttEst_Att; RR_Attend(1:3)'];         % Looking at only the first 3 CC
                corrUnAttEst_UnAtt = [corrUnAttEst_UnAtt; RR_UnAttend(1:3)'];   % Looking at only the first 3 CC   
                
            else
                
                corrUnAttEst_Att = [corrUnAttEst_Att; RR_Attend];         % Looking at only the first 3 CC
                corrUnAttEst_UnAtt = [corrUnAttEst_UnAtt; RR_UnAttend];   % Looking at only the first 3 CC                 
                
            end
            

            
            windowLabels_fromUnAttLoop = [windowLabels_fromUnAttLoop, thisTrialGroupInformation + "_" + string(windowCTR)];
                        
        end
         
    end
    
    assert(prod(strcmp(windowLabels_fromAttLoop, windowLabels_fromUnAttLoop), 'all') == 1);
    windowLabels = windowLabels_fromAttLoop;
    
end

%% Function that returns correlations as a time series vector

function [corrAttEst_Att, corrAttEst_UnAtt, corrUnAttEst_Att, corrUnAttEst_UnAtt, windowLabels] = getCorrelationsTimeSeries(toPassNeuralResponse, toPassStimuliDataAttend, toPassStimuliDataUnAttend, toPassGroupInformation, g, fNeuralData, thisWindowSize, slidingNeuralWindowDurn) % Remove overlap

    corrAttEst_Att = [];
    corrAttEst_UnAtt = [];
    
    corrUnAttEst_Att = [];
    corrUnAttEst_UnAtt = [];
    
    windowLabels_fromAttLoop = [];
    windowLabels_fromUnAttLoop = [];

    % Processing preparation
    hopSize = 1; %(1 - overlap/100) * thisWindowSize * fNeuralData;
    noTrials = length(toPassNeuralResponse);
    Nmax = slidingNeuralWindowDurn * fNeuralData; % May not be relevant for CCA
    
    % Populate the testing correlations
    
    %%%%%%%%%%%%
    %    gA    %
    %%%%%%%%%%%%
    
    for trialCTR = 1:1:noTrials
        
        fprintf("Att loop - processing trial %d of %d\n", trialCTR, noTrials);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load accumulator - att
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        iBest = g.gA.iBest;
        AA = g.gA.AA{trialCTR}(:, :, iBest);
        BB = g.gA.BB{trialCTR}(:, :, iBest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        thisTrialNeuralResponse = toPassNeuralResponse{trialCTR};
        thisTrialStimuliDataAttend = toPassStimuliDataAttend{trialCTR};
        thisTrialStimuliDataUnAttned = toPassStimuliDataUnAttend{trialCTR};
        
        thisTrialGroupInformation = toPassGroupInformation{trialCTR};
        
        windowStartIDXs = [];
        windowEndIDXs = [];
        
        endflag = 0;
        samplePTR = 1;
        
        % Obtain window start and end idxs
        
        while endflag == 0
            
            thisWindowStartIDX = samplePTR;
            thisWindowEndIDX = samplePTR + thisWindowSize * fNeuralData - 1;
            
            if thisWindowEndIDX + Nmax  <= size(thisTrialNeuralResponse, 1) && thisWindowEndIDX <= size(thisTrialStimuliDataAttend, 1)
                
                windowStartIDXs = [windowStartIDXs, thisWindowStartIDX];
                windowEndIDXs   = [windowEndIDXs, thisWindowEndIDX];
                
                samplePTR = samplePTR + hopSize;
                
            else
                
                endflag = 1;
                
            end
            
        end
        
        % totWindows = totWindows + length(windowStartIDXs);
        
        for windowCTR = 1:1:length(windowStartIDXs)
            
            thisWindowStartIDX  = windowStartIDXs(windowCTR);
            thisWindowEndIDX    = windowEndIDXs(windowCTR);
            
            % thisWindowNeuralResponse = thisTrialNeuralResponse(:, thisWindowStartIDX:thisWindowEndIDX + Nmax); % check if this is correct for NAPLab Stim Recon package
            thisWindowNeuralResponse = thisTrialNeuralResponse(thisWindowStartIDX:thisWindowEndIDX, :); % For NAPLab Stim Recon package
            thisWindowStimuliDataAttend = thisTrialStimuliDataAttend(thisWindowStartIDX:thisWindowEndIDX, :);
            thisWindowStimuliDataUnAttend = thisTrialStimuliDataUnAttned(thisWindowStartIDX:thisWindowEndIDX, :);
            
            RR_Attend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            RR_UnAttend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataUnAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            
            if length(RR_Attend) >= 3
            
                corrAttEst_Att = [corrAttEst_Att;  RR_Attend(1:3)'];
                corrAttEst_UnAtt = [corrAttEst_UnAtt; RR_UnAttend(1:3)'];
                
            else
                
                corrAttEst_Att = [corrAttEst_Att;  RR_Attend];
                corrAttEst_UnAtt = [corrAttEst_UnAtt; RR_UnAttend];
                
            end
            
            windowLabels_fromAttLoop = [windowLabels_fromAttLoop, thisTrialGroupInformation + "_" + string(windowCTR)];
                        
        end
         
    end
    
    %%%%%%%%%%%%
    %    gU    %
    %%%%%%%%%%%%
    
    for trialCTR = 1:1:noTrials
        
        fprintf("UnAtt loop - processing trial %d of %d\n", trialCTR, noTrials);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load accumulator - unAtt filters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        iBest = g.gU.iBest;
        AA = g.gU.AA{trialCTR}(:, :, iBest);
        BB = g.gU.BB{trialCTR}(:, :, iBest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        thisTrialNeuralResponse = toPassNeuralResponse{trialCTR};
        thisTrialStimuliDataAttend = toPassStimuliDataAttend{trialCTR};
        thisTrialStimuliDataUnAttned = toPassStimuliDataUnAttend{trialCTR};
        
        thisTrialGroupInformation = toPassGroupInformation{trialCTR};
        
        windowStartIDXs = [];
        windowEndIDXs = [];
        
        endflag = 0;
        samplePTR = 1;
        
        % Obtain window start and end idxs
        
        while endflag == 0
            
            thisWindowStartIDX = samplePTR;
            thisWindowEndIDX = samplePTR + thisWindowSize * fNeuralData - 1;
            
            if thisWindowEndIDX + Nmax  <= size(thisTrialNeuralResponse, 1) && thisWindowEndIDX <= size(thisTrialStimuliDataAttend, 1)
                
                windowStartIDXs = [windowStartIDXs, thisWindowStartIDX];
                windowEndIDXs   = [windowEndIDXs, thisWindowEndIDX];
                
                samplePTR = samplePTR + hopSize;
                
            else
                
                endflag = 1;
                
            end
            
        end
        
        % totWindows = totWindows + length(windowStartIDXs);
        
        for windowCTR = 1:1:length(windowStartIDXs)
            
            thisWindowStartIDX  = windowStartIDXs(windowCTR);
            thisWindowEndIDX    = windowEndIDXs(windowCTR);
            
            % thisWindowNeuralResponse = thisTrialNeuralResponse(:, thisWindowStartIDX:thisWindowEndIDX + Nmax); % check if this is correct for NAPLab Stim Recon package
            thisWindowNeuralResponse = thisTrialNeuralResponse(thisWindowStartIDX:thisWindowEndIDX, :); % For NAPLab Stim Recon package
            thisWindowStimuliDataAttend = thisTrialStimuliDataAttend(thisWindowStartIDX:thisWindowEndIDX, :);
            thisWindowStimuliDataUnAttend = thisTrialStimuliDataUnAttned(thisWindowStartIDX:thisWindowEndIDX, :);
            
            RR_Attend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            RR_UnAttend = getCanonicalCorrelations(thisWindowNeuralResponse, thisWindowStimuliDataUnAttend, AA, BB, g.MetaData.shifts_samples(iBest));
            
            
            if length(RR_Attend) >=3 
                
                corrUnAttEst_Att = [corrUnAttEst_Att; RR_Attend(1:3)'];         % Looking at only the first 3 CC
                corrUnAttEst_UnAtt = [corrUnAttEst_UnAtt; RR_UnAttend(1:3)'];   % Looking at only the first 3 CC   
                
            else
                
                corrUnAttEst_Att = [corrUnAttEst_Att; RR_Attend];         % Looking at only the first 3 CC
                corrUnAttEst_UnAtt = [corrUnAttEst_UnAtt; RR_UnAttend];   % Looking at only the first 3 CC                 
                
            end
            

            
            windowLabels_fromUnAttLoop = [windowLabels_fromUnAttLoop, thisTrialGroupInformation + "_" + string(windowCTR)];
                        
        end
         
    end
    
    assert(prod(strcmp(windowLabels_fromAttLoop, windowLabels_fromUnAttLoop), 'all') == 1);
    windowLabels = windowLabels_fromAttLoop;
    
end

%% Function to obtain canonical correlations

function RR = getCanonicalCorrelations(xx, yy, AA, BB, shift)

    % RR = (nComp, 1)

    [xxx, yyy] = nt_relshift(xx, yy, shift);
    xxx = nt_normcol( nt_demean( nt_mmat(xxx,AA) ) );
    yyy = nt_normcol( nt_demean( nt_mmat(yyy,BB) ) );
    
    x = xxx;
    y = yyy;
    
    RR = diag(x'*y) / size(x,1);

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