%% 清空工作区并关闭所有图窗
clear; close all; clc;

%% 读取并处理图片
img = imread('peppers.png');              % 读取图片
img_gray = double(rgb2gray(img));         % 转换为灰度图，并转换为 double 类型
img_resized = imresize(img_gray, [28, 28], 'bilinear');  % 调整到 28×28
img_resized = img_resized / max(img_resized(:));         % 归一化

% 利用图像旋转生成相位分量（这里模拟 mandrill 图像旋转 90°）
phase = rot90(img_resized, 1); 
complex_obj = img_resized .* exp(1j * 2 * pi * phase);

% 显示输入的复振幅图像
figure;
imagesc(abs(complex_obj)); axis image;
colormap gray;
title('输入的复振幅图像');

% 将复振幅场转换为列向量，便于后续运算
complex_obj = reshape(complex_obj, [], 1);

%% 参数设置
% 结构参数（这里假设系统包含三个层面：物体、介质（滤波器层）和传感器）
unitSize = [28, 28, 28, 28];               % 每层单元数目（每层为 28×28）
unitWidth = [4.5, 4.5, 4.5, 4.5];           % 每个单元的边长（单位 mm）
layerDistance = [0.01, 0.01, 0.01];                % 层间距离（单位 mm）,注意不要太大，如果太大，训练效果会很糟糕

% 光学参数
frequency = 26.8e9;                  % 频率（Hz）
radius = 10;                         % 低通滤波器的半径（描述散射介质的静态特征）
sigma = 4;
maskpatterns = 20;
T = 10;

%% 生成 GroundTruth
% 该对象用于生成正向传播得到的传感器上接收的光场（复振幅及其强度）
groundtruth = GroundTruth_Generate(unitSize, unitWidth, layerDistance, frequency, radius, maskpatterns, T, complex_obj);

%% 训练参数设置
trainingOptions = struct();
trainingOptions.MaxEpochs = 1000;              % 最大训练轮数
trainingOptions.InitialLearnRate_O = 0.3;     % O 的初始学习率
trainingOptions.InitialLearnRate_S = 0.3;     % 频率滤波器的初始学习率
trainingOptions.LearnRateDropPeriod = 100000;     % 学习率衰减周期
trainingOptions.LearnRateDropFactor = 0.9;    % 学习率衰减因子

%% 初始化 DSM 类进行训练
dsmModel = DSM(unitSize, unitWidth, layerDistance, frequency, radius, maskpatterns, T, complex_obj, trainingOptions);

% 开始训练
dsmModel = dsmModel.trainDSM();

% 显示最终训练的结果
figure;
imagesc(abs(reshape(dsmModel.O, unitSize(1), unitSize(1))));  % 还原为 28×28 图像并显示
axis image; colormap gray;
title('训练后的重构目标光场');
colorbar;

figure;
imagesc((reshape(dsmModel.predFreqFilter, unitSize(2), unitSize(2))));  % 还原为 28×28 图像并显示
axis image; colormap gray;
title('训练后的重构目标滤波器');
colorbar;

%% 显示训练过程中 RMSE 的变化
figure;
plot(1:trainingOptions.MaxEpochs, dsmModel.RMSE, '-');
xlabel('Epoch');
ylabel('RMSE');
title('训练过程中的 RMSE 变化');
grid on;
