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

figure;
imagesc(abs(complex_obj)); axis image;
colormap gray;
colorbar;
title('输入的复振幅图像');

%% 低通滤波处理
radius = 10;
sigma = 2;
spatialFrequencyFilter = create_Gauss_low_pass_filter(sigma);

figure;
imagesc(spatialFrequencyFilter);
colormap gray;
title('空间滤波器');
colorbar;

function spatialFrequencyFilter = create_circular_low_pass_filter(radius)
% 创建圆形低通滤波器，尺寸为 unitSize(2) x unitSize(2)
[X, Y] = meshgrid(-28/2:28/2-1, -28/2:28/2-1);
distance = sqrt(X.^2 + Y.^2);
spatialFrequencyFilter = double(distance <= radius);
end

function spatialFrequencyFilter = create_Gauss_low_pass_filter(sigma)
% 创建圆形低通滤波器，尺寸为 unitSize(2) x unitSize(2)
[X, Y] = meshgrid(-28/2:28/2-1, -28/2:28/2-1);
factor = 1 / (2 * pi * sigma * sigma);
spatialFrequencyFilter = factor * exp(- (X.^2 + Y.^2)/(2 * sigma * sigma));
end

%% 图片频谱获取
% 先变换为一维
filter = reshape(spatialFrequencyFilter, [], 1);
complex_obj = reshape(complex_obj, [], 1);

% 计算得到频谱分布
obj_fft = fft(complex_obj);
obj_fft = fftshift(obj_fft);

% 滤波
obj_filter = obj_fft .* filter;

% fft逆变换
complex_obj_filter = ifftshift(obj_filter);
complex_obj_filter = ifft(complex_obj_filter);

complex_obj_filter = reshape(complex_obj_filter, 28, 28);

figure;
imagesc(abs(complex_obj_filter));
colormap gray;
title('滤波后图像');
colorbar;