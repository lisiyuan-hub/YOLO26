%% 医疗图像离线预处理展示脚本 (YOLO-GCE 单张对比版)
clear; clc; close all;

% ==================== 1. 配置与图像选择 ====================
cfg.inputDir = 'E:\Deeplearning\Datasets\BCCD\datasets\train\images'; 
cfg.gaborWavelength = [3, 6]; 
cfg.gaborOrientations = 0:45:135;

% 获取图像列表并随机选择一张
imds = imageDatastore(cfg.inputDir);
fileList = imds.Files;
imgIdx = randi(numel(fileList)); % 随机选择索引
[~, name, ext] = fileparts(fileList{imgIdx});
originalImg = imread(fileList{imgIdx});

fprintf('正在展示图像: %s\n', [name, ext]);

% 预处理：统一转为 RGB
if size(originalImg, 3) == 1
    originalImg = cat(3, originalImg, originalImg, originalImg);
end

% 创建 Gabor 滤波器组[cite: 1]
gaborBank = gabor(cfg.gaborWavelength, cfg.gaborOrientations);

% ==================== 2. 执行预处理管线 ====================

% --- 模块A: CLAHE 增强[cite: 1] ---
labImg = rgb2lab(originalImg);
L_channel = labImg(:,:,1) / 100;
L_clahe = adapthisteq(L_channel, 'NumTiles', [8 8], 'ClipLimit', 0.015);
labImg(:,:,1) = L_clahe * 100;
enhImg = lab2rgb(labImg);
enhImg_uint8 = im2uint8(enhImg);

% --- 模块B: 各向异性扩散平滑[cite: 1] ---
diffuseImg = imdiffusefilt(enhImg_uint8, 'NumberOfIterations', 3, 'ConductionMethod', 'exponential');

% --- 模块C: 弹性形变[cite: 1] ---
[h, w, ~] = size(originalImg);
alpha = 30; sigma = 8;
dx = imgaussfilt(randn(h, w) * alpha, sigma);
dy = imgaussfilt(randn(h, w) * alpha, sigma);
[X, Y] = meshgrid(1:w, 1:h);
D = cat(3, dx, dy);
elasticImg = imwarp(diffuseImg, D, 'SmoothEdges', true, 'Interp', 'cubic');

% --- 模块D: Gabor 纹理特征蒸馏[cite: 1] ---
grayImg = rgb2gray(diffuseImg);
[mag, ~] = imgaborfilt(grayImg, gaborBank);
maxMag = max(mag, [], 3);
maxMag = mat2gray(maxMag); 
distillImg = imfuse(diffuseImg, uint8(maxMag * 255), 'blend', 'Scaling', 'joint');

% ==================== 3. 结果可视化展示 ====================
figure('Name', 'YOLO-GCE 医疗图像预处理管线对比', 'Color', 'w', 'NumberTitle', 'off');

subplot(2,2,1);
imshow(originalImg);
title('1. 原始医疗图像 (Original)', 'FontSize', 12);

subplot(2,2,2);
imshow(diffuseImg);
title('2. CLAHE + 边缘保留平滑 (Anisotropic)', 'FontSize', 12);

subplot(2,2,3);
imshow(elasticImg);
title('3. 非刚性弹性形变 (Elastic Aug)', 'FontSize', 12);

subplot(2,2,4);
imshow(distillImg);
title('4. Gabor 特征蒸馏融合 (Distillation)', 'FontSize', 12);

% 自动调整窗口大小
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.8, 0.8]);
fprintf('展示完成。\n');