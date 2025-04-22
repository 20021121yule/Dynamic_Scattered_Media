classdef DSM < GroundTruth_Generate
    % 继承自 GroundTruth_Generate，用于模型的反问题求解
    % 主要修改包括：在正向预测中引入 T 次随机采样，梯度更新时累加每个 mask 模式与时间采样 t 的贡献，
    % 并分离目标 O 与频率滤波器 predFreqFilter 的学习率。

    properties (SetAccess = private, GetAccess = public)
        O               % 待重构的目标光场，大小为 unitNums(1) x 1
        predFreqFilter  % 待重构的散射介质静态频率滤波器 s，尺寸为 unitSize(2) x unitSize(2)
    end

    properties (SetAccess = private, GetAccess = public)
        trainingOptions   % 训练参数结构体，包含 MaxEpochs, InitialLearnRate_O, InitialLearnRate_S 等
        trainStartTime    % 训练起始时间
        trainDuration     % 训练总时间
        mu_O              % 目标光场的学习率
        mu_S              % 滤波器的学习率
        iter              % 当前迭代次数
        RMSE              % 存储每次迭代的 RMSE

        % Adam 算法相关变量，用于自动优化参数更新
        averageGrad_O = [];  % 目标光场 O 的一阶梯度均值（动量）
        averageSqGrad_O = []; % 目标光场 O 的二阶梯度均值

        averageGrad_S = [];   % 频率滤波器 s 的一阶梯度均值
        averageSqGrad_S = []; % 频率滤波器 s 的二阶梯度均值

        gradDecay = 0.75;     % Adam 算法一阶矩衰减因子
        sqGradDecay = 0.95;   % Adam 算法二阶矩衰减因子
    end

    methods(Access = public)
        function obj = DSM(unitSize, unitWidth, layerDistance, frequency, radius, maskpatterns, time, object_field, trainingOptions)
            % 构造函数：调用超类构造器生成 groundTruth 后初始化反问题变量
            obj@GroundTruth_Generate(unitSize, unitWidth, layerDistance, frequency, radius, maskpatterns, time, object_field);

            % 初始化目标 O 和频率滤波器 predFreqFilter
            obj.O = rand(obj.unitNums(1), 1);
            obj.predFreqFilter = rand(obj.unitNums(2), 1);

            % 复制训练参数，并初始化学习率
            obj.trainingOptions = trainingOptions;
            obj.mu_O = trainingOptions.InitialLearnRate_O;
            obj.mu_S = trainingOptions.InitialLearnRate_S;

            obj.iter = 1;
            obj.RMSE = zeros(trainingOptions.MaxEpochs, 1);
        end

        function [predictedComplexField, predictedField, predDynamicMedia] = netPredict(obj)
            % 计算正向传播预测
            predictedComplexField = cell(obj.Maskpatterns, 1);
            predictedField = zeros(obj.unitNums(4), obj.Maskpatterns);
            predDynamicMedia = cell(obj.Maskpatterns, 1);

            for ii = 1:obj.Maskpatterns
                predDynamicMedia{ii} = obj.dynamicMediaGenerator(obj.random_wave{ii}, obj.predFreqFilter);

                current_g = obj.P{1} * obj.O;
                current_g = predDynamicMedia{ii} .* current_g;
                current_g = obj.P{2} * current_g;

                predictedComplexField{ii} = obj.P{3} * (current_g .* obj.mask(:, ii));
                predictedField(:, ii) = mean(abs(predictedComplexField{ii}).^2, 2);
            end

        end

        function obj = gradientUpdate(obj)
            % 梯度更新：累加每个 mask 模式与每个时间采样的贡献，并更新 O 和 predFreqFilter
            [predComplexField, predField, predDynamicMedia] = obj.netPredict();
            e = predField - obj.groundTruth;  % 计算误差

            grad_O_temp = zeros(obj.unitNums(1), obj.Maskpatterns);
            grad_S_temp = zeros(obj.unitNums(2), obj.Maskpatterns);
            se_m = 0;

            for ii = 1:obj.Maskpatterns
                e_m = e(:, ii);
                complexfield = predComplexField{ii};
                mask = obj.mask(:, ii);

                tmp = complexfield .* e_m;
                tmp = obj.P{3}' * tmp;
                tmp = mask .* tmp;
                tmp = obj.P{2}' * tmp;

                % 计算 O 的梯度
                tmp_grad_O = conj(predDynamicMedia{ii}) .* tmp;
                tmp_grad_O = obj.P{1}' * tmp_grad_O;
                grad_O_temp(:, ii) = mean(tmp_grad_O, 2);

                % 计算 S 的梯度
                tmp_grad_S = conj(obj.P{1} * obj.O) .* tmp;  % 计算 S 梯度时的中间变量
                tmp_grad_S = fftshift(fft(tmp_grad_S));  % 进行 FFT 变换
                temp = fftshift(fft(obj.random_wave{ii}));  % 计算随机波的 FFT
                tmp_grad_S = real(conj(temp) .* tmp_grad_S);  % 按照论文的方式计算 S 的梯度
                grad_S_temp(:, ii) = mean(real(tmp_grad_S), 2);  % 平均化得到最终的梯

                threshold = 1e2;
                grad_S_temp(abs(grad_S_temp) > threshold) = 0;

                se_m = se_m + sum(e_m(:).^2);  % 累加误差
            end

            % 按论文中的归一化系数计算梯度
            factor = 4 / (obj.Maskpatterns * obj.T);
            grad_O = factor * mean(grad_O_temp, 2);
            grad_S = factor * mean(grad_S_temp, 2);

            % 更新参数（简单的梯度下降）
            obj.O = obj.O - obj.mu_O * grad_O;
            obj.predFreqFilter = obj.predFreqFilter - obj.mu_S * grad_S;

            % 计算当前 RMSE（均方根误差）
            obj.RMSE(obj.iter) = se_m / obj.Maskpatterns;
        end

        function obj = trainDSM(obj)
            % 训练循环，根据训练参数进行多次迭代更新
            obj.trainStartTime = datetime('now');
            fprintf('Duration\tIter\tMaxEpochs\tRMSE\n');
            for epoch = 1:obj.trainingOptions.MaxEpochs
                % 执行一次梯度更新
                obj = obj.gradientUpdate();

                % 学习率调整（每 LearnRateDropPeriod 个 epoch 衰减一次）
                if mod(epoch, obj.trainingOptions.LearnRateDropPeriod) == 0
                    obj.mu_O = obj.mu_O * obj.trainingOptions.LearnRateDropFactor;
                    obj.mu_S = obj.mu_S * obj.trainingOptions.LearnRateDropFactor;
                end

                % 每隔一定次数打印当前进度
                if mod(obj.iter, max(1, floor(obj.trainingOptions.MaxEpochs / 10))) == 0
                    duration = datetime('now') - obj.trainStartTime;
                    fprintf('%s\t%d/%d\t%.3e\n', char(duration), obj.iter, obj.trainingOptions.MaxEpochs, obj.RMSE(obj.iter));
                end

                obj.iter = obj.iter + 1;
            end
            obj.trainDuration = datetime('now') - obj.trainStartTime;
        end
    end
end
