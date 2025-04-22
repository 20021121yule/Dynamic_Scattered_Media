classdef GroundTruth_Generate
    % 该类用于生成训练所需的 GroundTruth
    % 主要修改包括：引入 T 次随机采样、时间平均以及正确的正向传播顺序。

    properties (Constant, GetAccess = private)
        c0 = 299792458 * 1e3;   % 光速，单位 mm/s
    end

    properties (SetAccess = immutable, GetAccess = public)
        unitSize        % 每层单元的尺寸，大小 4 x 1
        unitNums        % 每层单元数，大小 4 x 1，等于 unitSize.^2
        unitWidth       % 每层单元的边长（mm），大小 4 x 1
        layerDistance   % 层间距离（mm），大小 3 x 1
        frequency       % 频率（Hz）
    end

    properties (SetAccess = immutable, GetAccess = protected)
        T               % 数据采集时刻数
    end

    properties (SetAccess = immutable, GetAccess = protected)
        Maskpatterns    % 掩膜模式数量
    end

    properties (SetAccess = private, GetAccess = protected)
        mask            % 掩膜矩阵
    end

    properties (SetAccess = private, GetAccess = protected)
        P               % 衍射矩阵（cell 数组，大小 3 x 1）
    end

    properties (SetAccess = private, GetAccess = protected)
        S                % 用于生成介质的低通滤波器
    end

    properties (SetAccess = private, GetAccess = protected)
        random_wave      % 随机复振幅，cell 数组,用于生成动态介质
        random_wave_pred % 随机复振幅，cell 数组,用于生成反向梯度计算

        d                % 生成的动态介质，cell 数组
    end

    properties (SetAccess = private, GetAccess = protected)
        g              % 存储每个 mask 模式下 T 次正向传播得到的复杂振幅场（cell 数组）
        groundTruth    % 传感器上接收到的光场强度（平均 T 次）
    end

    methods (Access = public)
        function obj = GroundTruth_Generate(unitSize, unitWidth, layerDistance, frequency, radius, maskpatterns, time, object_field)
            % 构造函数：初始化所有参数
            assert(length(unitSize) == 4, 'unitSize must be 4x1');
            assert(length(unitWidth) == 4, 'unitWidth must be 4x1');
            assert(length(layerDistance) == 3, 'layerDistance must be 3x1');
            assert(layerDistance(1) >= 0 && all(layerDistance(2:end) > 0), 'Layer distances must be positive');

            obj.unitSize = unitSize;
            obj.unitNums = unitSize .^ 2;
            obj.unitWidth = unitWidth;
            obj.layerDistance = layerDistance;
            obj.frequency = frequency;

            obj.Maskpatterns = maskpatterns;
            obj.T = time;

            % 生成低通滤波器，用于描述散射介质的静态特征
            obj.S = reshape(obj.create_circular_low_pass_filter(radius), [], 1);

            % 生成掩膜模式
            obj.mask = obj.maskPatternGenerate();

            % 初始化动态介质和随机波
            [obj.d, obj.random_wave, obj.random_wave_pred] = obj.dynamicMedia(obj.S);

            % 生成衍射矩阵 P
            obj.P = cell(3, 1);
            for ii = 1:3
                obj.P{ii} = obj.PGenerate(obj.unitWidth([ii, ii + 1]), obj.unitSize([ii, ii + 1]), layerDistance(ii), frequency);
            end

            % 生成 GroundTruth：对每个 mask 模式进行 T 次随机散射，再做时间平均
            obj = obj.computeSensorFeild(object_field);
        end

        % 生成 Fresnel 传播矩阵
        function P = PGenerate(obj, width, unitSize, dis, freq)
            k = 2 * pi * freq / obj.c0;
            l1 = width(1) * ((1:unitSize(1)) - (unitSize(1) + 1) / 2);
            l2 = width(2) * ((1:unitSize(2)) - (unitSize(2) + 1) / 2);
            [x1, y1] = meshgrid(l1, l1);
            [x2, y2] = meshgrid(l2, l2);
            x1 = x1(:)'; y1 = y1(:)';
            x2 = x2(:); y2 = y2(:);
            r = sqrt((x1 - x2) .^ 2 + (y1 - y2) .^ 2 + dis ^ 2);
            P = exp(-1j * k * r) ./ (r .^ 2);
            P = P / norm(P, 2);
        end

        % 创建圆形低通滤波器
        function spatialFrequencyFilter = create_circular_low_pass_filter(obj, radius)
            [X, Y] = meshgrid(-obj.unitSize(2) / 2:obj.unitSize(2) / 2 - 1, -obj.unitSize(2) / 2:obj.unitSize(2) / 2 - 1);
            distance = sqrt(X .^ 2 + Y .^ 2);
            spatialFrequencyFilter = double(distance <= radius);
        end

        function spatialFrequencyFilter = create_Gauss_low_pass_filter(obj, sigma)

            % 生成网格，计算每个点的距离
            [X, Y] = meshgrid(-obj.unitSize(2)/2:obj.unitSize(2)/2-1, -obj.unitSize(2)/2:obj.unitSize(2)/2-1);
            % 计算每个点到中心的距离
            distance = sqrt(X.^2 + Y.^2);

            % 计算高斯滤波器
            factor = 1;
            spatialFrequencyFilter = factor * exp(- (distance.^2) / (2 * sigma^2));
        end


        % 生成随机掩膜模式
        function mask = maskPatternGenerate(obj)
            num_ones_per_column = round(0.25 * obj.unitNums(3));
            mask = zeros(obj.unitNums(3), obj.Maskpatterns);  % 初始化为全 0

            for col = 1:obj.Maskpatterns
                idx = randperm(obj.unitNums(3), num_ones_per_column);
                mask(idx, col) = 1;
            end

            mask = logical(mask);  % 转为逻辑类型
        end

        % 初始化动态介质
        function [d, random_wave, random_wave_pred] = dynamicMedia(obj, spatialFrequencyFilter)
            random_wave = cell(obj.Maskpatterns, 1);
            random_wave_pred = cell(obj.Maskpatterns, 1);
            d = cell(obj.Maskpatterns, 1);

            for ii = 1:obj.Maskpatterns
                random_Amp = 10 * rand(obj.unitNums(2), obj.T);
                random_Phase = rand(obj.unitNums(2), obj.T);
                random_Amp_pred = 10 * rand(obj.unitNums(2), obj.T);
                random_Phase_pred = rand(obj.unitNums(2), obj.T);

                random_wave{ii} = random_Amp .* exp(1j * random_Phase);
                random_wave_pred{ii} = random_Amp_pred .* exp(1j * random_Phase_pred);

                d{ii} = obj.dynamicMediaGenerator(random_wave{ii}, spatialFrequencyFilter);
            end

        end

        % 生成滤波后的动态介质
        function random_wave_filtered = dynamicMediaGenerator(obj, random_wave, spatialFrequencyFilter)
            random_wave_filtered = zeros(obj.unitNums(2), obj.T);

            for tt = 1:obj.T
                random_wave_tt = reshape(random_wave(:, tt), [], 1);
                spatialFrequencyFilter = reshape(spatialFrequencyFilter, [], 1);
                wave_fft = fft(random_wave_tt);
                wave_fft = fftshift(wave_fft);
                wave_filter = wave_fft .* spatialFrequencyFilter;
                random_wave_filter = ifftshift(wave_filter);
                random_wave_filtered(:, tt) = ifft(random_wave_filter);
            end

        end

        % 计算传感器接收到的光场强度
        function obj = computeSensorFeild(obj, object_field)
            obj.groundTruth = zeros(obj.unitNums(4), obj.Maskpatterns);
            obj.g = cell(obj.Maskpatterns, 1);

            for ii = 1:obj.Maskpatterns
                current_g = obj.P{1} * object_field;
                current_g = obj.d{ii} .* current_g;
                current_g = obj.P{2} * current_g;
                obj.g{ii} = obj.P{3} * (current_g .* obj.mask(:, ii));
                obj.groundTruth(:, ii) = mean(abs(obj.g{ii}) .^ 2, 2);
            end

        end

    end
end
