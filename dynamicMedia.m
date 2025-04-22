classdef dynamicMedia < GroundTruth_Generate
    % dynamicMedia 此类用来生成动态介质
    % 它继承自 GroundTruth_Generate 并利用已有的参数和功能生成动态介质。

    properties
        random_wave    % 随机复振幅，存储为 cell 数组，每个元素大小为 obj.unitNums(2) * obj.T
    end
    
    properties
        d              % 生成的动态介质，存储为 cell 数组，每个元素大小为 obj.unitNums(2) * obj.T
    end

    methods
        function obj = dynamicMedia(unitSize, unitWidth, layerDistance, frequency, radius, object_field)
            % 构造函数：调用父类 GroundTruth_Generate 的构造函数
            obj@GroundTruth_Generate(unitSize, unitWidth, layerDistance, frequency, radius, object_field);

            % 初始化 random_wave 和 d 为 cell 数组
            obj.random_wave = cell(obj.Maskpatterns);
            obj.d = cell(obj.Maskpatterns);

            for ii = 1:obj.Maskpatterns
                % 为每个掩膜模式生成一个随机的复振幅
                obj.random_wave{ii} = randn(obj.unitNums(2), obj.T); 
                random_wave = obj.random_wave{ii};
                % 生成动态介质
                obj.d{ii} = obj.dynamicMediaGenerator(random_wave);
            end
        end

        function random_wave_filtered = dynamicMediaGenerator(obj, random_wave)
            % 生成动态介质
            % 输入：
            %   random_wave - 随机复振幅，大小为 obj.unitNums(2) x obj.T
            % 输出：
            %   random_wave_filtered - 经滤波器处理后的随机复振幅

            random_wave_filtered = zeros(obj.unitNums(2), obj.T);
            spatialFrequencyFilter = obj.S;  % 采用静态滤波器 S

            for tt = 1:obj.T
                % 对每个时间采样进行处理
                random_wave_tt = reshape(random_wave(:, tt), [], 1); % 对当前时间的振幅进行展平
                spatialFrequencyFilter = reshape(spatialFrequencyFilter, [], 1);  % 确保滤波器形状一致

                % FFT 变换并施加低通滤波器
                wave_fft = fft(random_wave_tt);  
                wave_fft = fftshift(wave_fft);  % 将低频移至中心

                wave_filter = wave_fft .* spatialFrequencyFilter;  % 施加滤波器

                random_wave_filter = ifftshift(wave_filter);  % 反向移频
                random_wave_filtered(:, tt) = ifft(random_wave_filter);  % IFFT 还原信号
            end
        end
    end
end
