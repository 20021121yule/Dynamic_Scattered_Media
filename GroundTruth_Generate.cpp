//
// Created by 余乐 on 25-4-19.
//

#include "GroundTruth_Generate.h"

GroundTruth_Generate::GroundTruth_Generate(const std::string &_filename, int _layerNum,
                                           float _layerDistance): layerNum(_layerNum), layerDistance(_layerDistance),
                                                                  filename(_filename) {
    const cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
    }

    // 转换成光场数据
    this->image_data = torch::zeros({image.rows, image.cols}); // 先占位置
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            this->image_data[i][j] = image.at<uchar>(i, j) / 255.0f;
        }
    }
    // 保存图片信息
    this->rows = image.rows;
    this->cols = image.cols;

    // 制作低通滤波器
    this->spatialFreqFilter = spatialFreqFilter_generater();

    // 生成动态介质，但是我们目前没有用他
    this->dynamicMedia = dynamicMedia_generator(spatialFreqFilter);

    // 生成掩膜Mask
    this->Mask = Mask_generator();

    // 传播路径
    for (int mm = 0; mm < this->mask_patterns; mm++) {
        torch::Tensor groundTruth_mm = torch::zeros_like(image_data);
        for (int tt = 0; tt < this->T; tt++) {
            // 注意input的作用域
            torch::Tensor input = this->image_data;
            for (int i = 0; i < this->layerNum; i++) {
                // 光场传播
                input = Wave_Propagation(input);
                // 下面我们直接单独考虑每层所需要的操作
                // 在第2层上是动态介质的操作
                if (i == 1) {
                    input = filter_process(input, spatialFreqFilter); // 在这里是随机动态介质的乘积
                }
                // 在第三层是光学掩膜的操作
                if (i == 2) {
                    input = input * Mask[mm]; // 在这里是光学掩膜
                }
            }
            // 在固定光学掩膜下，在时间上的累积
            groundTruth_mm = groundTruth_mm + input.abs().pow(2);

        }
        // 注意要保存的是在时间上的均值
        groundtruth.push_back(groundTruth_mm / this->T);
    }
}

torch::Tensor GroundTruth_Generate::Wave_Propagation(const torch::Tensor &input) const {
    const auto float_rows = static_cast<float>(rows);
    const auto float_cols = static_cast<float>(cols);
    const torch::Tensor k_x = 2 * M_PI / (pixel_size * float_rows) * torch::arange(-rows / 2, rows / 2);
    const torch::Tensor k_y = 2 * M_PI / (pixel_size * float_cols) * torch::arange(-cols / 2, cols / 2);

    // meshgrid操作
    const std::vector k_vec{k_x, k_y};
    const std::vector<torch::Tensor> k_meshgrid = meshgrid(k_vec);
    const torch::Tensor &k_x_mesh = k_meshgrid[0];
    const torch::Tensor &k_y_mesh = k_meshgrid[1];

    torch::Tensor k_z_mesh = sqrt(k * k - k_x_mesh * k_x_mesh - k_y_mesh * k_y_mesh);

    // 对k_z_mesh的检查
    torch::Tensor nan_mask = torch::isnan(k_z_mesh);
    if (torch::any(nan_mask).item<bool>()) {
        std::cerr << "Warning: k_z_mesh contains NaN values!" << std::endl;
    }

    // 构造传播函数
    torch::Tensor H_imag_part = k_z_mesh * layerDistance;
    torch::Tensor H_real_part = torch::zeros_like(H_imag_part, torch::kFloat);
    torch::Tensor exponent_part = complex(H_real_part, H_imag_part);
    torch::Tensor H = torch::exp(exponent_part); // H函数

    // 角谱定理具体实现
    torch::Tensor input_OptField_fft = torch::fft::fft2(input);
    torch::Tensor input_opt_field_fftshift = torch::fft::fftshift(input_OptField_fft);
    input_opt_field_fftshift = input_opt_field_fftshift * H;
    torch::Tensor opt_field_ifftshift = torch::fft::ifftshift(input_opt_field_fftshift);
    torch::Tensor input_opt_field_ifft = torch::fft::ifft2(input_opt_field_fftshift);

    return input_opt_field_ifft;
}

torch::Tensor GroundTruth_Generate::spatialFreqFilter_generater() const {
    const torch::Tensor x = torch::arange(-rows / 2, rows / 2);
    const torch::Tensor y = torch::arange(-rows / 2, rows / 2);

    // meshgrid操作
    const std::vector coord_vec{x, y};
    const std::vector<torch::Tensor> coord_meshgrid = meshgrid(coord_vec);
    const torch::Tensor xx = coord_meshgrid[0];
    const torch::Tensor yy = coord_meshgrid[1];

    const torch::Tensor distance = torch::sqrt(xx * xx + yy * yy);

    // 注意这里直接比较其实转换的是bool值，但是既然是Tensor类型，就可以直接当作我们生成的低通滤波器
    torch::Tensor spatial_freq_filter = distance < radius;

    return spatial_freq_filter;
}

torch::Tensor GroundTruth_Generate::spatialFreqFilter_Guass_generater() const {
    const torch::Tensor x = torch::arange(-rows / 2, rows / 2);
    const torch::Tensor y = torch::arange(-rows / 2, rows / 2);

    // meshgrid操作
    const std::vector coord_vec{x, y};
    const std::vector<torch::Tensor> coord_meshgrid = meshgrid(coord_vec);
    const torch::Tensor xx = coord_meshgrid[0];
    const torch::Tensor yy = coord_meshgrid[1];

    const torch::Tensor distance = torch::sqrt(xx * xx + yy * yy);

    float factor = 1;// 这里的factor本质上
    torch::Tensor spatial_freq_filter = factor * torch::exp(-distance.pow(2) / (2 * sigma * sigma));

    return spatial_freq_filter;
}

torch::Tensor GroundTruth_Generate::filter_process(const torch::Tensor &input, const torch::Tensor &filter) const {
    // 过滤低频操作
    torch::Tensor input_OptField_fft = torch::fft::fft2(input);
    torch::Tensor input_opt_field_fftshift = torch::fft::fftshift(input_OptField_fft);
    input_opt_field_fftshift = input_opt_field_fftshift * filter;
    torch::Tensor opt_field_ifftshift = torch::fft::ifftshift(input_opt_field_fftshift);
    torch::Tensor input_opt_field_ifft = torch::fft::ifft2(input_opt_field_fftshift);

    return input_opt_field_ifft;
}

// 暂时没有使用这个函数，因为在复现时有一些问题。
std::vector<torch::Tensor> GroundTruth_Generate::dynamicMedia_generator(const torch::Tensor &filter) const {
    // 按照公式(3)计算动态介质，目前只考虑时间
    std::vector<torch::Tensor> dynamic_media;

    // 生成随机振幅
    for (int i = 0; i < this->T; i++) {
        // 生成随机复振幅
        torch::Tensor random_wave_real_part = torch::rand({this->rows, this->cols});
        torch::Tensor random_wave_imag_part = torch::rand({this->rows, this->cols});
        torch::Tensor random_wave = complex(random_wave_real_part, random_wave_imag_part);

        // 对复振幅做滤波处理，其实复振幅的本质就是滤波后的随机复振幅
        torch::Tensor current = torch::fft::fft2(random_wave);
        current = torch::fft::fftshift(current);
        current = current * filter;
        current = torch::fft::ifftshift(current);
        current = torch::fft::ifft2(current);

        dynamic_media.push_back(current);
    }

    return dynamic_media;
}

// 生成光学掩膜
std::vector<torch::Tensor> GroundTruth_Generate::Mask_generator() const {
    std::vector<torch::Tensor> masks; // 临时储存光学掩膜的矩阵

    // 遍历生成指定数量的掩膜
    for (int mm = 0; mm < this->mask_patterns; mm++) {
        torch::Tensor current_mask = torch::rand({this->rows, this->cols});// 生成随机矩阵，值在 [0, 1) 范围内
        torch::Tensor binary_mask = current_mask < this->transmittance;// current_mask < this->transmittance 会生成一个布尔类型的张量
        masks.push_back(binary_mask.to(torch::kByte));
    }

    return masks;
}
