//
// Created by 余乐 on 25-4-19.
//

#ifndef GROUNDTRUTH_GENERATE_H
#define GROUNDTRUTH_GENERATE_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


class GroundTruth_Generate {
public:
    // 物理参数
    static constexpr float pixel_size = 36e-6; // 像素宽度，单位m
    static constexpr float lambda = 532.8e-9; // 照明波长，单位m
    static constexpr float k = 2.0 * M_PI / lambda; // 波矢，注意波长单位是m

    int radius = 200; // 低通滤波器半径
    float sigma = 80.0f; // 高斯滤波
    torch::Tensor spatialFreqFilter; // 空间滤波器

    int T = 10; //数据采集时刻
    std::vector<torch::Tensor> dynamicMedia; // 动态介质

    int mask_patterns = 10; // 掩膜个数
    float transmittance = 0.8; // 透射率，目前像素比较高，360*360，应该用更加高的透过率训练效果才好
    std::vector<torch::Tensor> Mask; // 文章当中用到的mask

    int layerNum; // 层数
    float layerDistance; // 单位m

    // 图片数据
    std::string filename; // 文件名
    torch::Tensor image_data; // 图片数据(输入光场分布)
    int rows; // 图片(输入光场)像素行数
    int cols; // 图片(输入光场)像素列数

    // 输出光场
    std::vector<torch::Tensor> groundtruth; // 大小为mask_patterns * 1，是储存的每个mask pattern对应的光场

    // 构造函数
    GroundTruth_Generate(const std::string &filename, int _layerNum, float _layerDistance);

    // 光场传播
    [[nodiscard]] torch::Tensor Wave_Propagation(const torch::Tensor &input) const;

    // 生成滤波器
    [[nodiscard]] torch::Tensor spatialFreqFilter_generater() const; // 生成低通滤波器
    [[nodiscard]] torch::Tensor spatialFreqFilter_Guass_generater() const; // 生成高斯滤波器

    // 滤波处理，这里代替动态介质
    [[nodiscard]] torch::Tensor filter_process(const torch::Tensor &input, const torch::Tensor &filter) const;

    // 生成动态介质
    [[nodiscard]] std::vector<torch::Tensor> dynamicMedia_generator(const torch::Tensor &filter) const;

    // 生成光学掩膜
    [[nodiscard]] std::vector<torch::Tensor> Mask_generator() const;
};


#endif //GROUNDTRUTH_GENERATE_H
