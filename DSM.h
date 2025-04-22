#ifndef DSM_H
#define DSM_H

#include "GroundTruth_Generate.h"
#include <torch/torch.h>

class DSM : public GroundTruth_Generate {
public:
    torch::Tensor O;  // 需要复现的光场
    torch::Tensor predfilter; // 需要复现的滤波器

    // 训练时需要的参数列表
    std::vector<torch::Tensor> parameters() {
        return {O, predfilter};
    }

    // 构造函数
    DSM(const std::string &_filename, int _layerNum, float _layerDistance)
        : GroundTruth_Generate(_filename, _layerNum, _layerDistance) {
        torch::Tensor O_real_part = torch::ones_like(image_data);
        torch::Tensor O_imag_part = torch::ones_like(image_data);

        O = complex(O_real_part, O_imag_part).requires_grad_(true);// 需要计算梯度
        predfilter = torch::randn_like(image_data).requires_grad_(true);


    }

    std::vector<torch::Tensor> netPredict() const;  // 网络预测器，也即forward方法

    torch::Tensor loss() const; // loss函数
};

#endif // DSM_H
