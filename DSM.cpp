//
// Created by 余乐 on 25-4-20.
//

#include "DSM.h"

std::vector<torch::Tensor> DSM::netPredict() const {
    std::vector<torch::Tensor> dynamicMediaPred = dynamicMedia_generator(predfilter);
    std::vector<torch::Tensor> pred;

    for (int mm = 0; mm < this->mask_patterns; mm++) {
        torch::Tensor pred_mm = torch::zeros_like(image_data);
        for (int tt = 0; tt < this->T; tt++) {
            torch::Tensor input = this->O;
            for (int i = 0; i < this->layerNum; i++) {
                input = Wave_Propagation(input);
                if (i == 1) {
                    input = filter_process(input, predfilter);
                }
                if (i == 2) {
                    input = input * Mask[mm]; // 在这里是光学掩膜
                }
            }
            pred_mm = pred_mm + input.abs().pow(2);
        }
        pred.push_back(pred_mm / this->T);
    }

    return pred;
}

torch::Tensor DSM::loss() const {
    const std::vector<torch::Tensor> pred = netPredict();
    const std::vector<torch::Tensor> groundTruth = this->groundtruth;

    torch::Tensor loss_matrix = torch::zeros_like(image_data);; // Squared error (MSE)
    for (int mm = 0; mm < this->mask_patterns; mm++) {
        torch::Tensor loss_curr = torch::zeros_like(image_data);
        loss_curr = (pred[mm] - groundTruth[mm]).abs().pow(2);

        loss_matrix = loss_matrix + loss_curr;
    }

    return torch::sqrt(loss_matrix.sum() / this->mask_patterns); // Mean squared error
}
