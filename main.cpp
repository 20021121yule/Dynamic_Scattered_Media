#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>  // 用于文件输出

#include "DSM.h"

int main() {
    std::string filename = "/Users/yule/Desktop/test.jpg";
    int layerNum = 4;
    float layerDistance = 0.05f;
    DSM dsmModel(filename, layerNum, layerDistance);

    // Set up optimizer
    torch::optim::Adam optimizer(dsmModel.parameters(), torch::optim::AdamOptions(1e-3));

    // Training loop
    for (int epoch = 0; epoch < 10000; epoch++) {
        optimizer.zero_grad();
        torch::Tensor loss = dsmModel.loss();
        loss.backward();
        optimizer.step();

        dsmModel.predfilter.data() = torch::clamp(dsmModel.predfilter.data(), 0, 1);

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Convert the tensor O to a format suitable for OpenCV display
    torch::Tensor O = torch::abs(dsmModel.O);
    O = O.squeeze();  // Remove any extra dimensions if necessary

    // Convert the tensor O to a format suitable for file output
    std::ofstream O_file("/Users/yule/Desktop/O.txt");
    if (O_file.is_open()) {
        for (int i = 0; i < O.size(0); i++) {
            for (int j = 0; j < O.size(1); j++) {
                O_file << O[i][j].item<float>() << " ";
            }
            O_file << "\n";
        }
        O_file.close();
        std::cout << "O saved to O.txt" << std::endl;
    } else {
        std::cerr << "Unable to open O.txt for writing" << std::endl;
    }

    // Convert predfilter to a format suitable for file output
    torch::Tensor S = dsmModel.predfilter;
    S = S.squeeze();  // Remove any extra dimensions if necessary

    // Convert the tensor S to a format suitable for file output
    std::ofstream S_file("/Users/yule/Desktop/predfilter.txt");
    if (S_file.is_open()) {
        for (int i = 0; i < S.size(0); i++) {
            for (int j = 0; j < S.size(1); j++) {
                S_file << S[i][j].item<float>() << " ";
            }
            S_file << "\n";
        }
        S_file.close();
        std::cout << "predfilter saved to predfilter.txt" << std::endl;
    } else {
        std::cerr << "Unable to open predfilter.txt for writing" << std::endl;
    }

    return 0;
}
