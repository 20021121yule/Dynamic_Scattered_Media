cmake_minimum_required(VERSION 3.30)
project(DynamicScatteredMedia)


set(Torch_DIR "/Users/yule/Desktop/CLion配置/libtorch/share/cmake/Torch") # 指定 LibTorch 的路径
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4") # 设置 OpenCV 的安装路径


find_package(Torch REQUIRED)# 查找 LibTorch
find_package(OpenCV REQUIRED)# 查找 OpenCV 包

# 添加你的 C++ 源代码
add_executable(DynamicScatteredMedia main.cpp
        GroundTruth_Generate.cpp
        GroundTruth_Generate.h
        DSM.cpp
        DSM.h
)

# 链接 LibTorch 库
target_link_libraries(DynamicScatteredMedia "${TORCH_LIBRARIES}")
# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(DynamicScatteredMedia ${OpenCV_LIBS})

# 设置 C++ 标准
set_property(TARGET DynamicScatteredMedia PROPERTY CXX_STANDARD 17)

