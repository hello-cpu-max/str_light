cmake_minimum_required(VERSION 3.14)
project(str_light_gray)
add_compile_options("/utf-8")

# 新增文件用到 <filesystem>，需要 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/opencv/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 先创建目标
#add_executable(str_light_gray gray.cpp)
#add_executable(str_light_gray_pro )
# 新增：格雷码 + 正弦相移 投影图生成与解码（格雷码用全局阈值解码）
#add_executable(str_light_gray_ps gray_ps_decode.cpp)
# 新增：实拍格雷码解码验证（全局阈值 + 阶次台阶状检查）
# add_executable(gray_code_verify 
# gray_code_verify.cpp
# gray_pro.cpp
# main_g&v.cpp)
add_executable(cam_proj_calib 
cam_proj_calib.cpp
#reconstruct_pointcloud.cpp
gray_code_verify.cpp
gray_pro.cpp
reconstruct_pointcloud.cpp
#main_g&v.cpp
)
# 新增：正弦相移条纹生成器（周期可指定）
#add_executable(sine_fringe_gen sine_fringe_gen.cpp)

# 再链接库
#target_link_libraries(str_light_gray ${OpenCV_LIBS})
#target_link_libraries(str_light_gray_pro ${OpenCV_LIBS})
#target_link_libraries(str_light_gray_ps ${OpenCV_LIBS})
# target_link_libraries(gray_code_verify ${OpenCV_LIBS})
#target_link_libraries(sine_fringe_gen ${OpenCV_LIBS})
target_link_libraries(cam_proj_calib  ${OpenCV_LIBS})