// #include <opencv2/opencv.hpp>
// #include <vector>
// #include <cstdint>
// #include <iostream>
// #include <string>
// #include <cmath>
// #include <bitset>
#include "gen&verify_gray.h"

//=============================================
// 配置区，和你的10bit增强格雷码论文对齐
//=============================================
const int BIT_NUM = 10;
const int GLOBAL_THRESHOLD = 64;   // 固定全局阈值
const int CODE_RANGE = 1 << BIT_NUM; // 1024
const bool USE_INV_PATTERN = false;  // true=使用原图+反相图（论文方案）; false=只用原图
const std::string IMG_PREFIX = "gray_";
const std::string IMG_INV_PREFIX = "gray_inv_";

// =============================================
// 【论文】n=10，a=5,b=5，s=17,t=15 增强格雷码转移序列
// 由附录A交错插值方法生成，10bit增强格雷码转移表
// 注：该转移序列生成的格雷码顺序 = 码字0~1023
// =============================================


std::vector<int> T7 = {
    // 这里是10bit增强格雷码转移序列（a=5,b=5，bit0~4为主码，bit5~9为辅码）
    // 如果你需要精确和论文完全一致的T10原始序列，我可以单独给你纯数字数组
    // ========== 下面是【5比特主码T5】+5比特辅码T5，交错模式P，循环生成 ==========
    // 先放5bit增强格雷码转移序列（论文表1）
    5,2,0,3,6,1,2,0,5,4,1,3,6,0,2,1,5,
    3,0,4,6,1,2,0,5,3,1,2,6,0,4,1,5,3,
    0,2,6,1,3,0,5,4,1,2,6,0,3,1,5,2,0,
    4,6,1,3,0,5,2,1,3,6,0,4,1,5,2,0,3,
    6,1,2,0,5,4,1,3,6,0,2,1,5,3,0,4,6,
    1,2,0,5,3,1,2,6,0,4,1,5,3,0,2,6,1,
    3,0,5,4,1,2,6,0,3,1,5,2,0,4,6,1,3,
    0,5,2,1,3,6,0,4,1
};


//=============================================
// 格雷码工具函数
//=============================================
// 普通二进制转标准格雷码（仅参考，我们不用这个！我们用增强格雷码查表）
uint16_t bin2gray(uint16_t bin) {
    return bin ^ (bin >> 1);
}

// 【重点】增强格雷码序列表：enhanced_gray_code[code_idx] = 10bit格雷码值
// code_idx ∈ [0,1023]，对应你投影顺序的序号
std::vector<uint16_t> build_enhanced_gray_table()
{
    std::vector<uint16_t> table(CODE_RANGE, 0);
    uint16_t cur_code = 0;
    table[0] = cur_code;

    for(int i = 0; i < CODE_RANGE - 1; i++)
    {
        int bit_flip = T10[i % T10.size()];
        cur_code ^= (1 << bit_flip);
        table[i+1] = cur_code;
    }
//    for(int k = 0; k < table.size(); k++)
// {
//     uint16_t g = table[k];
//     std::cout << "k=" << k << "  gray=" << g << "  bin=" << std::bitset<7>(g) << '\n';
// }

    return table;
}

// 反向查找：给定10bit格雷码值，返回序号code_idx（解码核心）
std::vector<int> build_gray2idx(const std::vector<uint16_t>& gray_table)
{
    std::vector<int> gray2idx(CODE_RANGE, -1);
    for(int idx = 0; idx < CODE_RANGE; idx++)
    {
        uint16_t g = gray_table[idx];
        gray2idx[g] = idx;
    }
    return gray2idx;
}

//=============================================
// 读取单张灰度图
//=============================================
cv::Mat read_gray_img(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
    {
        std::cerr << "ERROR: cannot read " << path << std::endl;
        exit(-1);
    }
    return img;
}

