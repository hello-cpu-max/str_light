#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string>
#include <cmath>
#include <bitset>
extern std::vector<uint16_t> T10;
extern const int BIT_NUM ;
extern const int GLOBAL_THRESHOLD ;   // 固定全局阈值
extern const int tau ;
extern const int CODE_RANGE ; // 1024
extern const bool USE_INV_PATTERN ;  // true=使用原图+反相图（论文方案）; false=只用原图
extern const std::string IMG_PREFIX ;
extern const std::string IMG_INV_PREFIX ;
extern uint16_t nbit;
extern std::vector<std::vector<int>> enhanced_gray_trans_table;

std::vector<uint16_t> interleave_trans_seq(int a, int b, const std::vector<bool>& P);
std::vector<uint16_t> gen_gray_by_trans(int n, const std::vector<int>& trans_seq);
std::vector<uint16_t> gen_gray_7bit();
std::vector<uint16_t> gen_gray_10bit();
std::vector<uint16_t> interleave_trans_seq(int a, int b, const std::vector<bool>& P);
std::vector<uint16_t> gen_gray_by_trans(int n, const std::vector<int>& trans_seq);
std::vector<cv::Mat> gray_codes_to_patterns(const std::vector<uint16_t>& codes, int height, int n);
uint16_t bin2gray(uint16_t bin) ;
std::vector<uint16_t> build_enhanced_gray_table();
float getSubPixValue(const cv::Mat& img, float x, float y);
cv::Mat decodeGrayCode(const std::vector<cv::Mat>& imgs,
                       const std::vector<cv::Mat>& imgs_inv,
                       const std::vector<int>& gray2idx,
                       int BIT_NUM,
                       bool USE_INV_PATTERN,
                       int GLOBAL_THRESHOLD,
                       int H, int W);

std::vector<int> build_gray2idx(const std::vector<uint16_t>& gray_table);
cv::Mat read_gray_img(const std::string& path);
bool loadImagesFromDir(const std::string& dir, std::vector<cv::Mat>& out_imgs);