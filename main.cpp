#include "gen&verify_gray.h"

using namespace std;
using namespace cv;

int main()
{
    // ========== 这里填入你的转移序列codes ==========
    // codes[0] 占位，不用；codes[1],codes[2]... 是每一步翻转的bit编号
    // vector<uint16_t> codes = { 5,2,0,3,6,1,2,0,5,4,1,3,6,0,2,1,5,
    // 3,0,4,6,1,2,0,5,3,1,2,6,0,4,1,5,3,
    // 0,2,6,1,3,0,5,4,1,2,6,0,3,1,5,2,0,
    // 4,6,1,3,0,5,2,1,3,6,0,4,1,5,2,0,3,
    // 6,1,2,0,5,4,1,3,6,0,2,1,5,3,0,4,6,
    // 1,2,0,5,3,1,2,6,0,4,1,5,3,0,2,6,1,
    // 3,0,5,4,1,2,6,0,3,1,5,2,0,4,6,1,3,
    // 0,5,2,1,3,6,0,4,1};
    vector<Mat> pat;
    int H = 1140;
    if(nbit==7)
    {
        auto code7 = gen_gray_7bit();
       // cout << "7bitsize:" << code7.size() << endl;
        pat = gray_codes_to_patterns(code7, 1140,nbit);
    }
    else {
        
        auto code10 = gen_gray_10bit();
     //   cout << "10bitsize:" << code10.size() << endl;
        pat = gray_codes_to_patterns(code10, 1140, nbit);
    }
    


    for(int b=0,a=0;a<nbit;b++,b++,a++)
{
    imwrite("enh_gray_" + to_string(nbit) + "bit_" + to_string(a) + ".bmp", pat[b]);
    imwrite("enh_gray_inv_" + to_string(nbit) + "bit_" + to_string(a) + ".bmp", pat[b+1]);
}

//*********************************************************************** */
    // Step1：构建增强格雷码映射表
    std::cout << "Building 10bit enhanced gray code table..." << std::endl;
    auto gray_table = build_enhanced_gray_table();
    auto gray2idx = build_gray2idx(gray_table);
    std::cout << "Table built, total codes: " << gray_table.size() << std::endl;

    // Step2：读取全部10组原图、反相图
    std::vector<cv::Mat> imgs(BIT_NUM);
    std::vector<cv::Mat> imgs_inv(BIT_NUM);
    for(int b = 0; b < BIT_NUM; b++)
    {
        std::string folder = "C:/Users/Lenovo/Desktop/gray/";
        std::string forename = "enh_gray_10bit_";
       // std::string name = folder + forename + std::to_string(b) + ".bmp";
        std::string name = folder  + std::to_string(b) + ".bmp";
        imgs[b] = read_gray_img(name);
        if(USE_INV_PATTERN)
        {
            std::string inv_name = IMG_INV_PREFIX + std::to_string(b) + ".png";
            imgs_inv[b] = read_gray_img(inv_name);
        }
    //     // ========== 新增：绘制并显示当前图直方图 ==========
    // cv::Mat hist_b = draw_hist(imgs[b]);
    // std::string win_name = "Hist of gray_" + std::to_string(b);
    // cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    // cv::imshow(win_name, hist_b);
    // // ================================================

    }

    H = imgs[0].rows;
    int W = imgs[0].cols;
    std::cout << "Image size: W=" << W << ", H=" << H << std::endl;

    // Step3：解码输出序号图（float方便可视化）
    cv::Mat idx_map(H, W, CV_32F, cv::Scalar(-1));
    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            uint16_t gray_val = 0;
            for(int b = 0; b < BIT_NUM; b++)
            {
                uint8_t pix = imgs[b].at<uint8_t>(y,x);
                bool bit;
                if(USE_INV_PATTERN)
                {
                    uint8_t pix_inv = imgs_inv[b].at<uint8_t>(y,x);
                    int diff = pix - pix_inv;
                    bit = (diff > GLOBAL_THRESHOLD);
                }
                else
                {
                    bit = (pix > GLOBAL_THRESHOLD);
                }
                if(bit) gray_val |= (1 << b);
            }
            // 查表得到序号，找不到=-1（无效解码点）
            int code_idx = gray2idx[gray_val];
      
            idx_map.at<float>(y,x) = (float)code_idx;
        }
    }

    // Step4：生成彩色渐变图（校验连续性）
    cv::Mat valid_mask = (idx_map >= 0);
    cv::Mat idx_normalized;
    idx_map.copyTo(idx_normalized, valid_mask);
    // 归一化到0~255用于colormap
    cv::normalize(idx_normalized, idx_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1, valid_mask);
    cv::Mat color_result;
    cv::applyColorMap(idx_normalized, color_result, cv::COLORMAP_JET);

    // 把无效点（-1，解码失败）涂黑
    color_result.setTo(cv::Scalar(0,0,0), ~valid_mask);

    cv::Mat jump_mask = cv::Mat::zeros(idx_map.size(), CV_8UC1);
for(int y = 0; y < idx_map.rows; y++)
{
    for(int x = 1; x < idx_map.cols; x++)
    {
        float cur_idx = idx_map.at<float>(y,x);
        float left_idx = idx_map.at<float>(y,x-1);
        // 两个都是有效索引，且序号差不是 ±1
        if(cur_idx >= 0 && left_idx >=0 && fabs(cur_idx - left_idx) != 1.0f)
        {
            jump_mask.at<uint8_t>(y,x) = 255;
        }
    }
}

// // ========== 打印中间一行序列号，校验连续性 ==========
// int mid_y = idx_map.rows / 2;
// std::cout << "\n===== 中间一行 y = " << mid_y << " 的解码序号 =====\n";
// float last_idx = -9999;
// for(int x = 0; x < idx_map.cols; x++)
// {
//     float cur_idx = idx_map.at<float>(mid_y, x);
//     std::cout << "x=" << x << "\tidx=" << cur_idx;
//     if(last_idx > -9990)
//     {
//         float diff = cur_idx - last_idx;
//         if(fabs(diff) == 1.0f)
//         {
//             std::cout << "\tOK";
//         }
//         else
//         {
//             std::cout << "\t【ERROR】diff = " << diff;
//         }
//     }
//     std::cout << "\n";
//     last_idx = cur_idx;
// }
// std::cout << "====================================\n";


cv::imshow("jump_error_mask", jump_mask);


    // Step5：保存结果
    cv::imwrite("decode_gray_index_map.png", idx_normalized);
    cv::imwrite("decode_color_verify.png", color_result);
    std::cout << "Saved: decode_color_verify.png (color gradient for continuity check)" << std::endl;

    // 可视化窗口
    cv::imshow("Color Verify Map (Continuity Check)", color_result);
    cv::imshow("Gray Index Map", idx_normalized);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

