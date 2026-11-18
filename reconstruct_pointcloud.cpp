// reconstruct_pointcloud.cpp
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include "gen&verify_gray.h"

namespace fs = std::filesystem;
using namespace std;

// 双线性插值读取float索引图
// float getSubPixValue(const cv::Mat& img, float x, float y)
// {
//     int x0 = static_cast<int>(floor(x));
//     int y0 = static_cast<int>(floor(y));
//     int x1 = x0 + 1;
//     int y1 = y0 + 1;
//     if (x0 < 0 || x1 >= img.cols || y0 < 0 || y1 >= img.rows)
//         return -1.0f;
//     float a = x - x0;
//     float b = y - y0;
//     float v00 = img.at<float>(y0, x0);
//     float v01 = img.at<float>(y0, x1);
//     float v10 = img.at<float>(y1, x0);
//     float v11 = img.at<float>(y1, x1);
//     return (1 - a) * (1 - b) * v00 + a * (1 - b) * v01 + (1 - a) * b * v10 + a * b * v11;
// }

// 格雷码解码函数，和标定代码保持完全一致
// cv::Mat decodeGrayCode(const std::vector<cv::Mat>& imgs,
//                        const std::vector<cv::Mat>& imgs_inv,
//                        const std::vector<int>& gray2idx,
//                        int BIT_NUM,
//                        bool USE_INV_PATTERN,
//                        int GLOBAL_THRESHOLD,
//                        int H, int W)
// {
//     cv::Mat idx_map(H, W, CV_32F, cv::Scalar(-1));
//     const int TH_HIGH = 8;
//     const int TH_LOW  = -8;
//     for (int y = 0; y < H; y++)
//     {
//         for (int x = 0; x < W; x++)
//         {
//             uint16_t gray_val = 0;
//             bool valid = true;
//             for (int b = 0; b < BIT_NUM; b++)
//             {
//                 uint8_t pix = imgs[b].at<uint8_t>(y, x);
//                 bool bit;
//                 if (USE_INV_PATTERN)
//                 {
//                     uint8_t pix_inv = imgs_inv[b].at<uint8_t>(y, x);
//                     int diff = pix - pix_inv;
//                     if (diff > TH_HIGH)
//                         bit = true;
//                     else if (diff < TH_LOW)
//                         bit = false;
//                     else
//                     {
//                         valid = false;
//                         break;
//                     }
//                 }
//                 else
//                 {
//                     bit = (pix > GLOBAL_THRESHOLD);
//                 }
//                 if (bit)
//                     gray_val |= (1 << b);
//             }
//             int code_idx = -1;
//             if (valid)
//             {
//                 if (gray_val < static_cast<uint16_t>(gray2idx.size()))
//                 {
//                     code_idx = gray2idx[gray_val];
//                 }
//             }
//             idx_map.at<float>(y, x) = static_cast<float>(code_idx);
//         }
//     }
//     return idx_map;
// }

// 读取文件夹灰度图，排序
bool loadImagesFromDir(const string& dir, vector<cv::Mat>& out_imgs)
{
    out_imgs.clear();
    vector<fs::path> paths;
    for (auto& entry : fs::directory_iterator(dir))
    {
        if (entry.is_regular_file())
            paths.push_back(entry.path());
    }
    sort(paths.begin(), paths.end());
    for (auto& p : paths)
    {
        cv::Mat img = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cerr << "读取失败：" << p << endl;
            continue;
        }
        out_imgs.push_back(img);
    }
    if (out_imgs.empty())
        return false;
    cout << dir << " 加载：" << out_imgs.size() << " 张\n";
    return true;
}

// 保存PLY点云（ASCII格式，MeshLab/CloudCompare直接打开）
void savePointCloudPLY(const string& ply_path, const vector<cv::Point3f>& points)
{
    ofstream f(ply_path);
    if (!f.is_open())
    {
        cerr << "无法写入PLY文件：" << ply_path << endl;
        return;
    }
    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "element vertex " << points.size() << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    f << "end_header\n";
    for (auto& p : points)
    {
        f << p.x << " " << p.y << " " << p.z << "\n";
    }
    f.close();
    cout << "点云保存成功，点数：" << points.size() << " -> " << ply_path << "\n";
}

int main()
{
    // ===================== 参数区（和标定代码保持一致） =====================
    const string yaml_path = "D:/fpp_project/str_light_gray/calib_result.yaml";
    const string ply_out = "D:/fpp_project/str_light_gray/recon.ply";

    const int BIT_NUM = 10;
    const bool USE_INV_PATTERN = true;
    const int GLOBAL_THRESHOLD = 60;

    // 本次重建的格雷码图像目录（一组重建的条纹）
    string gray_lev_dir    = "D:/fpp_project/str_light_gray/recon_data/gray_lev";
    string gray_lev_inv_dir= "D:/fpp_project/str_light_gray/recon_data/gray_lev_inv";
    string gray_vert_dir   = "D:/fpp_project/str_light_gray/recon_data/gray_vert";
    string gray_vert_inv_dir="D:/fpp_project/str_light_gray/recon_data/gray_vert_inv";
    // =======================================================================

    // 1. 读取标定yaml
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "无法打开标定yaml！" << yaml_path << endl;
        return -1;
    }
    cv::Mat Kc, Dc, Kp, Dp, R, T;
    fs["camera_matrix"] >> Kc;
    fs["dist_coeffs"] >> Dc;
    fs["proj_matrix"] >> Kp;
    fs["proj_dist_coeffs"] >> Dp;
    fs["R"] >> R;
    fs["T"] >> T;
    fs.release();
    cout << "标定参数加载完成\n";

    // 2. 加载一组重建用格雷码
    vector<cv::Mat> lev, lev_inv, vert, vert_inv;
    if (!loadImagesFromDir(gray_lev_dir, lev) || !loadImagesFromDir(gray_lev_inv_dir, lev_inv)
        || !loadImagesFromDir(gray_vert_dir, vert) || !loadImagesFromDir(gray_vert_inv_dir, vert_inv))
    {
        cerr << "重建格雷码图像加载失败\n";
        return -1;
    }
    int H = lev[0].rows;
    int W = lev[0].cols;

    // =========【重点】gray2idx 和标定代码必须完全相同！！=========
    vector<int> gray2idx;
    gray2idx.resize(1 << BIT_NUM, -1);
    // 这里填入你生成格雷码的映射，和标定程序同一份
    // for(int i=0; i<(1<<BIT_NUM); i++) gray2idx[gray_code[i]] = i;
    // ===========================================================

    // 3. 解码水平、垂直索引图
    cv::Mat idx_lev  = decodeGrayCode(lev, lev_inv, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H, W);
    cv::Mat idx_vert = decodeGrayCode(vert, vert_inv, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H, W);

    // 4. 构造投影矩阵 P1(相机)、P2(投影仪)
    cv::Mat P1 = cv::Mat::eye(3,4,CV_64F);
    cv::Mat P2;
    cv::hconcat(R, T, P2);
    P1 = Kc * P1;
    P2 = Kp * P2;

    // 5. 遍历所有像素，有效点三角化
    vector<cv::Point3f> pointcloud;
    const float min_z = 10.0f;   // 过滤太近的噪声(mm)
    const float max_z = 1500.0f; // 过滤太远的噪声(mm)
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            float u_proj = idx_lev.at<float>(y, x);
            float v_proj = idx_vert.at<float>(y, x);
            if (u_proj < 0 || v_proj < 0)
                continue;

            // 相机像素点
            cv::Point2f pt_cam((float)x, (float)y);
            // 投影仪像素点
            cv::Point2f pt_proj(u_proj, v_proj);

            // 去畸变（非常关键！不去畸变误差巨大）
            vector<cv::Point2f> pt_cam_vec = {pt_cam};
            vector<cv::Point2f> pt_proj_vec = {pt_proj};
            cv::undistortPoints(pt_cam_vec, pt_cam_vec, Kc, Dc, cv::Mat(), Kc);
            cv::undistortPoints(pt_proj_vec, pt_proj_vec, Kp, Dp, cv::Mat(), Kp);

            cv::Mat pts4d;
            cv::triangulatePoints(P1, P2, pt_cam_vec, pt_proj_vec, pts4d);
            if (fabs(pts4d.at<float>(3,0)) < 1e-6)
                continue;
            float scale = 1.0f / pts4d.at<float>(3,0);
            cv::Point3f pw;
            pw.x = pts4d.at<float>(0,0) * scale;
            pw.y = pts4d.at<float>(1,0) * scale;
            pw.z = pts4d.at<float>(2,0) * scale;

            // 简单深度过滤
            if (pw.z > min_z && pw.z < max_z)
            {
                pointcloud.push_back(pw);
            }
        }
    }

    cout << "有效重建点数量：" << pointcloud.size() << "\n";
    savePointCloudPLY(ply_out, pointcloud);

    return 0;
}
