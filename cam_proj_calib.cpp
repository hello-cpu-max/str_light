#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include "gen&verify_gray.h"

namespace fs = std::filesystem;
using namespace std;

// 【双线性插值读取索引图亚像素值】
float getSubPixValue(const cv::Mat& img, float x, float y)
{
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x0 < 0 || x1 >= img.cols || y0 < 0 || y1 >= img.rows)
        return -1.0f;
    float a = x - x0;
    float b = y - y0;
    float v00 = img.at<float>(y0, x0);
    float v01 = img.at<float>(y0, x1);
    float v10 = img.at<float>(y1, x0);
    float v11 = img.at<float>(y1, x1);
    return (1 - a) * (1 - b) * v00 + a * (1 - b) * v01 + (1 - a) * b * v10 + a * b * v11;
}

// 格雷码解码函数
cv::Mat decodeGrayCode(const std::vector<cv::Mat>& imgs,
                       const std::vector<cv::Mat>& imgs_inv,
                       const std::vector<int>& gray2idx,
                       int BIT_NUM,
                       bool USE_INV_PATTERN,
                       int GLOBAL_THRESHOLD,
                       int H, int W)
{
    cv::Mat idx_map(H, W, CV_32F, cv::Scalar(-1));
    const int TH_HIGH = 8;
    const int TH_LOW  = -8;
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            uint16_t gray_val = 0;
            bool valid = true;
            for (int b = 0; b < BIT_NUM; b++)
            {
                uint8_t pix = imgs[b].at<uint8_t>(y, x);
                bool bit;
                if (USE_INV_PATTERN)
                {
                    uint8_t pix_inv = imgs_inv[b].at<uint8_t>(y, x);
                    int diff = pix - pix_inv;
                    if (diff > TH_HIGH)
                        bit = true;
                    else if (diff < TH_LOW)
                        bit = false;
                    else
                    {
                        valid = false;
                        break;
                    }
                }
                else
                {
                    bit = (pix > GLOBAL_THRESHOLD);
                }
                if (bit)
                    gray_val |= (1 << b);
            }
            int code_idx = -1;
            if (valid)
            {
                if (gray_val < static_cast<uint16_t>(gray2idx.size()))
                {
                    code_idx = gray2idx[gray_val];
                }
            }
            idx_map.at<float>(y, x) = static_cast<float>(code_idx);
        }
    }
    return idx_map;
}

// 计算重投影RMSE
double calcReprojectionError(
    const vector<vector<cv::Point3f>>& objPoints,
    const vector<vector<cv::Point2f>>& imgPoints,
    const vector<cv::Mat>& rvecs,
    const vector<cv::Mat>& tvecs,
    const cv::Mat& K, const cv::Mat& D)
{
    double totalErr = 0.0;
    int totalPt = 0;
    for (int i = 0; i < objPoints.size(); i++)
    {
        vector<cv::Point2f> projPts;
        cv::projectPoints(objPoints[i], rvecs[i], tvecs[i], K, D, projPts);
        double err = cv::norm(imgPoints[i], projPts, cv::NORM_L2SQR);
        totalErr += err;
        totalPt += (int)projPts.size();
    }
    double rmse = sqrt(totalErr / totalPt);
    return rmse;
}

// 读取文件夹下所有图片，按文件名排序
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
            cerr << "图片读取失败：" << p << endl;
            continue;
        }
        out_imgs.push_back(img);
    }
    if (out_imgs.empty())
        return false;
    cout << dir << " 加载图片数量：" << out_imgs.size() << endl;
    return true;
}

int main()
{
    // ===================== 【参数区，按需修改】 =====================
    cv::Size board_size(11, 8);
    float square_size = 15.0f;        // mm
    const int BIT_NUM = 10;
    const bool USE_INV_PATTERN = true;
    const int GLOBAL_THRESHOLD = 60;

    // 相机分辨率
    cv::Size cam_img_size(2448, 2048);
    // DLP（Digital Micromirror Device，数字微镜器件）投影仪分辨率
    cv::Size proj_img_size(912, 1140);

    string calib_cam_dir = "D:/fpp_project/str_light_gray/calib_img";
    string gray_lev_dir  = "D:/fpp_project/str_light_gray/gray_lev";
    string gray_lev_inv_dir = "D:/fpp_project/str_light_gray/gray_lev_inv";
    string gray_vert_dir = "D:/fpp_project/str_light_gray/gray_vert";
    string gray_vert_inv_dir = "D:/fpp_project/str_light_gray/gray_vert_inv";
    string save_yaml_path = "D:/fpp_project/str_light_gray/calib_result.yaml";
    // =================================================================

    // 1. 生成单张棋盘世界3D点
    vector<cv::Point3f> board_obj;
    for (int y = 0; y < board_size.height; y++)
    {
        for (int x = 0; x < board_size.width; x++)
        {
            board_obj.emplace_back(x * square_size, y * square_size, 0.0f);
        }
    }

    // ========== 步骤1：标定相机内参 ==========
    vector<vector<cv::Point3f>> cam_obj_points;
    vector<vector<cv::Point2f>> cam_img_points;
    vector<fs::path> cam_img_paths;
    for (auto& entry : fs::directory_iterator(calib_cam_dir))
    {
        if (!entry.is_regular_file()) continue;
        cam_img_paths.push_back(entry.path());
    }
    sort(cam_img_paths.begin(), cam_img_paths.end());

    for (auto& p : cam_img_paths)
    {
        cv::Mat img = cv::imread(p.string());
        if (img.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        vector<cv::Point2f> corners;
        bool ok = cv::findChessboardCorners(gray, board_size, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (!ok)
        {
            cout << p.filename() << " 角点检测失败，跳过\n";
            continue;
        }
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        cam_obj_points.push_back(board_obj);
        cam_img_points.push_back(corners);
    }
    if (cam_obj_points.empty())
    {
        cerr << "【相机标定】没有有效标定图！\n";
        return -1;
    }
    cv::Mat Kc, Dc;
    vector<cv::Mat> rvecs_c, tvecs_c;
    double cam_rms = cv::calibrateCamera(cam_obj_points, cam_img_points, cam_img_size, Kc, Dc, rvecs_c, tvecs_c);
    cout << "\n===== 相机标定结果 =====\n";
    cout << "相机RMS：" << cam_rms << endl;
    cout << "Kc:\n" << Kc << endl;
    cout << "Dc:\n" << Dc << endl;

    // ========== 步骤2：加载格雷码图像，解码，收集投影仪标定数据 ==========
    vector<cv::Mat> imgs_lev, imgs_lev_inv;
    vector<cv::Mat> imgs_vert, imgs_vert_inv;
    if (!loadImagesFromDir(gray_lev_dir, imgs_lev)
        || !loadImagesFromDir(gray_lev_inv_dir, imgs_lev_inv)
        || !loadImagesFromDir(gray_vert_dir, imgs_vert)
        || !loadImagesFromDir(gray_vert_inv_dir, imgs_vert_inv))
    {
        cerr << "格雷码图像加载失败！检查路径\n";
        return -1;
    }
    int H = imgs_lev[0].rows;
    int W = imgs_lev[0].cols;

    auto gray_table = build_enhanced_gray_table();
    auto gray2idx = build_gray2idx(gray_table);

    // 【重点】gray2idx 格雷码→十进制映射表，你自己生成格雷码序列后填入


    vector<vector<cv::Point3f>> proj_obj_points;
    vector<vector<cv::Point2f>> proj_img_points;
    vector<vector<cv::Point2f>> stereo_cam_points;

    // 逐姿态解码：每一组格雷码对应棋盘一个姿态
    int frame_cnt = imgs_lev.size() / BIT_NUM;
    for (int f = 0; f < frame_cnt; f++)
    {
        // 取出当前帧对应的 BIT_NUM 张正、反水平图案
        vector<cv::Mat> lev_f, lev_inv_f;
        vector<cv::Mat> vert_f, vert_inv_f;
        for (int b = 0; b < BIT_NUM; b++)
        {
            lev_f.push_back(imgs_lev[f * BIT_NUM + b]);
            lev_inv_f.push_back(imgs_lev_inv[f * BIT_NUM + b]);
            vert_f.push_back(imgs_vert[f * BIT_NUM + b]);
            vert_inv_f.push_back(imgs_vert_inv[f * BIT_NUM + b]);
        }
        cv::Mat idx_lev = decodeGrayCode(lev_f, lev_inv_f, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H, W);
        cv::Mat idx_vert = decodeGrayCode(vert_f, vert_inv_f, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H, W);

        // 读取该姿态对应的相机原图，检测棋盘角点
        cv::Mat cam_gray = cv::imread(cam_img_paths[f].string(), cv::IMREAD_GRAYSCALE);
        vector<cv::Point2f> corners_cam;
        bool found = cv::findChessboardCorners(cam_gray, board_size, corners_cam,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (!found)
        {
            cout << "第" << f << "组格雷码：相机角点检测失败，跳过\n";
            continue;
        }
        cv::cornerSubPix(cam_gray, corners_cam, cv::Size(11,11), cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,30,0.001));

        vector<cv::Point2f> corners_proj;
        bool all_valid = true;
        for (auto& p : corners_cam)
        {
            float u_proj = getSubPixValue(idx_lev, p.x, p.y);
            float v_proj = getSubPixValue(idx_vert, p.x, p.y);
            if (u_proj < 0 || v_proj < 0)
            {
                all_valid = false;
                break;
            }
            corners_proj.emplace_back(u_proj, v_proj);
        }
        if (!all_valid)
        {
            cout << "第" << f << "组格雷码：存在解码无效点，跳过\n";
            continue;
        }
        proj_obj_points.push_back(board_obj);
        proj_img_points.push_back(corners_proj);
        stereo_cam_points.push_back(corners_cam);
    }
    if (proj_obj_points.empty())
    {
        cerr << "【投影仪标定】没有有效解码点组！\n";
        return -1;
    }

    // ========== 步骤3：标定投影仪内参（把投影仪当成相机） ==========
    cv::Mat Kp, Dp;
    vector<cv::Mat> rvecs_p, tvecs_p;
    double proj_rms = cv::calibrateCamera(proj_obj_points, proj_img_points, proj_img_size, Kp, Dp, rvecs_p, tvecs_p);
    cout << "\n===== 投影仪标定结果 =====\n";
    cout << "投影仪RMS：" << proj_rms << endl;
    cout << "Kp:\n" << Kp << endl;
    cout << "Dp:\n" << Dp << endl;

    // ========== 步骤4：stereoCalibrate 固定内参，求解相机-投影仪外参 R T ==========
    cv::Mat R, T, E, F;
    double stereo_rms = cv::stereoCalibrate(
        proj_obj_points,
        stereo_cam_points,
        proj_img_points,
        Kc, Dc,
        Kp, Dp,
        cam_img_size,
        R, T, E, F,
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-6)
    );
    cout << "\n===== 立体标定（相机-投影仪）结果 =====\n";
    cout << "立体标定RMS：" << stereo_rms << endl;
    cout << "旋转矩阵 R：\n" << R << endl;
    cout << "平移向量 T：\n" << T << endl;

    // ========== 步骤5：全部参数写入YAML ==========
    cv::FileStorage fs(save_yaml_path, cv::FileStorage::WRITE);
    fs << "camera_matrix" << Kc;
    fs << "dist_coeffs" << Dc;
    fs << "proj_matrix" << Kp;
    fs << "proj_dist_coeffs" << Dp;
    fs << "R" << R;
    fs << "T" << T;
    fs << "cam_image_size" << cam_img_size;
    fs << "proj_image_size" << proj_img_size;
    fs << "cam_rms" << cam_rms;
    fs << "proj_rms" << proj_rms;
    fs << "stereo_rms" << stereo_rms;
    fs.release();
    cout << "\n全部标定参数保存至：" << save_yaml_path << endl;

    return 0;
}
