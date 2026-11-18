#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include "gen&verify_gray.h"

namespace fs = std::filesystem;
using namespace std;

// ===================== 公共工具函数（标定、重建共用） =====================


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

// 你之前的双线性插值采样函数

// ===================== 【标定函数：相机+投影仪联合标定，输出yaml】 =====================
bool runCalibration(
    const string& calib_cam_dir,
    const string& gray_lev_dir,
    const string& gray_lev_inv_dir,
    const string& gray_vert_dir,
    const string& gray_vert_inv_dir,
    const string& save_yaml_path,
    cv::Size board_size,
    float square_size,
    int BIT_NUM,
    bool USE_INV_PATTERN,
    int GLOBAL_THRESHOLD,
    cv::Size cam_img_size,
    cv::Size proj_img_size
)
{
    // 1. 棋盘格世界点（固定）
    vector<cv::Point3f> board_obj;
    for (int y = 0; y < board_size.height; y++)
    {
        for (int x = 0; x < board_size.width; x++)
        {
            board_obj.emplace_back(x * square_size, y * square_size, 0.0f);
        }
    }

    // ========== 相机内参标定 ==========
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
        return false;
    }
    cv::Mat Kc, Dc;
    vector<cv::Mat> rvecs_c, tvecs_c;
    double cam_rms = cv::calibrateCamera(cam_obj_points, cam_img_points, cam_img_size, Kc, Dc, rvecs_c, tvecs_c);
    cout << "\n===== 相机标定结果 =====\n";
    cout << "相机RMS：" << cam_rms << endl;
    cout << "Kc:\n" << Kc << endl;
    cout << "Dc:\n" << Dc << endl;

    // ========== 加载格雷码解码图片 ==========
    vector<cv::Mat> imgs_lev, imgs_lev_inv;
    vector<cv::Mat> imgs_vert, imgs_vert_inv;
    if (!loadImagesFromDir(gray_lev_dir, imgs_lev)
        || !loadImagesFromDir(gray_lev_inv_dir, imgs_lev_inv)
        || !loadImagesFromDir(gray_vert_dir, imgs_vert)
        || !loadImagesFromDir(gray_vert_inv_dir, imgs_vert_inv))
    {
        cerr << "格雷码图像加载失败！检查路径\n";
        return false;
    }
    int H_calib = imgs_lev[0].rows;
    int W_calib = imgs_lev[0].cols;

    auto gray_table = build_enhanced_gray_table();
    auto gray2idx = build_gray2idx(gray_table);

    vector<vector<cv::Point3f>> proj_obj_points;
    vector<vector<cv::Point2f>> proj_img_points;
    vector<vector<cv::Point2f>> stereo_cam_points;

    int frame_cnt = imgs_lev.size() / BIT_NUM;
    // 校验：棋盘姿态数量必须等于格雷码组数量
    if (frame_cnt != cam_img_paths.size())
    {
        cerr << "警告：格雷码组数 和 棋盘标定图数量不一致！" << frame_cnt << " vs " << cam_img_paths.size() << "\n";
    }

    for (int f = 0; f < frame_cnt; f++)
    {
        vector<cv::Mat> lev_f, lev_inv_f;
        vector<cv::Mat> vert_f, vert_inv_f;
        for (int b = 0; b < BIT_NUM; b++)
        {
            lev_f.push_back(imgs_lev[f * BIT_NUM + b]);
            lev_inv_f.push_back(imgs_lev_inv[f * BIT_NUM + b]);
            vert_f.push_back(imgs_vert[f * BIT_NUM + b]);
            vert_inv_f.push_back(imgs_vert_inv[f * BIT_NUM + b]);
        }
        cv::Mat idx_lev = decodeGrayCode(lev_f, lev_inv_f, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H_calib, W_calib);
        cv::Mat idx_vert = decodeGrayCode(vert_f, vert_inv_f, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H_calib, W_calib);

        if (f >= cam_img_paths.size())
        {
            cout << "第" << f << "组格雷码：无对应棋盘图，跳过\n";
            continue;
        }
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
        return false;
    }

    // ========== 投影仪内参标定 ==========
    cv::Mat Kp, Dp;
    vector<cv::Mat> rvecs_p, tvecs_p;
    double proj_rms = cv::calibrateCamera(proj_obj_points, proj_img_points, proj_img_size, Kp, Dp, rvecs_p, tvecs_p);
    cout << "\n===== 投影仪标定结果 =====\n";
    cout << "投影仪RMS：" << proj_rms << endl;
    cout << "Kp:\n" << Kp << endl;
    cout << "Dp:\n" << Dp << endl;

    // ========== 立体标定，求相机-投影仪外参 R T ==========
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

    // ========== 写入yaml ==========
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
    cout << "\n标定参数保存至：" << save_yaml_path << endl;
    return true;
}

// ===================== 【重建函数：加载yaml + 格雷码，输出PLY点云】 =====================
bool runRecon(
    const string& yaml_path,
    const string& gray_lev_dir_re,
    const string& gray_lev_inv_dir_re,
    const string& gray_vert_dir_re,
    const string& gray_vert_inv_dir_re,
    const string& ply_out,
    int BIT_NUM,
    bool USE_INV_PATTERN,
    int GLOBAL_THRESHOLD,
    float min_z,
    float max_z
)
{
    // 1. 读取标定yaml
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "无法打开标定yaml！" << yaml_path << endl;
        return false;
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

    // 2. 加载重建格雷码
    vector<cv::Mat> lev, lev_inv, vert, vert_inv;
    if (!loadImagesFromDir(gray_lev_dir_re, lev) || !loadImagesFromDir(gray_lev_inv_dir_re, lev_inv)
        || !loadImagesFromDir(gray_vert_dir_re, vert) || !loadImagesFromDir(gray_vert_inv_dir_re, vert_inv))
    {
        cerr << "重建格雷码图像加载失败\n";
        return false;
    }
    int H_recon = lev[0].rows;
    int W_recon = lev[0].cols;

    // gray2idx 必须和标定完全一致
    vector<int> gray2idx;
    gray2idx.resize(1 << BIT_NUM, -1);
    // for(int i=0; i<(1<<BIT_NUM); i++) gray2idx[gray_code[i]] = i;

    // 3. 解码水平、垂直索引图
    cv::Mat idx_lev  = decodeGrayCode(lev, lev_inv, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H_recon, W_recon);
    cv::Mat idx_vert = decodeGrayCode(vert, vert_inv, gray2idx, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD, H_recon, W_recon);

    // 4. 投影矩阵 P1相机，P2投影仪
    cv::Mat P1 = cv::Mat::eye(3,4,CV_64F);
    cv::Mat P2;
    cv::hconcat(R, T, P2);
    P1 = Kc * P1;
    P2 = Kp * P2;

    // 5. 遍历像素三角化
    vector<cv::Point3f> pointcloud;
    for (int y = 0; y < H_recon; y++)
    {
        for (int x = 0; x < W_recon; x++)
        {
            float u_proj = idx_lev.at<float>(y, x);
            float v_proj = idx_vert.at<float>(y, x);
            if (u_proj < 0 || v_proj < 0)
                continue;

            cv::Point2f pt_cam((float)x, (float)y);
            cv::Point2f pt_proj(u_proj, v_proj);

            vector<cv::Point2f> pt_cam_vec = {pt_cam};
            vector<cv::Point2f> pt_proj_vec = {pt_proj};
            cv::undistortPoints(pt_cam_vec, pt_cam_vec, Kc, Dc, cv::Mat(), Kc);
            cv::undistortPoints(pt_proj_vec, pt_proj_vec, Kp, Dp, cv::Mat(), Kp);

            cv::Mat pts4d;
            cv::triangulatePoints(P1, P2, pt_cam_vec, pt_proj_vec, pts4d);
            if (fabs(pts4d.at<double>(3,0)) < 1e-6)
                continue;
            double scale = 1.0 / pts4d.at<double>(3,0);
            cv::Point3f pw;
            pw.x = static_cast<float>(pts4d.at<double>(0,0) * scale);
            pw.y = static_cast<float>(pts4d.at<double>(1,0) * scale);
            pw.z = static_cast<float>(pts4d.at<double>(2,0) * scale);

            if (pw.z > min_z && pw.z < max_z)
            {
                pointcloud.push_back(pw);
            }
        }
    }

    cout << "有效重建点数量：" << pointcloud.size() << "\n";
    savePointCloudPLY(ply_out, pointcloud);
    return true;
}

// ===================== 主函数：在这里选择执行标定 / 重建 =====================
int main()
{
    // ========== 【全局共用参数】标定和重建必须一致 ==========
    const int BIT_NUM = 10;
    const bool USE_INV_PATTERN = true;
    const int GLOBAL_THRESHOLD = 60;
    cv::Size board_size(11, 8);
    float square_size = 15.0f;
    cv::Size cam_img_size(2448, 2048);
    cv::Size proj_img_size(912, 1140);

    // ========== 【标定路径】 ==========
    string calib_cam_dir     = "D:/fpp_project/str_light_gray/calib_img";
    string gray_lev_dir      = "D:/fpp_project/str_light_gray/gray_lev";
    string gray_lev_inv_dir  = "D:/fpp_project/str_light_gray/gray_lev_inv";
    string gray_vert_dir     = "D:/fpp_project/str_light_gray/gray_vert";
    string gray_vert_inv_dir = "D:/fpp_project/str_light_gray/gray_vert_inv";
    string save_yaml_path    = "D:/fpp_project/str_light_gray/calib_result.yaml";

    // ========== 【重建路径】 ==========
    string gray_lev_dir_re    = "D:/fpp_project/str_light_gray/recon_data/gray_lev";
    string gray_lev_inv_dir_re= "D:/fpp_project/str_light_gray/recon_data/gray_lev_inv";
    string gray_vert_dir_re   = "D:/fpp_project/str_light_gray/recon_data/gray_vert";
    string gray_vert_inv_dir_re="D:/fpp_project/str_light_gray/recon_data/gray_vert_inv";
    string ply_out            = "D:/fpp_project/str_light_gray/recon.ply";
    float min_z = 10.0f;
    float max_z = 1500.0f;

    // ========== 开关：二选一运行 ==========
    const bool RUN_CALIB = true;
    const bool RUN_RECON = true;

    if(RUN_CALIB)
    {
        bool ok = runCalibration(
            calib_cam_dir, gray_lev_dir, gray_lev_inv_dir, gray_vert_dir, gray_vert_inv_dir,
            save_yaml_path,
            board_size, square_size, BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD,
            cam_img_size, proj_img_size
        );
        if(!ok)
        {
            cerr << "标定失败，终止！" << endl;
            return -1;
        }
    }

    if(RUN_RECON)
    {
        bool ok = runRecon(
            save_yaml_path,
            gray_lev_dir_re, gray_lev_inv_dir_re, gray_vert_dir_re, gray_vert_inv_dir_re,
            ply_out,
            BIT_NUM, USE_INV_PATTERN, GLOBAL_THRESHOLD,
            min_z, max_z
        );
        if(!ok)
        {
            cerr << "重建失败！" << endl;
            return -1;
        }
    }

    return 0;
}
