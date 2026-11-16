// #include <iostream>
// #include <vector>
// #include <bitset>
// #include <opencv2/opencv.hpp>

#include "gen&verify_gray.h"

using namespace std;
using namespace cv;

uint16_t nbit=10;
vector<uint16_t> T10;

//vector<uint16_t> interleave_trans_seq(int a, int b, const vector<bool>& P);
//vector<uint16_t> gen_gray_by_trans(int n, const vector<int>& trans_seq);

vector<vector<int>> enhanced_gray_trans_table = {
    {}, // n=0 占位
    {}, // n=1
    {0,1,0,1},          // n=2 原文：0,1,0
    {0,1,0,2,0,1,2},  // n=3
    {0,1,2,3,2,1,0,2,0,3,0,1,3,2,3,1}, // n=4
    {2,0,3,1,2,0,4,1,3,0,2,1,3,0,4,1,
     2,0,3,1,2,0,4,1,3,0,2,1,3,0,4,1}, // n=5 论文原文T
    {0,1,2,3,4,5,0,2,4,1,3,2,0,5,4,2,
     3,1,4,0,3,5,2,4,1,3,2,0,5,4,1,3,
     2,0,4,2,3,1,4,0,2,5,3,4,2,1,0,4,
     3,5,2,4,1} // n=6
};

vector<uint16_t> gen_gray_7bit()
{
    int a=5, b=2;
    // P：false=b'(辅码), true=a'(主码)
    vector<bool> P = {false, true, true,
         true};
    auto T7 = interleave_trans_seq(a,b,P);
  //  return gen_gray_by_trans(7, T7);
  return T7;
}

vector<uint16_t> gen_gray_10bit()
{
    int a=5, b=5;
    // P：false=b'(辅码), true=a'(主码)
    vector<bool> P = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
         true, true, true ,true, true, true, true, true, true, true, true, true, true, true, true, true,
         true};
     T10 = interleave_trans_seq(a,b,P);
   // return gen_gray_by_trans(10, T10);
   return T10;
}

vector<uint16_t> interleave_trans_seq(int a, int b, const vector<bool>& P)
{
    auto Ta = enhanced_gray_trans_table[a];
    auto Tb = enhanced_gray_trans_table[b];
    // B比特偏移a，放到高位
    vector<int> Tb_shift;
    for(auto bit : Tb) Tb_shift.push_back(bit + a);

    vector<uint16_t> Tn;
    int ptrA = 0, ptrB = 0;
    int repeat = (1 << a); // P重复2^a次
    for(int r=0; r < repeat; r++)
    {
        for(auto op : P)
        {
            if(op)
            {
                Tn.push_back(Ta[ptrA % Ta.size()]);
                ptrA ++;
            }
            else
            {
                Tn.push_back(Tb_shift[ptrB % Tb_shift.size()]);
                ptrB ++;
            }
        }
    }
    return Tn;
}

vector<uint16_t> gen_gray_by_trans(int n, const vector<int>& trans_seq)
{
    vector<uint16_t> code_list;
    uint16_t cur_code = 0;

    code_list.push_back(cur_code);

    // 一共需要 2^n -1 次翻转
    int total_step = (1 << n) - 1;
    for(int i=0; i < total_step; i++)
    {
        int flip_bit = trans_seq[i];
        cur_code ^= (1 << flip_bit); // 翻转指定bit
        code_list.push_back(cur_code);
    }
    return code_list;
}

// 【1】根据跳变序列生成图案，x=0固定码字0，全黑
vector<Mat> gray_codes_to_patterns(const vector<uint16_t>& codes, int height, int n)
{
    int w = 912;
    vector<uint16_t> code_at_x(w, 0);
    uint16_t curr = 0;
    code_at_x[0] = curr;   // x=0 初始码字0，全黑
    
    
    for(int x = 0; x < w; x++)
    {
        curr ^= (1U << codes[x % codes.size()]);
        code_at_x[x] = curr;
    }

    for(int x = 0; x < code_at_x.size(); x++)
{
    cout << "x=" << x << " , code = " << code_at_x[x] << " , binary = " << bitset<7>(code_at_x[x]) << endl;
}

    vector<Mat> patterns;

// ========== 竖直格雷码（u_p）==========
for(int bit = 0; bit < n; bit++)
{
    Mat pat(height, w, CV_8UC1, Scalar(0));
    Mat pat_inv(height, w, CV_8UC1, Scalar(0));
    pat.col(0).setTo(0);
    for(int x = 0; x < w-1; x++)
    {
        uint8_t val = ((code_at_x[x+1] >> bit) & 1) ? 255 : 0;
        pat.col(x).setTo(val);
    }
    patterns.push_back(pat);
    for(int x = 0; x < w-1; x++)
    {
        uint8_t val = ((code_at_x[x+1] >> bit) & 1) ? 0 : 255;
        pat_inv.col(x).setTo(val);
    }
    patterns.push_back(pat_inv);
}

// ========== 水平格雷码（v_p）==========
for(int bit = 0; bit < n; bit++)
{
    Mat pat(height, w, CV_8UC1, Scalar(0));
    Mat pat_inv(height, w, CV_8UC1, Scalar(0));
    pat.row(0).setTo(0);
    for(int y = 0; y < height - 1; y++)
    {
        uint8_t val = ((code_at_x[(y+1)%912] >> bit) & 1) ? 255 : 0;
        pat.row(y).setTo(val);
    }
    patterns.push_back(pat);
    for(int y = 0; y < height - 1; y++)
    {
        uint8_t val = ((code_at_x[(y+1)%912] >> bit) & 1) ? 0 : 255;
        pat_inv.row(y).setTo(val);
    }
    patterns.push_back(pat_inv);
}
cout << "x="  << endl;
    return patterns;
}


