#pragma once
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <vector>
#include "HEaaN/heaan.hpp"
#include "convtools.hpp"
#include "imageEncode.hpp"

 void Scaletxtreader(vector<double>& kernel, const string filename, const double cnst) {

     string line;
     ifstream input_file(filename);

     while (getline(input_file, line)) {
         double temp = stod(line);
         double temp1 = cnst* temp;
         kernel.push_back(temp1);
     }

     input_file.close();
     return;

 }


void kernel_ptxt(Context context, vector<double>& weight, vector<vector<vector<Plaintext>>>& output, 
u64 level, u64 gap_in, u64 stride, const int out_ch, const int in_ch, const int ker_size, EnDecoder ecd) {


    if (ker_size == 3) {
        #pragma omp parallel for collapse(4)
        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        weightToPtxt(output[i][j][3 * k + l], level, weight[in_ch * 9 * i + 9 * j + 3 * k + l], gap_in, stride, k , l, ecd);
                    }
                }
            }
        }
        return;
    }
    
    if (ker_size == 1) { //output vector is out_ch * in_ch * 1
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {
                weightToPtxt(output[i][j][0], level, weight[in_ch * i + j], gap_in, stride, 1, 1, ecd);
            }
        }
        return;
    }

}


void addBNsummands(Context context, HomEvaluator eval, vector<vector<Ciphertext>>& afterConv, vector<Plaintext>& summands, const int n, const int ch) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ch; ++j) {
            eval.add(afterConv[i][j], summands[j], afterConv[i][j]);
        }
    }

    return;
}



void kernelEncode_first(Context context, string pathmult, string pathsum, vector<vector<vector<Plaintext>>>& weight, vector<Plaintext>& bias, const double cnst, 
u64 level, u64 gap_in, u64 stride, const int out_ch, const int in_ch, const int ker_size, EnDecoder ecd){

    vector<double> temp0;
    Scaletxtreader(temp0, pathmult, cnst);
    
    kernel_ptxt(context, temp0, weight, level, gap_in, stride, out_ch, in_ch, ker_size, ecd);

    vector<double> temp0a;
    Scaletxtreader(temp0a, pathsum, cnst);

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < out_ch; ++i) {
        {
            Message msg(15, temp0a[i]);
            bias[i]=ecd.encode(msg, 4, 0);
        }
    }

    return;
    
}



void kernelEncode(Context context, string pathmult, string pathsum, vector<vector<vector<Plaintext>>>& weight, vector<Plaintext>& bias, const double cnst, 
u64 level, u64 gap_in, u64 stride, const int out_ch, const int in_ch, const int ker_size, EnDecoder ecd){

    vector<double> temp0;
    txtreader(temp0, pathmult);
    
    kernel_ptxt(context, temp0, weight, level, gap_in, stride, out_ch, in_ch, ker_size, ecd);

    vector<double> temp0a;
    Scaletxtreader(temp0a, pathsum, cnst);

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < out_ch; ++i) {
        {
            Message msg(15, temp0a[i]);
            bias[i]=ecd.encode(msg, 4, 0);
        }
    }

    return;
    
}
