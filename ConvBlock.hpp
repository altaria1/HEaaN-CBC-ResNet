#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

#include "HEaaN/heaan.hpp" 
#include "examples.hpp" 
#include "Conv.hpp"
#include "AppReLU.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "kernelEncode.hpp"
#include "imageEncode.hpp"
#include "AvgpoolFC64.hpp"

namespace {
    using namespace HEaaN;
    using namespace std;
}    
        
vector<vector<Ciphertext>> Conv_first(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
string pathmult, string pathsum, vector<vector<Ciphertext>>& input){
    
    cout << "parameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3(16, vector<vector<Plaintext>>(3, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> block0conv0summands16(16, ptxt_init);

    kernelEncode_first(context, pathmult, pathsum, block0conv0multiplicands16_3_3_3, block0conv0summands16, cnst, 5, 1, 1, 16, 3, 3, ecd);

    // vector<double> temp0;
    // vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3(16, vector<vector<Plaintext>>(3, vector<Plaintext>(9, ptxt_init)));
    // Scaletxtreader(temp0, pathmult, cnst);

    // kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 5, 1, 1, 16, 3, 3, ecd);

    // temp0.clear();
    // temp0.shrink_to_fit();

    // vector<Plaintext> block0conv0summands16(16, ptxt_init);
    // vector<double> temp0a;
    // Scaletxtreader(temp0a, pathsum, cnst);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 16; ++i) {
    //     {
    //         Message msg(log_slots, temp0a[i]);
    //         block0conv0summands16[i]=ecd.encode(msg, 4, 0);
    //     }
    // }
    // temp0a.clear();
    // temp0a.shrink_to_fit();
    
    timer.end();
    
    // Convolution 0
    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block0conv0_out(16, vector<Ciphertext>(16, ctxt_init));
    
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) { 
        {
            ctxt_block0conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 3, 16, input[i], block0conv0multiplicands16_3_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block0conv0_out, block0conv0summands16, 16, 16);
    timer.end();

    input.clear();
    input.shrink_to_fit();

    cout <<"\n";

    block0conv0multiplicands16_3_3_3.clear();
    block0conv0multiplicands16_3_3_3.shrink_to_fit();
    block0conv0summands16.clear();
    block0conv0summands16.shrink_to_fit();
    
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block0relu0_out(16, vector<Ciphertext>(16, ctxt_init)); 
    ApproxReLU_bundle(context, eval, ctxt_block0relu0_out, ctxt_block0relu0_out);
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[i / 5][i % 5], ctxt_block0relu0_out[i / 5][i % 5]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[8+(i /5)][(i%5)], ctxt_block0relu0_out[8+(i /5)][(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[i / 5][5+(i%5)], ctxt_block0relu0_out[i / 5][5+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[8+(i /5)][5+(i%5)], ctxt_block0relu0_out[8+(i /5)][5+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[i / 5][10+(i%5)], ctxt_block0relu0_out[i / 5][10+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[8+(i /5)][10+(i%5)], ctxt_block0relu0_out[8+(i /5)][10+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 16; ++i) {
    //     ApproxReLU(context, eval, ctxt_block0conv0_out[i][15], ctxt_block0relu0_out[i][15]);
    // }


    timer.end();
    
    ctxt_block0conv0_out.clear();
    ctxt_block0conv0_out.shrink_to_fit();

    cout << "DONE!\n";
    
    return ctxt_block0relu0_out;
}




vector<vector<Ciphertext>> RB1(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
string pathmult1, string pathsum1, string pathmult2, string pathsum2,
vector<vector<Ciphertext>>& input){

   // 1st conv
    cout<< "conv0 start!\n parameter encoding ...";
    timer.start("");

    vector<vector<vector<Plaintext>>> kernel1(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> BNsum1(16, ptxt_init);

    kernelEncode(context, pathmult1, pathsum1, kernel1, BNsum1, cnst, 5, 1, 1, 16, 16, 3, ecd);
    // vector<double> temp1;
    // vector<vector<vector<Plaintext>>> kernel1(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp1, pathmult1);
    // kernel_ptxt(context, temp1, kernel1, 5, 1, 1, 16, 16, 3, ecd);
    // temp1.clear();
    // temp1.shrink_to_fit();

    // vector<Plaintext> BNsum1(16, ptxt_init);
    // vector<double> temp1a;
    // Scaletxtreader(temp1a, pathsum1, cnst);
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 16; ++i) {
    //     //#pragma omp parallel num_threads(5)
    //     {
    //     Message msg(log_slots, temp1a[i]);
    //     BNsum1[i]=ecd.encode(msg, 4, 0);
    //     }
    // }
    // temp1a.clear();
    // temp1a.shrink_to_fit();
    
    timer.end();

    ///////////////////////// Main flow /////////////////////////////////////////
    cout << "convolution ...";
    timer.start("");

    vector<vector<Ciphertext>> ctxt_conv0_out(16, vector<Ciphertext>(16, ctxt_init));
    
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) {
        //#pragma omp parallel num_threads(5)
        {
            ctxt_conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, input[i], kernel1);
        }
    }

    addBNsummands(context, eval, ctxt_conv0_out, BNsum1, 16, 16);
    timer.end();

    // dec.decrypt(ctxt_conv0_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    kernel1.clear();
    kernel1.shrink_to_fit();
    BNsum1.clear();
    BNsum1.shrink_to_fit();


    // AppReLU
    cout << "AppReLU ...";
    timer.start("");
    
    vector<vector<Ciphertext>> ctxt_relu0_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_conv0_out, ctxt_relu0_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i / 5][i % 5], ctxt_relu0_out[i / 5][i % 5]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[8+(i /5)][(i%5)], ctxt_relu0_out[8+(i /5)][(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i / 5][5+(i%5)], ctxt_relu0_out[i / 5][5+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[8+(i /5)][5+(i%5)], ctxt_relu0_out[8+(i /5)][5+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i / 5][10+(i%5)], ctxt_relu0_out[i / 5][10+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[8+(i /5)][10+(i%5)], ctxt_relu0_out[8+(i /5)][10+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 16; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i][15], ctxt_relu0_out[i][15]);
    // }

    timer.end();
    
    ctxt_conv0_out.clear();
    ctxt_conv0_out.shrink_to_fit();
    cout << "DONE!\n";
    

    cout<< "conv1 start!\n parameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> kernel2(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> BNsum2(16, ptxt_init);

    kernelEncode(context, pathmult2, pathsum2, kernel2, BNsum2, cnst, 5, 1, 1, 16, 16, 3, ecd);

    // vector<double> temp2;
    // vector<vector<vector<Plaintext>>> kernel2(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp2, pathmult2);
    // kernel_ptxt(context, temp2, kernel2, 5, 1, 1, 16, 16, 3, ecd);
    // temp2.clear();
    // temp2.shrink_to_fit();

    // vector<Plaintext> BNsum2(16, ptxt_init);
    // vector<double> temp2a;
    // Scaletxtreader(temp2a, pathsum2, cnst);
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 16; ++i) {
    //     Message msg(log_slots, temp2a[i]);
    //     BNsum2[i] = ecd.encode(msg, 4, 0);
    // }

    // temp2a.clear();
    // temp2a.shrink_to_fit();
    
    
    timer.end();
    
    // Second convolution
    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv1_out(16, vector<Ciphertext>(16, ctxt_init));
    
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) {
        {
        ctxt_conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_relu0_out[i], kernel2);
        }
    }

    addBNsummands(context, eval, ctxt_conv1_out, BNsum2, 16, 16);

    timer.end();

    ctxt_relu0_out.clear();
    ctxt_relu0_out.shrink_to_fit();
    kernel2.clear();
    kernel2.shrink_to_fit();
    BNsum2.clear();
    BNsum2.shrink_to_fit();
    

    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Add two flows ..." << endl;
    vector<vector<Ciphertext>> ctxt_add_out(16, vector<Ciphertext>(16, ctxt_init));
    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 16; ++i) {
        for (int ch = 0; ch < 16; ++ch) {
            eval.add(ctxt_conv1_out[i][ch], input[i][ch], ctxt_add_out[i][ch]);
        }
    }
    
    input.clear();
    input.shrink_to_fit();
    ctxt_conv1_out.clear();
    ctxt_conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> output(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_add_out, output);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i / 5][i % 5], output[i / 5][i % 5]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[8+(i /5)][(i%5)], output[8+(i /5)][(i%5)]);
    // }


    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i / 5][5+(i%5)], output[i / 5][5+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[8+(i /5)][5+(i%5)], output[8+(i /5)][5+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i / 5][10+(i%5)], output[i / 5][10+(i%5)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[8+(i /5)][10+(i%5)], output[8+(i /5)][10+(i%5)]);
    // }


    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 16; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i][15], output[i][15]);
    // }

    timer.end();
    
    ctxt_add_out.clear();
    ctxt_add_out.shrink_to_fit();
    
    cout << " DONE!\n";

    return output;

}


vector<vector<Ciphertext>> RB2(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
string pathmult1, string pathsum1, string pathmult2, string pathsum2,
vector<vector<Ciphertext>>& input){

    /// 1st conv...

    cout << "conv0 start!\n parameter encoding ...";
    timer.start("");

    vector<vector<vector<Plaintext>>> kernel1(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> BNsum1(32, ptxt_init);

    kernelEncode(context, pathmult1, pathsum1, kernel1, BNsum1, cnst, 5, 2, 1, 32, 32, 3, ecd);

    // vector<double> temp10;
    // vector<vector<vector<Plaintext>>> kernel1(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp10, pathmult1);
    // kernel_ptxt(context, temp10, kernel1, 5, 2, 1, 32, 32, 3, ecd);
    // temp10.clear();
    // temp10.shrink_to_fit();


    // vector<Plaintext> BNsum1(32, ptxt_init);
    // vector<double> temp10a;
    // Scaletxtreader(temp10a, pathsum1, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 32; ++i) {
    //     {
    //     Message msg(log_slots, temp10a[i]);
    //     BNsum1[i]=ecd.encode(msg, 4, 0);
    //     }
    // }

    // temp10a.clear();
    // temp10a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv0_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(10)
        {
            ctxt_conv0_out[i] = Conv(context, pack, eval, 32, 2, 1, 32, 32, input[i], kernel1);
        }
    }

    addBNsummands(context, eval, ctxt_conv0_out, BNsum1, 4, 32);
    timer.end();

    kernel1.clear();
    kernel1.shrink_to_fit();
    BNsum1.clear();
    BNsum1.shrink_to_fit();


    // AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_relu0_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_conv0_out, ctxt_relu0_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i / 10][i % 10], ctxt_relu0_out[i / 10][i % 10]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i / 10][10+(i % 10)], ctxt_relu0_out[i / 10][10+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[i / 10][20+(i % 10)], ctxt_relu0_out[i / 10][20+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 8; ++i) {
    //     #pragma omp parallel num_threads(5)
    //     {
    //         ApproxReLU(context, eval, ctxt_conv0_out[i / 2][30+(i %2)], ctxt_relu0_out[i / 2][30+ (i% 2)]);
    //     }
    // }

    timer.end();
    cout << "DONE!\n";

    ctxt_conv0_out.clear();
    ctxt_conv0_out.shrink_to_fit();

    
    // Second convolution

    cout << "conv1 start!\n parameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> kernel2(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> BNsum2(32, ptxt_init);

    kernelEncode(context, pathmult2, pathsum2, kernel2, BNsum2, cnst, 5, 2, 1, 32, 32, 3, ecd);

    // vector<double> temp11;
    // vector<vector<vector<Plaintext>>> kernel2(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp11, pathmult2);
    // kernel_ptxt(context, temp11, kernel2, 5, 2, 1, 32, 32, 3, ecd);
    // temp11.clear();
    // temp11.shrink_to_fit();


    // vector<Plaintext> BNsum2(32, ptxt_init);
    // vector<double> temp11a;
    // Scaletxtreader(temp11a, pathsum2, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 32; ++i) {
    //     {
    //     Message msg(log_slots, temp11a[i]);
    //     BNsum2[i]=ecd.encode(msg, 4, 0);
    //     }
    // }
    // temp11a.clear();
    // temp11a.shrink_to_fit();
    
    timer.end();


    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv1_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(10)
        {
            ctxt_conv1_out[i] = Conv(context, pack, eval, 32, 2, 1, 32, 32, ctxt_relu0_out[i], kernel2);
        }
    }

    addBNsummands(context, eval, ctxt_conv1_out, BNsum2, 4, 32);
    timer.end();

    ctxt_relu0_out.clear();
    ctxt_relu0_out.shrink_to_fit();
    kernel2.clear();
    kernel2.shrink_to_fit();
    BNsum2.clear();
    BNsum2.shrink_to_fit();
    


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Add two flows ..." << endl;
    vector<vector<Ciphertext>> ctxt_add_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            eval.add(ctxt_conv1_out[i][ch], input[i][ch], ctxt_add_out[i][ch]);
        }
    }
    
    input.clear();
    input.shrink_to_fit();
    ctxt_conv1_out.clear();
    ctxt_conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "AppReLU ... ";
    timer.start("");

    vector<vector<Ciphertext>> output(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_add_out, output);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i / 10][i % 10], output[i / 10][i % 10]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i / 10][10+(i % 10)], output[i / 10][10+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[i / 10][20+(i % 10)], output[i / 10][20+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 8; ++i) {
    //     #pragma omp parallel num_threads(5)
    //     {
    //         ApproxReLU(context, eval, ctxt_add_out[i / 2][30+(i %2)], output[i / 2][30+ (i% 2)]);
    //     }
    // }

    timer.end();

    ctxt_add_out.clear();
    ctxt_add_out.shrink_to_fit();
    
    cout << "DONE!\n";

    return output;
    
}    


vector<vector<Ciphertext>> RB3(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
string pathmult1, string pathsum1, string pathmult2, string pathsum2,
vector<vector<Ciphertext>>& input){


    ///////////////////////// Main flow /////////////////////////////////////////

    cout << "conv0 start!\n parameter encoding ";
    timer.start("");
    
    vector<vector<vector<Plaintext>>> kernel1(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> bias1(64, ptxt_init);

    kernelEncode(context, pathmult1, pathsum1, kernel1, bias1, cnst, 5, 4, 1, 64, 64, 3, ecd);

    // vector<double> temp17;
    // vector<vector<vector<Plaintext>>> kernel1(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp17, pathmult1);
    // kernel_ptxt(context, temp17, kernel1, 5, 4, 1, 64, 64, 3, ecd);
    // temp17.clear();
    // temp17.shrink_to_fit();

    // vector<Plaintext> bias1(64, ptxt_init);
    // vector<double> temp17a;
    // Scaletxtreader(temp17a, pathsum1, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     Message msg(log_slots, temp17a[i]);
    //     bias1[i]=ecd.encode(msg, 4, 0);
    // }
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     Message msg(log_slots, temp17a[i]);
    //     bias1[i]=ecd.encode(msg, 4, 0);
    // }


    // temp17a.clear();
    // temp17a.shrink_to_fit();
    
    timer.end();


    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv0_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) { 
        ctxt_conv0_out[i] = Conv_parallel(context, pack, eval, 32, 4, 1, 64, 64, input[i], kernel1);
    }

    addBNsummands(context, eval, ctxt_conv0_out, bias1, 1, 64);
    timer.end();


    kernel1.clear();
    kernel1.shrink_to_fit();
    bias1.clear();
    bias1.shrink_to_fit();

    // AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_relu0_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_conv0_out, ctxt_relu0_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[0][i], ctxt_relu0_out[0][i]);
    // }
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[0][i], ctxt_relu0_out[0][i]);
    // }

    timer.end();
    cout << "DONE!" << "\n";

    ctxt_conv0_out.clear();
    ctxt_conv0_out.shrink_to_fit();


    // Second convolution

    
    cout << "conv1 start!\n parameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> kernel2(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> bias2(64, ptxt_init);

    kernelEncode(context, pathmult2, pathsum2, kernel2, bias2, cnst, 5, 4, 1, 64, 64, 3, ecd);

    // vector<double> temp18;
    // vector<vector<vector<Plaintext>>> kernel2(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp18, pathmult2);
    // kernel_ptxt(context, temp18, kernel2, 5, 4, 1, 64, 64, 3, ecd);
    // temp18.clear();
    // temp18.shrink_to_fit();

    // vector<Plaintext> bias2(64, ptxt_init);
    // vector<double> temp18a;
    // Scaletxtreader(temp18a, pathsum2, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     Message msg(log_slots, temp18a[i]);
    //     bias2[i]=ecd.encode(msg, 4, 0);
    // }
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     Message msg(log_slots, temp18a[i]);
    //     bias2[i]=ecd.encode(msg, 4, 0);
    // }

    // temp18a.clear();
    // temp18a.shrink_to_fit();
    
    timer.end();


    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv1_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) {
        ctxt_conv1_out[i] = Conv_parallel(context, pack, eval, 32, 4, 1, 64, 64, ctxt_relu0_out[i], kernel2);
    }

    addBNsummands(context, eval, ctxt_conv1_out, bias2, 1, 64);
    timer.end();

    ctxt_relu0_out.clear();
    ctxt_relu0_out.shrink_to_fit();
    kernel2.clear();
    kernel2.shrink_to_fit();
    bias2.clear();
    bias2.shrink_to_fit();
    

    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Add two flows ..." << endl;
    vector<vector<Ciphertext>> ctxt_add_out(1, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            eval.add(ctxt_conv1_out[i][ch], input[i][ch], ctxt_add_out[i][ch]);
        }
    }


    input.clear();
    input.shrink_to_fit();
    ctxt_conv1_out.clear();
    ctxt_conv1_out.shrink_to_fit();


    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> output(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_add_out, output);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[0][i], output[0][i]);
    // }
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     ApproxReLU(context, eval, ctxt_add_out[0][i], output[0][i]);
    // }
    
    timer.end();

    ctxt_add_out.clear();
    ctxt_add_out.shrink_to_fit();

    cout << " DONE!\n";
    
    return output;

}


vector<vector<Ciphertext>> RB3last(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
string pathmult1, string pathsum1, string pathmult2, string pathsum2,
vector<vector<Ciphertext>>& input){


    ///////////////////////// Main flow /////////////////////////////////////////

    cout << "conv0 start!\n parameter encoding ... ";
    timer.start("");
    
    vector<vector<vector<Plaintext>>> kernel1(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> bias1(64, ptxt_init);

    kernelEncode(context, pathmult1, pathsum1, kernel1, bias1, cnst, 5, 4, 1, 64, 64, 3, ecd);

    // vector<double> temp17;
    // vector<vector<vector<Plaintext>>> kernel1(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp17, pathmult1);
    // kernel_ptxt(context, temp17, kernel1, 5, 4, 1, 64, 64, 3, ecd);
    // temp17.clear();
    // temp17.shrink_to_fit();

    // vector<Plaintext> bias1(64, ptxt_init);
    // vector<double> temp17a;
    // Scaletxtreader(temp17a, pathsum1, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     Message msg(log_slots, temp17a[i]);
    //     bias1[i]=ecd.encode(msg, 4, 0);
    // }
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     Message msg(log_slots, temp17a[i]);
    //     bias1[i]=ecd.encode(msg, 4, 0);
    // }


    // temp17a.clear();
    // temp17a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv0_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) { 
        ctxt_conv0_out[i] = Conv_parallel(context, pack, eval, 32, 4, 1, 64, 64, input[i], kernel1);
    }

    addBNsummands(context, eval, ctxt_conv0_out, bias1, 1, 64);
    timer.end();


    kernel1.clear();
    kernel1.shrink_to_fit();
    bias1.clear();
    bias1.shrink_to_fit();

    // AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_relu0_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_conv0_out, ctxt_relu0_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[0][i], ctxt_relu0_out[0][i]);
    // }
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     ApproxReLU(context, eval, ctxt_conv0_out[0][i], ctxt_relu0_out[0][i]);
    // }

    timer.end();
    cout << "DONE!" << "\n";

    ctxt_conv0_out.clear();
    ctxt_conv0_out.shrink_to_fit();



    // Second convolution
    
    cout << "conv1 start!\n parameter encoding ... ";
    timer.start("");
    
    vector<vector<vector<Plaintext>>> kernel2(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> bias2(64, ptxt_init);

    kernelEncode(context, pathmult2, pathsum2, kernel2, bias2, cnst, 5, 4, 1, 64, 64, 3, ecd);

    // vector<double> temp18;
    // vector<vector<vector<Plaintext>>> kernel2(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp18, pathmult2);
    // kernel_ptxt(context, temp18, kernel2, 5, 4, 1, 64, 64, 3, ecd);
    // temp18.clear();
    // temp18.shrink_to_fit();

    // vector<Plaintext> bias2(64, ptxt_init);
    // vector<double> temp18a;
    // Scaletxtreader(temp18a, pathsum2, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     Message msg(log_slots, temp18a[i]);
    //     bias2[i]=ecd.encode(msg, 4, 0);
    // }
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     Message msg(log_slots, temp18a[i]);
    //     bias2[i]=ecd.encode(msg, 4, 0);
    // }

    // temp18a.clear();
    // temp18a.shrink_to_fit();
    
    timer.end();


    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_conv1_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) {
        ctxt_conv1_out[i] = Conv_parallel(context, pack, eval, 32, 4, 1, 64, 64, ctxt_relu0_out[i], kernel2);
    }

    addBNsummands(context, eval, ctxt_conv1_out, bias2, 1, 64);
    timer.end();


    ctxt_relu0_out.clear();
    ctxt_relu0_out.shrink_to_fit();
    kernel2.clear();
    kernel2.shrink_to_fit();
    bias2.clear();
    bias2.shrink_to_fit();
    

    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Add two flows ... " << endl;
    vector<vector<Ciphertext>> ctxt_add_out(1, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            eval.add(ctxt_conv1_out[i][ch], input[i][ch], ctxt_add_out[i][ch]);
        }
    }
    
    input.clear();
    input.shrink_to_fit();
    ctxt_conv1_out.clear();
    ctxt_conv1_out.shrink_to_fit();


    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> output(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_add_out, output);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLUlast(context, eval, ctxt_add_out[0][i], output[0][i]);
    // }
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     ApproxReLUlast(context, eval, ctxt_add_out[0][i], output[0][i]);
    // }
    
    timer.end();

    ctxt_add_out.clear();
    ctxt_add_out.shrink_to_fit();

    cout << " DONE! " << "\n";
    
    return output;

}


vector<vector<Ciphertext>> DSB1(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
    Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
    string pathmult0, string pathsum0, string pathmult1, string pathsum1, string pathmult2, string pathsum2,
    vector<vector<Ciphertext>>& input){
    
    cout << "conv_downsample start!\n parameter encoding ... ";
    timer.start("");
    
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(1, ptxt_init)));
    vector<Plaintext> block4conv_onebyone_summands32(32, ptxt_init);

    kernelEncode(context, pathmult0, pathsum0, block4conv_onebyone_multiplicands32_16_1_1, block4conv_onebyone_summands32, cnst, 5, 1, 2, 32, 16, 1, ecd);

    // vector<double> temp7;
    // vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16__1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(1, ptxt_init)));
    // txtreader(temp7, pathmult0);

    // kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 5, 1, 2, 32, 16, 1, ecd);

    // temp7.clear();
    // temp7.shrink_to_fit();


    // vector<Plaintext> block4conv_onebyone_summands32(32, ptxt_init);
    // vector<double> temp7a;
    // Scaletxtreader(temp7a, pathsum0, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 32; ++i) {
    //     Message msg(log_slots, temp7a[i]);
    //     block4conv_onebyone_summands32[i]=ecd.encode(msg, 4, 0);
    // }1
    // temp7a.clear();
    // temp7a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block4conv_onebyone_out(16, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) {
        {
            ctxt_block4conv_onebyone_out[i] = Conv(context, pack, eval, 32, 1, 2, 16, 32, input[i], block4conv_onebyone_multiplicands32_16_1_1);
        }
    }

    block4conv_onebyone_multiplicands32_16_1_1.clear();
    block4conv_onebyone_multiplicands32_16_1_1.shrink_to_fit();

    // MPP input bundle making
    cout << "Multiplex Parallel packing ... ";
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP1_in(4, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3) num_threads(40)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP1_in[i][ch][k] = ctxt_block4conv_onebyone_out[4 * i + k][ch];
            }
        }
    }
    
    ctxt_block4conv_onebyone_out.clear();
    ctxt_block4conv_onebyone_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block4MPP1_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            ctxt_block4MPP1_out[i][ch] = MPPacking1(context, pack, eval, 32, ctxt_block4MPP1_in[i][ch]);
        }
    }

    ctxt_block4MPP1_in.clear();
    ctxt_block4MPP1_in.shrink_to_fit();
    

    addBNsummands(context, eval, ctxt_block4MPP1_out, block4conv_onebyone_summands32, 4, 32);
    timer.end();


    block4conv_onebyone_summands32.clear();
    block4conv_onebyone_summands32.shrink_to_fit();
    

    ///////////////////////// Main flow /////////////////////////////////////////

    
    cout << "conv0 start!\n parameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> block4conv0summands32(32, ptxt_init);

    kernelEncode(context, pathmult1, pathsum1, block4conv0multiplicands32_16_3_3, block4conv0summands32, cnst, 5, 1, 2, 32, 16, 3, ecd);

    // vector<double> temp8;
    // vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp8, pathmult1);
    // {
    // kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 5, 1, 2, 32, 16, 3, ecd);
    // }
    // temp8.clear();
    // temp8.shrink_to_fit();

    // vector<Plaintext> block4conv0summands32(32, ptxt_init);
    // vector<double> temp8a;
    // Scaletxtreader(temp8a, pathsum1, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 32; ++i) {
    //     {
    //     Message msg(log_slots, temp8a[i]);
    //     block4conv0summands32[i]=ecd.encode(msg, 4, 0);
    //     }
    // }
    
    // temp8a.clear();
    // temp8a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block4conv0_out(16, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) {
        {
            ctxt_block4conv0_out[i] = Conv(context, pack, eval, 32, 1, 2, 16, 32, input[i], block4conv0multiplicands32_16_3_3);
        }
    }

    input.clear();
    input.shrink_to_fit();

    block4conv0multiplicands32_16_3_3.clear();
    block4conv0multiplicands32_16_3_3.shrink_to_fit();

    
    // MPP input bundle making
    cout << "Multiplexed Parallel Packing ... ";
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP0_in(4, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));
    #pragma omp parallel for collapse(3) num_threads(40)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP0_in[i][ch][k] = ctxt_block4conv0_out[4 * i + k][ch];
            }
        }
    }

    ctxt_block4conv0_out.clear();
    ctxt_block4conv0_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block4MPP0_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            ctxt_block4MPP0_out[i][ch] = MPPacking1(context, pack, eval, 32, ctxt_block4MPP0_in[i][ch]);
        }
    }

    ctxt_block4MPP0_in.clear();
    ctxt_block4MPP0_in.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block4MPP0_out, block4conv0summands32, 4, 32);
    timer.end();

    block4conv0summands32.clear();
    block4conv0summands32.shrink_to_fit();

    // AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block4relu0_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_block4MPP0_out, ctxt_block4relu0_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 10][i % 10], ctxt_block4relu0_out[i / 10][i % 10]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 10][10+(i % 10)], ctxt_block4relu0_out[i / 10][10+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 10][20+(i % 10)], ctxt_block4relu0_out[i / 10][20+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 8; ++i) {
    //     #pragma omp parallel num_threads(5)
    //     {
    //         ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 2][30+(i %2)], ctxt_block4relu0_out[i / 2][30+ (i% 2)]);
    //     }
    // }

    timer.end();

    ctxt_block4MPP0_out.clear();
    ctxt_block4MPP0_out.shrink_to_fit();

    cout << "DONE!\n";
    
    
    // Second convolution
    
    cout << "conv1 start!\nparameter encoding ... ";
    timer.start("");
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> block4conv1summands32(32, ptxt_init);

    kernelEncode(context, pathmult2, pathsum2, block4conv1multiplicands32_32_3_3, block4conv1summands32, cnst, 5, 2, 1, 32, 32, 3, ecd);

    // vector<double> temp9;
    // vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp9, pathmult2);
    // {
    // kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    // }
    // temp9.clear();
    // temp9.shrink_to_fit();

    // vector<Plaintext> block4conv1summands32(32, ptxt_init);
    // vector<double> temp9a;
    // Scaletxtreader(temp9a, pathsum2, cnst);
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 32; ++i) {
    //     {
    //     Message msg(log_slots, temp9a[i]);
    //     block4conv1summands32[i]=ecd.encode(msg, 4, 0);
    //     }
    // }
    // temp9a.clear();
    // temp9a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block4conv1_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(10)
        {
            ctxt_block4conv1_out[i] = Conv(context, pack, eval, 32, 2, 1, 32, 32, ctxt_block4relu0_out[i], block4conv1multiplicands32_32_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block4conv1_out, block4conv1summands32, 4, 32);
    timer.end();

    ctxt_block4relu0_out.clear();
    ctxt_block4relu0_out.shrink_to_fit();

    block4conv1multiplicands32_32_3_3.clear();
    block4conv1multiplicands32_32_3_3.shrink_to_fit();
    block4conv1summands32.clear();
    block4conv1summands32.shrink_to_fit();


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Add two flows ... " << endl;
    vector<vector<Ciphertext>> ctxt_block4add_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            eval.add(ctxt_block4conv1_out[i][ch], ctxt_block4MPP1_out[i][ch], ctxt_block4add_out[i][ch]);
        }
    }

    ctxt_block4conv1_out.clear();
    ctxt_block4conv1_out.shrink_to_fit();
    ctxt_block4MPP1_out.clear();
    ctxt_block4MPP1_out.shrink_to_fit();
    

    // Last AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block4relu1_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_block4add_out, ctxt_block4relu1_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block4add_out[i / 10][i % 10], ctxt_block4relu1_out[i / 10][i % 10]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block4add_out[i / 10][10+(i % 10)], ctxt_block4relu1_out[i / 10][10+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block4add_out[i / 10][20+(i % 10)], ctxt_block4relu1_out[i / 10][20+(i % 10)]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 8; ++i) {
    //     #pragma omp parallel num_threads(5)
    //     {
    //         ApproxReLU(context, eval, ctxt_block4add_out[i / 2][30+(i %2)], ctxt_block4relu1_out[i / 2][30+ (i% 2)]);
    //     }
    // }

    
    timer.end();

    ctxt_block4add_out.clear();
    ctxt_block4add_out.shrink_to_fit();
    
    cout << "DONE!\n\n";
    
    return ctxt_block4relu1_out;
    
}


vector<vector<Ciphertext>> DSB2(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
    Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
    string pathmult0, string pathsum0, string pathmult1, string pathsum1, string pathmult2, string pathsum2,
    vector<vector<Ciphertext>>& input){
    
    cout << "conv_downsample start!\nparameter encoding ... ";
    timer.start("");
    
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1(64, vector<vector<Plaintext>>(32, vector<Plaintext>(1, ptxt_init)));
    vector<Plaintext> block7conv_onebyone_summands64(64, ptxt_init);

    kernelEncode(context, pathmult0, pathsum0, block7conv_onebyone_multiplicands64_32_1_1, block7conv_onebyone_summands64, cnst, 5, 2, 2, 64, 32, 1, ecd);
    
    
    // vector<double> temp14;
    // txtreader(temp14, pathmult0);
    // vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1(64, vector<vector<Plaintext>>(32, vector<Plaintext>(1, ptxt_init)));
    // kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 5, 2, 2, 64, 32, 1, ecd);
    
    // temp14.clear();
    // temp14.shrink_to_fit();

    // vector<Plaintext> block7conv_onebyone_summands64(64, ptxt_init);
    
    // vector<double> temp14a;
    // Scaletxtreader(temp14a, pathsum0, cnst);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     Message msg(log_slots, temp14a[i]);
    //     block7conv_onebyone_summands64[i]=ecd.encode(msg, 4, 0);
    // }
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     Message msg(log_slots, temp14a[i]);
    //     block7conv_onebyone_summands64[i]=ecd.encode(msg, 4, 0);
    // }
    
    // temp14a.clear();
    // temp14a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ..." << endl;
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block7conv_onebyone_out(4, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(10)
        {
            ctxt_block7conv_onebyone_out[i] = Conv(context, pack, eval, 32, 2, 2, 32, 64, input[i], block7conv_onebyone_multiplicands64_32_1_1);
        }
    }

    block7conv_onebyone_multiplicands64_32_1_1.clear();
    block7conv_onebyone_multiplicands64_32_1_1.shrink_to_fit();

    cout << "Done!! \n" << "\n";

    // MPP input bundle making
    cout << "Multiplexed Parallel Packing ... ";
    vector<vector<vector<Ciphertext>>> ctxt_block7MPP1_in(1, vector<vector<Ciphertext>>(64, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block7MPP1_in[i][ch][k] = ctxt_block7conv_onebyone_out[4 * i + k][ch];
            }
        }
    }
    
    ctxt_block7conv_onebyone_out.clear();
    ctxt_block7conv_onebyone_out.shrink_to_fit();
    
    // MPP
    vector<vector<Ciphertext>> ctxt_block7MPP1_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            ctxt_block7MPP1_out[i][ch] = MPPacking2(context, pack, eval, 32, ctxt_block7MPP1_in[i][ch]);
        }
    }

    ctxt_block7MPP1_in.clear();
    ctxt_block7MPP1_in.shrink_to_fit();
    
    addBNsummands(context, eval, ctxt_block7MPP1_out, block7conv_onebyone_summands64, 1, 64);
    timer.end();

    block7conv_onebyone_summands64.clear();
    block7conv_onebyone_summands64.shrink_to_fit();

    cout << "DONE!\n\n";
    
    cout << "conv0 start!\nparameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> block7conv0summands64(64, ptxt_init);

    kernelEncode(context, pathmult1, pathsum1, block7conv0multiplicands64_32_3_3, block7conv0summands64, cnst, 5, 2, 2, 64, 32, 3, ecd);

    // vector<double> temp15;
    // vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));

    // txtreader(temp15, pathmult1);
    // kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 5, 2, 2, 64, 32, 3, ecd);
    
    // temp15.clear();
    // temp15.shrink_to_fit();

    // vector<Plaintext> block7conv0summands64(64, ptxt_init);
    // vector<double> temp15a;
    // Scaletxtreader(temp15a, pathsum1, cnst);
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 64; ++i) {
    //     Message msg(log_slots, temp15a[i]);
    //     block7conv0summands64[i]=ecd.encode(msg, 4, 0);
    // }
    // temp15a.clear();
    // temp15a.shrink_to_fit();
    
    timer.end();

    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block7conv0_out(4, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 4; ++i) { // 서로 다른 img
        ctxt_block7conv0_out[i] = Conv_parallel(context, pack, eval, 32, 2, 2, 32, 64, input[i], block7conv0multiplicands64_32_3_3);
    }

    input.clear();
    input.shrink_to_fit();

    block7conv0multiplicands64_32_3_3.clear();
    block7conv0multiplicands64_32_3_3.shrink_to_fit();

    // MPP input bundle making
    cout << "Multiplexed Parallel Packing ... ";
    vector<vector<vector<Ciphertext>>> ctxt_block7MPP0_in(1, vector<vector<Ciphertext>>(64, vector<Ciphertext>(4, ctxt_init)));
    #pragma omp parallel for collapse(3) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block7MPP0_in[i][ch][k] = ctxt_block7conv0_out[4 * i + k][ch];
            }
        }
    }

    ctxt_block7conv0_out.clear();
    ctxt_block7conv0_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block7MPP0_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            ctxt_block7MPP0_out[i][ch] = MPPacking2(context, pack, eval, 32, ctxt_block7MPP0_in[i][ch]);
        }
    }

    ctxt_block7MPP0_in.clear();
    ctxt_block7MPP0_in.shrink_to_fit();
    
    addBNsummands(context, eval, ctxt_block7MPP0_out, block7conv0summands64, 1, 64);
    timer.end();

    block7conv0summands64.clear();
    block7conv0summands64.shrink_to_fit();
    
    // AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block7relu0_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_block7MPP0_out, ctxt_block7relu0_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block7MPP0_out[0][i], ctxt_block7relu0_out[0][i]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     ApproxReLU(context, eval, ctxt_block7MPP0_out[0][i], ctxt_block7relu0_out[0][i]);
    // }

    timer.end();

    ctxt_block7MPP0_out.clear();
    ctxt_block7MPP0_out.shrink_to_fit();

    cout << "DONE!\n\n";
    
    cout << "conv1 start!\nparameter encoding ... ";
    timer.start("");

    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    vector<Plaintext> block7conv1summands64(64, ptxt_init);

    kernelEncode(context, pathmult2, pathsum2, block7conv1multiplicands64_64_3_3, block7conv1summands64, cnst, 5, 4, 1, 64, 64, 3, ecd);

    // vector<double> temp16;
    // vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    // txtreader(temp16, pathmult2);
    
    // {
    //     kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    // }
    
    // temp16.clear();
    // temp16.shrink_to_fit();

    // vector<Plaintext> block7conv1summands64(64, ptxt_init);
    // vector<double> temp16a;
    // Scaletxtreader(temp16a, pathsum2, cnst);
    
    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 64; ++i) {
    //     Message msg(log_slots, temp16a[i]);
    //     block7conv1summands64[i]=ecd.encode(msg, 4, 0);
    // }

    // temp16a.clear();
    // temp16a.shrink_to_fit();
    
    timer.end();
    
    cout << "convolution ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block7conv1_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) {
        ctxt_block7conv1_out[i] = Conv_parallel(context, pack, eval, 32, 4, 1, 64, 64, ctxt_block7relu0_out[i], block7conv1multiplicands64_64_3_3);
    }

    addBNsummands(context, eval, ctxt_block7conv1_out, block7conv1summands64, 1, 64);
    timer.end();

    ctxt_block7relu0_out.clear();
    ctxt_block7relu0_out.shrink_to_fit();

    block7conv1multiplicands64_64_3_3.clear();
    block7conv1multiplicands64_64_3_3.shrink_to_fit();
    block7conv1summands64.clear();
    block7conv1summands64.shrink_to_fit();

    
    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << " Add two flows ..." << endl;
    vector<vector<Ciphertext>> ctxt_block7add_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2) num_threads(40)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            eval.add(ctxt_block7conv1_out[i][ch], ctxt_block7MPP1_out[i][ch], ctxt_block7add_out[i][ch]);
        }
    }

    ctxt_block7conv1_out.clear();
    ctxt_block7conv1_out.shrink_to_fit();
    ctxt_block7MPP1_out.clear();
    ctxt_block7MPP1_out.shrink_to_fit();

    // Last AppReLU
    cout << "AppReLU ... ";
    timer.start("");
    vector<vector<Ciphertext>> ctxt_block7relu1_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle(context, eval, ctxt_block7add_out, ctxt_block7relu1_out);

    // #pragma omp parallel for num_threads(40)
    // for (int i = 0; i < 40; ++i) {
    //     ApproxReLU(context, eval, ctxt_block7add_out[0][i], ctxt_block7relu1_out[0][i]);
    // }

    // #pragma omp parallel for num_threads(40)
    // for (int i = 40; i < 64; ++i) {
    //     ApproxReLU(context, eval, ctxt_block7add_out[0][i], ctxt_block7relu1_out[0][i]);
    // }

    timer.end();

    ctxt_block7add_out.clear();
    ctxt_block7add_out.shrink_to_fit();
    
    cout << "DONE!\n\n";
    
    return ctxt_block7relu1_out;

}
