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
        
vector<vector<Ciphertext>> DSB1(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
    Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
    vector<vector<Ciphertext>>& input){

    // DSB 1 - residual flow

    cout << "layer3 DSB1 conv_onebyone ..." << endl;
    timer.start(" * ");
    
    
    string path7 = "/app/HEAAN-ResNet-20/kernel/multiplicands/" + string("layer3_0_downsample_weight_32_16_1_1.txt");
    
    vector<vector<Ciphertext>> ctxt_block4conv_onebyone_out;
    vector<Ciphertext> convtemp0(32, ctxt_init);

    vector<double> kernel_info0;
    txtreader(kernel_info0, path7);

    for (int i=0; i<16; ++i){
        convtemp0 = newConv(context, pack, eval, ecd, 32, 1, 2, 16, 32, input[i], kernel_info0, 1);
        ctxt_block4conv_onebyone_out.push_back(convtemp0);
    }

    kernel_info0.clear();
    kernel_info0.shrink_to_fit();

    // MPP input bundle making
    //cout << "block4MPP1 and BN summand ..." << endl;
    cout << "MPpacking ... \n\n";
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
    
    
    vector<Plaintext> block4conv_onebyone_summands32(32, ptxt_init);
    vector<double> temp7a;
    string path7a = "/app/HEAAN-ResNet-20/kernel/summands/" + string("layer3_0_downsample_bias_32.txt");
    Scaletxtreader(temp7a, path7a, cnst);
    
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 32; ++i) {
        //#pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp7a[i]);
        block4conv_onebyone_summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    
    temp7a.clear();
    temp7a.shrink_to_fit();
    

    addBNsummands(context, eval, ctxt_block4MPP1_out, block4conv_onebyone_summands32, 4, 32);
    timer.end();

    cout << "DONE!\n\n";

    block4conv_onebyone_summands32.clear();
    block4conv_onebyone_summands32.shrink_to_fit();
    
    

    ///////////////////////// Main flow /////////////////////////////////////////


    
    // DSB 1 - downsample
    
    cout << "layer3 DSB1 conv0 ...\n\n";
    timer.start(" * ");
    
    string path8 = "/app/HEAAN-ResNet-20/kernel/multiplicands/" + string("layer3_0_conv1_weight_32_16_3_3.txt");
    
    vector<vector<Ciphertext>> ctxt_block4conv0_out;
    vector<Ciphertext> convtemp1(32, ctxt_init);

    vector<double> kernel_info1;
    txtreader(kernel_info1, path8);

    for (int i=0; i<16; ++i){
        convtemp1 = newConv(context, pack, eval, ecd, 32, 1, 2, 16, 32, input[i], kernel_info1, 3);
        ctxt_block4conv0_out.push_back(convtemp0);
    }
    
    
    input.clear();
    input.shrink_to_fit();

    kernel_info1.clear();
    kernel_info1.shrink_to_fit();

    // MPP input bundle making
    cout << "MPpacking ...\n" << endl;
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
    
    vector<Plaintext> block4conv0summands32(32, ptxt_init);
    vector<double> temp8a;
    string path8a = "/app/HEAAN-ResNet-20/kernel/summands/" + string("block4conv0summands32.txt");
    Scaletxtreader(temp8a, path8a, cnst);
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 32; ++i) {
        //#pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp8a[i]);
        block4conv0summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp8a.clear();
    temp8a.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block4MPP0_out, block4conv0summands32, 4, 32);

    block4conv0summands32.clear();
    block4conv0summands32.shrink_to_fit();
    timer.end();
    
  
    cout << "layer3 DSB1 relu0 ..." << endl;
    timer.start(" layer3 DSB1 relu0 ");
    vector<vector<Ciphertext>> ctxt_block4relu0_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 10][i % 10], ctxt_block4relu0_out[i / 10][i % 10]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 10][10+(i % 10)], ctxt_block4relu0_out[i / 10][10+(i % 10)]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 10][20+(i % 10)], ctxt_block4relu0_out[i / 10][20+(i % 10)]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 8; ++i) {
        #pragma omp parallel num_threads(5)
        {
            ApproxReLU(context, eval, ctxt_block4MPP0_out[i / 2][30+(i %2)], ctxt_block4relu0_out[i / 2][30+ (i% 2)]);
        }
    }


    timer.end();

    ctxt_block4MPP0_out.clear();
    ctxt_block4MPP0_out.shrink_to_fit();

    cout << "DONE!\n" << "\n";
    
    
    // Second convolution

      // DSB 1 - 2
    
    cout << "layer3 DSB1 conv1 ..." << endl;
    timer.start(" * ");
    
    string path9 = "/app/HEAAN-ResNet-20/kernel/multiplicands/" + string("layer3_0_conv2_weight_32_32_3_3.txt");
    vector<vector<Ciphertext>> ctxt_block4conv1_out;
    vector<Ciphertext> convtemp2(32, ctxt_init);

    vector<double> kernel_info2;
    txtreader(kernel_info2, path9);

    for (int i=0; i<16; ++i){
        convtemp2 = newConv(context, pack, eval, ecd, 32, 2, 1, 32, 32, ctxt_block4relu0_out[i], kernel_info2, 3);
        ctxt_block4conv1_out.push_back(convtemp0);
    }

    kernel_info2.clear();
    kernel_info2.shrink_to_fit();

    vector<Plaintext> block4conv1summands32(32, ptxt_init);
    vector<double> temp9a;
    string path9a = "/app/HEAAN-ResNet-20/kernel/summands/" + string("block4conv1summands32.txt");
    Scaletxtreader(temp9a, path9a, cnst);
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 32; ++i) {
        //#pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp9a[i]);
        block4conv1summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp9a.clear();
    temp9a.shrink_to_fit();
    
    addBNsummands(context, eval, ctxt_block4conv1_out, block4conv1summands32, 4, 32);
    timer.end();

    ctxt_block4relu0_out.clear();
    ctxt_block4relu0_out.shrink_to_fit();
    
    block4conv1summands32.clear();
    block4conv1summands32.shrink_to_fit();



    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "layer3 DSB1 add..." << endl;
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
    cout << "layer3 DSB1 relu1 ..." << endl;
    timer.start(" layer3 DSB1 relu1 ");
    vector<vector<Ciphertext>> ctxt_block4relu1_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block4add_out[i / 10][i % 10], ctxt_block4relu1_out[i / 10][i % 10]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block4add_out[i / 10][10+(i % 10)], ctxt_block4relu1_out[i / 10][10+(i % 10)]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block4add_out[i / 10][20+(i % 10)], ctxt_block4relu1_out[i / 10][20+(i % 10)]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 8; ++i) {
        #pragma omp parallel num_threads(5)
        {
            ApproxReLU(context, eval, ctxt_block4add_out[i / 2][30+(i %2)], ctxt_block4relu1_out[i / 2][30+ (i% 2)]);
        }
    }

    

    timer.end();

    ctxt_block4add_out.clear();
    ctxt_block4add_out.shrink_to_fit();
    
    cout << "layer3 Downsampling DONE!" << "\n";

    return ctxt_block4relu1_out;

}



vector<vector<Ciphertext>> DSB2(HEaaNTimer timer, Context context, KeyPack pack, HomEvaluator eval, EnDecoder ecd, 
    Ciphertext& ctxt_init, Plaintext& ptxt_init, double cnst, auto log_slots, 
    vector<vector<Ciphertext>>& input){

    // DSB 2 - residual flow

    cout << "layer4 DSB2 conv_onebyone ..." << endl;
    timer.start(" * ");

    string path7 = "/app/HEAAN-ResNet-20/kernel/multiplicands/" + string("layer4_0_downsample_weight_64_32_1_1.txt");
    
    vector<vector<Ciphertext>> ctxt_block7conv_onebyone_out;
    vector<Ciphertext> convtemp0(64, ctxt_init);

    vector<double> kernel_info0;
    txtreader(kernel_info0, path7);

    for (int i=0; i<4; ++i){
        convtemp0 = newConv(context, pack, eval, ecd, 32, 2, 2, 32, 64, input[i], kernel_info0, 1);
        ctxt_block7conv_onebyone_out.push_back(convtemp0);
    }

    kernel_info0.clear();
    kernel_info0.shrink_to_fit();

    cout << "MPpacking ..." << endl;
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
    

    vector<Plaintext> block7conv_onebyone_summands64(64, ptxt_init);
    vector<double> temp14a;
    string path14a = "/app/HEAAN-ResNet-20/kernel/summands/" + string("layer4_0_downsample_bias_64.txt");
    Scaletxtreader(temp14a, path14a, cnst);

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp14a[i]);
        block7conv_onebyone_summands64[i]=ecd.encode(msg, 4, 0);
    }
    
    temp14a.clear();
    temp14a.shrink_to_fit();
 
    addBNsummands(context, eval, ctxt_block7MPP1_out, block7conv_onebyone_summands64, 1, 64);
    timer.end();

    block7conv_onebyone_summands64.clear();
    block7conv_onebyone_summands64.shrink_to_fit();



    ///////////////////////// Main flow /////////////////////////////////////////


    
    // DSB 2 - 1st conv
    
    cout << "layer4 DSB2 conv0 ..." << endl;
    timer.start(" * ");
    
    string path8 = "/app/HEAAN-ResNet-20/kernel/multiplicands/" + string("layer4_0_conv1_weight_64_64_3_3.txt");
    
    vector<vector<Ciphertext>> ctxt_block7conv0_out;
    vector<Ciphertext> convtemp1(64, ctxt_init);

    vector<double> kernel_info1;
    txtreader(kernel_info1, path8);

    for (int i=0; i<4; ++i){
        convtemp1 = newConv(context, pack, eval, ecd, 32, 1, 2, 16, 32, input[i], kernel_info1, 3);
        ctxt_block7conv0_out.push_back(convtemp0);
    }
    
    input.clear();
    input.shrink_to_fit();

    kernel_info1.clear();
    kernel_info1.shrink_to_fit();

    // MPP input bundle making
    cout << "MPpacking ..." << endl;
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

    vector<Plaintext> block7conv0summands64(64, ptxt_init);
    vector<double> temp15a;
    string path15a = "/app/HEAAN-ResNet-20/kernel/summands/" + string("layer4_0_conv1_bias_64.txt");
    Scaletxtreader(temp15a, path15a, cnst);

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp15a[i]);
        block7conv0summands64[i]=ecd.encode(msg, 4, 0);
    }
    temp15a.clear();
    temp15a.shrink_to_fit();

    
    addBNsummands(context, eval, ctxt_block7MPP0_out, block7conv0summands64, 1, 64);
    timer.end();

    block7conv0summands64.clear();
    block7conv0summands64.shrink_to_fit();


    // AppReLU
    cout << "layer4 DSB2 relu0 ..." << endl;
    timer.start(" layer4 DSB2 relu0 ");
    vector<vector<Ciphertext>> ctxt_block7relu0_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block7MPP0_out[0][i], ctxt_block7relu0_out[0][i]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 40; i < 64; ++i) {
        ApproxReLU(context, eval, ctxt_block7MPP0_out[0][i], ctxt_block7relu0_out[0][i]);
    }


    timer.end();

    ctxt_block7MPP0_out.clear();
    ctxt_block7MPP0_out.shrink_to_fit();

    cout << "DONE!" << "\n";

    
    
    // DSB 1 - 2
    
    cout << "layer4 DSB2 conv1 ..." << endl;
    timer.start(" * ");
    
    string path9 = "/app/HEAAN-ResNet-20/kernel/multiplicands/" + string("layer4_0_conv2_weight_64_64_3_3.txt");
    vector<vector<Ciphertext>> ctxt_block7conv1_out;
    vector<Ciphertext> convtemp2(64, ctxt_init);

    vector<double> kernel_info2;
    txtreader(kernel_info2, path9);

    for (int i=0; i<1; ++i){
        convtemp2 = newConv(context, pack, eval, ecd, 32, 4, 1, 64, 64, ctxt_block7relu0_out[i], kernel_info2, 3);
        ctxt_block7conv1_out.push_back(convtemp2);
    }

    ctxt_block7relu0_out.clear();
    ctxt_block7relu0_out.shrink_to_fit();

    kernel_info2.clear();
    kernel_info2.shrink_to_fit();


    vector<Plaintext> block7conv1summands64(64, ptxt_init);
    vector<double> temp16a;
    string path16a = "/app/HEAAN-ResNet-20/kernel/summands/" + string("layer4_0_conv2_bias_64.txt");
    Scaletxtreader(temp16a, path16a, cnst);

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp16a[i]);
        block7conv1summands64[i]=ecd.encode(msg, 4, 0);
    }

    temp16a.clear();
    temp16a.shrink_to_fit();

    
    addBNsummands(context, eval, ctxt_block7conv1_out, block7conv1summands64, 4, 32);
    timer.end();
    
    block7conv1summands64.clear();
    block7conv1summands64.shrink_to_fit();



    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "layer4 DSB2 add ..." << endl;
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

    cout << "DONE!" << "\n";

    // Last AppReLU
    cout << "layer4 DSB2 relu1 ..." << endl;
    timer.start(" layer4 DSB2 relu1 ");
    vector<vector<Ciphertext>> ctxt_block7relu1_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block7add_out[0][i], ctxt_block7relu1_out[0][i]);
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 40; i < 64; ++i) {
        ApproxReLU(context, eval, ctxt_block7add_out[0][i], ctxt_block7relu1_out[0][i]);
    }

    timer.end();

    ctxt_block7add_out.clear();
    ctxt_block7add_out.shrink_to_fit();

    cout << "downsampling block7 DONE!" << "\n";


    return ctxt_block7relu1_out;

}
