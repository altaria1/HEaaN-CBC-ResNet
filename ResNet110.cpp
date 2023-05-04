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
#include "convtools.hpp"
#include "kernelEncode.hpp"
#include "imageEncode.hpp"
#include "AvgpoolFC64.hpp"
#include "RB.hpp"
#include "DSB.hpp"

namespace {
    using namespace HEaaN;
    using namespace std;
}

int main() {
    
    /////////////////////////////////////////////
    ///////////////// SetUp /////////////////////////
    /////////////////////////////////////////////////
    
    HEaaNTimer timer(false);
    ParameterPreset preset = ParameterPreset::FGb;
    Context context = makeContext(preset);
    if (!isBootstrappableParameter(context)) {
        return -1;
    }
    
    const auto log_slots = getLogFullSlots(context);

    SecretKey sk(context);
    KeyPack pack(context);

    KeyGenerator keygen(context, sk, pack);

    timer.start("preprocessing ... ");
    
    cout << "Generate encryption key ... " << endl;
    keygen.genEncryptionKey();
    cout << "done" << endl << endl;

    makeBootstrappable(context);

    cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;

    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << endl;
    HomEvaluator eval(context, pack);
    timer.end();

    EnDecoder ecd(context);
    cout.precision(7);
    
    Message msg_zero(log_slots, 0);
    
    Plaintext ptxt_zero(context);
    ptxt_zero = ecd.encode(msg_zero, 5, 0);
    
    Ciphertext ctxt_zero(context);
    enc.encrypt(msg_zero, pack, ctxt_zero, 5, 0);

    double cnst = (double)(1.0 / 40.0);
    Ciphertext ctxt_init(context);
    enc.encrypt(msg_zero, pack, ctxt_init, 0, 0);
    Plaintext ptxt_init(context);
    ptxt_init = ecd.encode(msg_zero, 0, 0);

    

    ////////////////////////////////////////////////////////////
    ///////////// 10000 test image Encoding ///////////////////
    ////////////////////////////////////////////////////////////

    int num;
    
    cout << "10000 test image encoding; choose one of bundle from 1 to 20 \n;
    cin >> num;

    vector<vector<Ciphertext>> imageVec(16, vector<Ciphertext>(3, ctxt_zero));
    
    #pragma omp parallel for
    for (int i = (num-1) * 16; i < num*16; ++i) { // 313
        int ind = i+1;
        string str = "/app/HEAAN-ResNet-110/image/image_" + to_string(ind) + string(".txt");
        vector<double> temp;
        txtreader(temp, str);
        imageCompiler(context, pack, enc, 5, temp, imageVec[(i%16)]);

    }

    cout << "DONE, test for image encode ..." << "\n";

    Message dmsg;
    dec.decrypt(imageVec[0][0], sk, dmsg);
    printMessage(dmsg);

    cout << "DONE\n" << "\n";
    
    
    ////////////////common path//////////////////////////////////
    string common_path_mult = "/app/HEAAN-ResNet-110/resnet110/multiplicands/";
    string common_path_sum = "/app/HEAAN-ResNet-110/resnet110/summands/";
    ////////////////////////////////////////////////////////////
   
    
    //// 0st conv ///
    std::cout << "layer1 convolution .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block6_out;
    string path0 = common_path_mult + string("layer1_weight_16_3_3_3.txt");
    string path0a = common_path_sum + string("layer1_bias_16.txt");
    ctxt_block0relu0_out = Conv_first(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path0, path0a, imageVec);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block0relu0_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    imageVec.clear();
    imageVec.shrink_to_fit();
        
    cout << "block0 DONE!\n" << "\n";


    //////////RB1///////////
    std::cout << "layer2_RB1 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block0_out;
    string path1 = common_path_mult + string("layer2_0_conv1_weight_16_16_3_3.txt");
    string path1a = common_path_sum + string("layer2_0_conv1_bias_16.txt");
    string path1_2 = common_path_mult + string("layer2_0_conv2_weight_16_16_3_3.txt");
    string path1_2a = common_path_sum + string("layer2_0_conv2_bias_16.txt");
    layer2_block0_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path1, path1a, path1_2, path1_2a, ctxt_block0relu0_out);

    ctxt_block0relu0_out.clear();
    ctxt_block0relu0_out.shrink_to_fit();

    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block0_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";

    //////////RB2///////////
    std::cout << "layer2_RB2 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block1_out;
    string path2 = common_path_mult + string("layer2_1_conv1_weight_16_16_3_3.txt");
    string path2a = common_path_sum + string("layer2_1_conv1_bias_16.txt");
    string path2_2 = common_path_mult + string("layer2_1_conv2_weight_16_16_3_3.txt");
    string path2_2a = common_path_sum + string("layer2_1_conv2_bias_16.txt");
    layer2_block1_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path2, path2a, path2_2, path2_2a, layer2_block0_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";

    layer2_block0_out.clear();
    layer2_block0_out.shrink_to_fit();

    //////////RB3///////////
    std::cout << "layer2_RB3 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block2_out;
    string path3 = common_path_mult + string("layer2_2_conv1_weight_16_16_3_3.txt");
    string path3a = common_path_sum + string("layer2_2_conv1_bias_16.txt");
    string path3_2 = common_path_mult + string("layer2_2_conv2_weight_16_16_3_3.txt");
    string path3_2a = common_path_sum + string("layer2_2_conv2_bias_16.txt");
    layer2_block2_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path3, path3a, path3_2, path3_2a, layer2_block1_out);

    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block2_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block1_out.clear();
    layer2_block1_out.shrink_to_fit();

    //////////RB4///////////
    std::cout << "layer2_RB4 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block3_out;
    string path4 = common_path_mult + string("layer2_3_conv1_weight_16_16_3_3.txt");
    string path4a = common_path_sum + string("layer2_3_conv1_bias_16.txt");
    string path4_2 = common_path_mult + string("layer2_3_conv2_weight_16_16_3_3.txt");
    string path4_2a = common_path_sum + string("layer2_3_conv2_bias_16.txt");
    layer2_block3_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path4, path4a, path4_2, path4_2a, layer2_block2_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block3_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block2_out.clear();
    layer2_block2_out.shrink_to_fit();

    //////////RB5///////////
    std::cout << "layer2_RB5 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block4_out;
    string path5 = common_path_mult + string("layer2_4_conv1_weight_16_16_3_3.txt");
    string path5a = common_path_sum + string("layer2_4_conv1_bias_16.txt");
    string path5_2 = common_path_mult + string("layer2_4_conv2_weight_16_16_3_3.txt");
    string path5_2a = common_path_sum + string("layer2_4_conv2_bias_16.txt");
    layer2_block4_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path5, path5a, path5_2, path5_2a, layer2_block3_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block4_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block3_out.clear();
    layer2_block3_out.shrink_to_fit();

    // result save.
    cout << "saving layer2_RB5 ctxt_result_bundel.." << endl;
    string savepath1 = "/app/result/layer2_RB5/";
    saveMsgBundle(dec, sk, layer2_block4_out,savepath1);    
    cout << "done" << endl;


    //////////RB6///////////
    std::cout << "layer2_RB6 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block5_out;
    string path6 = common_path_mult + string("layer2_5_conv1_weight_16_16_3_3.txt");
    string path6a = common_path_sum + string("layer2_5_conv1_bias_16.txt");
    string path6_2 = common_path_mult + string("layer2_5_conv2_weight_16_16_3_3.txt");
    string path6_2a = common_path_sum + string("layer2_5_conv2_bias_16.txt");
    layer2_block5_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path6, path6a, path6_2, path6_2a, layer2_block4_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block5_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block4_out.clear();
    layer2_block4_out.shrink_to_fit();

    //////////RB7///////////
    std::cout << "layer2_RB7 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block6_out;
    string path7 = common_path_mult + string("layer2_6_conv1_weight_16_16_3_3.txt");
    string path7a = common_path_sum + string("layer2_6_conv1_bias_16.txt");
    string path7_2 = common_path_mult + string("layer2_6_conv2_weight_16_16_3_3.txt");
    string path7_2a = common_path_sum + string("layer2_6_conv2_bias_16.txt");
    layer2_block6_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path7, path7a, path7_2, path7_2a, layer2_block5_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block6_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block5_out.clear();
    layer2_block5_out.shrink_to_fit();

    //////////RB8///////////
    std::cout << "layer2_RB8 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block7_out;
    string path8 = common_path_mult + string("layer2_7_conv1_weight_16_16_3_3.txt");
    string path8a = common_path_sum + string("layer2_7_conv1_bias_16.txt");
    string path8_2 = common_path_mult + string("layer2_7_conv2_weight_16_16_3_3.txt");
    string path8_2a = common_path_sum + string("layer2_7_conv2_bias_16.txt");
    layer2_block7_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path8, path8a, path8_2, path8_2a, layer2_block6_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block7_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block6_out.clear();
    layer2_block6_out.shrink_to_fit();

    //////////RB9///////////
    std::cout << "layer2_RB9 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block8_out;
    string path9 = common_path_mult + string("layer2_8_conv1_weight_16_16_3_3.txt");
    string path9a = common_path_sum + string("layer2_8_conv1_bias_16.txt");
    string path9_2 = common_path_mult + string("layer2_8_conv2_weight_16_16_3_3.txt");
    string path9_2a = common_path_sum + string("layer2_8_conv2_bias_16.txt");
    layer2_block8_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path9, path9a, path9_2, path9_2a, layer2_block7_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block8_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block7_out.clear();
    layer2_block7_out.shrink_to_fit();

    //////////RB10///////////
    std::cout << "layer2_RB10 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block9_out;
    string path10 = common_path_mult + string("layer2_9_conv1_weight_16_16_3_3.txt");
    string path10a = common_path_sum + string("layer2_9_conv1_bias_16.txt");
    string path10_2 = common_path_mult + string("layer2_9_conv2_weight_16_16_3_3.txt");
    string path10_2a = common_path_sum + string("layer2_9_conv2_bias_16.txt");
    layer2_block9_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path10, path10a, path10_2, path10_2a, layer2_block8_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block9_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block8_out.clear();
    layer2_block8_out.shrink_to_fit();

     // result save.
    cout << "saving layer2_RB10 ctxt_result_bundel.." << endl;
    string savepath2 = "/app/result/layer2_RB10/";
    saveMsgBundle(dec, sk, layer2_block9_out,savepath2);    
    cout << "done" << endl;


    //////////RB11///////////
    std::cout << "layer2_RB11 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block10_out;
    string path11 = common_path_mult + string("layer2_10_conv1_weight_16_16_3_3.txt");
    string path11a = common_path_sum + string("layer2_10_conv1_bias_16.txt");
    string path11_2 = common_path_mult + string("layer2_10_conv2_weight_16_16_3_3.txt");
    string path11_2a = common_path_sum + string("layer2_10_conv2_bias_16.txt");
    layer2_block10_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path11, path11a, path11_2, path11_2a, layer2_block9_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block10_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block9_out.clear();
    layer2_block9_out.shrink_to_fit();

    //////////RB12///////////
    std::cout << "layer2_RB12 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block11_out;
    string path12 = common_path_mult + string("layer2_11_conv1_weight_16_16_3_3.txt");
    string path12a = common_path_sum + string("layer2_11_conv1_bias_16.txt");
    string path12_2 = common_path_mult + string("layer2_11_conv2_weight_16_16_3_3.txt");
    string path12_2a = common_path_sum + string("layer2_11_conv2_bias_16.txt");
    layer2_block11_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path12, path12a, path12_2, path12_2a, layer2_block10_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block11_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block10_out.clear();
    layer2_block10_out.shrink_to_fit();

    //////////RB13///////////
    std::cout << "layer2_RB13 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block12_out;
    string path13 = common_path_mult + string("layer2_12_conv1_weight_16_16_3_3.txt");
    string path13a = common_path_sum + string("layer2_12_conv1_bias_16.txt");
    string path13_2 = common_path_mult + string("layer2_12_conv2_weight_16_16_3_3.txt");
    string path13_2a = common_path_sum + string("layer2_12_conv2_bias_16.txt");
    layer2_block12_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path13, path13a, path13_2, path13_2a, layer2_block11_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block12_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block11_out.clear();
    layer2_block11_out.shrink_to_fit();

    //////////RB14///////////
    std::cout << "layer2_RB14 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block13_out;
    string path14 = common_path_mult + string("layer2_13_conv1_weight_16_16_3_3.txt");
    string path14a = common_path_sum + string("layer2_13_conv1_bias_16.txt");
    string path14_2 = common_path_mult + string("layer2_13_conv2_weight_16_16_3_3.txt");
    string path14_2a = common_path_sum + string("layer2_13_conv2_bias_16.txt");
    layer2_block13_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path14, path14a, path14_2, path14_2a, layer2_block12_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block13_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block12_out.clear();
    layer2_block12_out.shrink_to_fit();

    //////////RB15///////////
    std::cout << "layer2_RB15 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block14_out;
    string path15 = common_path_mult + string("layer2_14_conv1_weight_16_16_3_3.txt");
    string path15a = common_path_sum + string("layer2_14_conv1_bias_16.txt");
    string path15_2 = common_path_mult + string("layer2_14_conv2_weight_16_16_3_3.txt");
    string path15_2a = common_path_sum + string("layer2_14_conv2_bias_16.txt");
    layer2_block14_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path15, path15a, path15_2, path15_2a, layer2_block13_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block14_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block13_out.clear();
    layer2_block13_out.shrink_to_fit();

     // result save.
    cout << "saving layer2_RB15 ctxt_result_bundel.." << endl;
    string savepath3 = "/app/result/layer2_RB15/";
    saveMsgBundle(dec, sk, layer2_block14_out,savepath3);    
    cout << "done" << endl;   

    //////////RB16///////////
    std::cout << "layer2_RB16 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block15_out;
    string path16 = common_path_mult + string("layer2_15_conv1_weight_16_16_3_3.txt");
    string path16a = common_path_sum + string("layer2_15_conv1_bias_16.txt");
    string path16_2 = common_path_mult + string("layer2_15_conv2_weight_16_16_3_3.txt");
    string path16_2a = common_path_sum + string("layer2_15_conv2_bias_16.txt");
    layer2_block15_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path16, path16a, path16_2, path16_2a, layer2_block14_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block15_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block14_out.clear();
    layer2_block14_out.shrink_to_fit();

    //////////RB17///////////
    std::cout << "layer2_RB17 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block16_out;
    string path17 = common_path_mult + string("layer2_16_conv1_weight_16_16_3_3.txt");
    string path17a = common_path_sum + string("layer2_16_conv1_bias_16.txt");
    string path17_2 = common_path_mult + string("layer2_16_conv2_weight_16_16_3_3.txt");
    string path17_2a = common_path_sum + string("layer2_16_conv2_bias_16.txt");
    layer2_block16_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path17, path17a, path17_2, path17_2a, layer2_block15_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block16_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block15_out.clear();
    layer2_block15_out.shrink_to_fit();

    //////////RB18///////////
    std::cout << "layer2_RB18 .. " << std::endl;
    vector<vector<Ciphertext>> layer2_block17_out;
    string path18 = common_path_mult + string("layer2_17_conv1_weight_16_16_3_3.txt");
    string path18a = common_path_sum + string("layer2_17_conv1_bias_16.txt");
    string path18_2 = common_path_mult + string("layer2_17_conv2_weight_16_16_3_3.txt");
    string path18_2a = common_path_sum + string("layer2_17_conv2_bias_16.txt");
    layer2_block17_out = RB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path18, path18a, path18_2, path18_2a, layer2_block16_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer2_block17_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer2_block16_out.clear();
    layer2_block16_out.shrink_to_fit();

     // result save.
    cout << "saving layer2_RB18 ctxt_result_bundel.." << endl;
    string savepath4 = "/app/result/layer2_RB18/";
    saveMsgBundle(dec, sk, layer2_block17_out,savepath4);    
    cout << "done" << endl;   

    ////////////DSB////////////////

    // DSB 1 - res
    cout << "DSB1 start...!" << endl;
    cout << "uploading for DSB1_downsample_conv0 ...\n";
    timer.start(" * ");
    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(1, ptxt_init)));
    string path19 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("layer3_0_downsample_weight_32_16_1_1.txt");
    txtreader(temp7, path19);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 5, 1, 2, 32, 16, 1, ecd);
    temp7.clear();
    temp7.shrink_to_fit();


    vector<Plaintext> block4conv_onebyone_summands32(32, ptxt_init);
    vector<double> temp7a;
    string path19a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("layer3_0_downsample_bias_32.txt");
    Scaletxtreader(temp7a, path19a, cnst);
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
    
    
    timer.end();

    cout << "DSB1_conv_onebyone ..." << endl;
    timer.start(" DSB1_conv_onebyone .. ");
    cout << "convolution ...\n\n";
    vector<vector<Ciphertext>> ctxt_block4conv_onebyone_out(16, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        //#pragma omp parallel num_threads(5)
        {
            ctxt_block4conv_onebyone_out[i] = Conv(context, pack, eval, 32, 1, 2, 16, 32, layer2_block17_out[i], block4conv_onebyone_multiplicands32_16_1_1);
        }
    }


    block4conv_onebyone_multiplicands32_16_1_1.clear();
    block4conv_onebyone_multiplicands32_16_1_1.shrink_to_fit();


    // MPP input bundle making
    //cout << "block4MPP1 and BN summand ..." << endl;
    cout << "MPPacking ... \n\n";
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP1_in(4, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP1_in[i][ch][k] = ctxt_block4conv_onebyone_out[4 * i + k][ch];
            }
        }
    }
    
    ctxt_block4conv_onebyone_out.clear();
    ctxt_block4conv_onebyone_out.shrink_to_fit();
    
    
    // dec.decrypt(ctxt_block4MPP1_in[0][0][0], sk, dmsg);
    // printMessage(dmsg);


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
    
    // dec.decrypt(ctxt_block4MPP1_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    addBNsummands(context, eval, ctxt_block4MPP1_out, block4conv_onebyone_summands32, 4, 32);
    timer.end();

    cout << "DONE!\n";

    block4conv_onebyone_summands32.clear();
    block4conv_onebyone_summands32.shrink_to_fit();
    
    

    // cout << "Done!! level of ctxt is " << ctxt_block4MPP1_out[0][0].getLevel() << "\n";
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block4MPP1_out[0][0], sk, dmsg);
    // printMessage(dmsg);


    ///////////////////////// Main flow /////////////////////////////////////////


    
    // DSB 1 - 1
    
    cout << "uploading for DSB1_conv0 ...\n\n";
    timer.start(" * ");
    vector<double> temp8;
    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path19_1 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("layer3_0_conv1_weight_32_16_3_3.txt");
    txtreader(temp8, path19_1);
    kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 5, 1, 2, 32, 16, 3, ecd);
    temp8.clear();
    temp8.shrink_to_fit();


    vector<Plaintext> block4conv0summands32(32, ptxt_init);
    vector<double> temp8a;
    string path19_1a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("layer3_0_conv1_bias_32.txt");
    Scaletxtreader(temp8a, path19_1a, cnst);
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
    
    timer.end();

    cout << "DSB1_conv0 ..." << endl;
    timer.start(" DSB1_conv0 ");
    cout <<"convolution ...\n\n";
    vector<vector<Ciphertext>> ctxt_block4conv0_out(16, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        //#pragma omp parallel num_threads(5)
        {
            ctxt_block4conv0_out[i] = Conv(context, pack, eval, 32, 1, 2, 16, 32, layer2_block17_out[i], block4conv0multiplicands32_16_3_3);
        }
    }

    layer2_block17_out.clear();
    layer2_block17_out.shrink_to_fit();

    block4conv0multiplicands32_16_3_3.clear();
    block4conv0multiplicands32_16_3_3.shrink_to_fit();

    // cout << "Done!! level of ctxt is " << ctxt_block4conv0_out[0][0].getLevel() << "\n";
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block4conv0_out[0][0], sk, dmsg);
    // printMessage(dmsg);


    // MPP input bundle making
    cout << "MPPacking ...\n" << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP0_in(4, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP0_in[i][ch][k] = ctxt_block4conv0_out[4 * i + k][ch];
            }
        }
    }
    
    // dec.decrypt(ctxt_block4MPP0_in[0][0][0], sk, dmsg);
    // printMessage(dmsg);

    ctxt_block4conv0_out.clear();
    ctxt_block4conv0_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block4MPP0_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            ctxt_block4MPP0_out[i][ch] = MPPacking1(context, pack, eval, 32, ctxt_block4MPP0_in[i][ch]);
        }
    }

    ctxt_block4MPP0_in.clear();
    ctxt_block4MPP0_in.shrink_to_fit();
    
    // dec.decrypt(ctxt_block4MPP0_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    addBNsummands(context, eval, ctxt_block4MPP0_out, block4conv0summands32, 4, 32);
    timer.end();

    block4conv0summands32.clear();
    block4conv0summands32.shrink_to_fit();

    cout << "Done!! \n" ;
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block4MPP0_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    // ctxt_block4MPP0_out 첫번째 : 서로 다른 img, 두번째 : ch.

    // AppReLU
    cout << "DSB1_relu0 ..." << endl;
    timer.start(" DSB1_relu0 ");
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
    
    

    //dec.decrypt(ctxt_block4relu0_out[0][0], sk, dmsg);
    //printMessage(dmsg);

    // Second convolution

      // DSB 1 - 2
    
    cout << "uploading for DSB1_conv1 ...\n\n";
    timer.start(" * ");
    vector<double> temp9;
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path19_2 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("layer3_0_conv2_weight_32_32_3_3.txt");
    txtreader(temp9, path19_2);
    kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp9.clear();
    temp9.shrink_to_fit();


    vector<Plaintext> block4conv1summands32(32, ptxt_init);
    vector<double> temp9a;
    string path19_2a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("layer3_0_conv2_bias_32.txt");
    Scaletxtreader(temp9a, path19_2a, cnst);
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
    
    timer.end();

    cout << "DSB1_conv1 ..." << endl;
    timer.start(" DSB1_conv1 ");
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

    cout << "Done!!\n" << "\n";
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block4conv1_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    



    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "DSB1_add..." << endl;
    vector<vector<Ciphertext>> ctxt_block4add_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for collapse(2)
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
    cout << "DSB1_relu1 ..." << endl;
    timer.start(" DSB1_relu1 ");
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
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block4relu1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout << "Downsampling block19 DONE!" << "\n";

     // result save.
    cout << "saving DSB1 ctxt_result_bundel.." << endl;
    string savepath5 = "/app/result/DSB1/";
    saveMsgBundle(dec, sk, ctxt_block4relu1_out,savepath5);    
    cout << "done" << endl;   

    //////////RB20///////////////
    std::cout << "layer3_RB1 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block1_out;
    string path20 = common_path_mult + string("layer3_1_conv1_weight_32_32_3_3.txt");
    string path20a = common_path_sum + string("layer3_1_conv1_bias_32.txt");
    string path20_2 = common_path_mult + string("layer3_1_conv2_weight_32_32_3_3.txt");
    string path20_2a = common_path_sum + string("layer3_1_conv2_bias_32.txt");
    layer3_block1_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path20, path20a, path20_2, path20_2a, ctxt_block4relu1_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    ctxt_block4relu1_out.clear();
    ctxt_block4relu1_out.shrink_to_fit();

    //////////RB21///////////////
    std::cout << "layer3_RB2 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block2_out;
    string path21 = common_path_mult + string("layer3_2_conv1_weight_32_32_3_3.txt");
    string path21a = common_path_sum + string("layer3_2_conv1_bias_32.txt");
    string path21_2 = common_path_mult + string("layer3_2_conv2_weight_32_32_3_3.txt");
    string path21_2a = common_path_sum + string("layer3_2_conv2_bias_32.txt");
    layer3_block2_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path21, path21a, path21_2, path21_2a, layer3_block1_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block2_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block1_out.clear();
    layer3_block1_out.shrink_to_fit();

    //////////RB22///////////////
    std::cout << "layer3_RB3 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block3_out;
    string path22 = common_path_mult + string("layer3_3_conv1_weight_32_32_3_3.txt");
    string path22a = common_path_sum + string("layer3_3_conv1_bias_32.txt");
    string path22_2 = common_path_mult + string("layer3_3_conv2_weight_32_32_3_3.txt");
    string path22_2a = common_path_sum + string("layer3_3_conv2_bias_32.txt");
    layer3_block3_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path22, path22a, path22_2, path22_2a, layer3_block2_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block3_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block2_out.clear();
    layer3_block2_out.shrink_to_fit();

    //////////RB23///////////////
    std::cout << "layer3_RB4 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block4_out;
    string path23 = common_path_mult + string("layer3_4_conv1_weight_32_32_3_3.txt");
    string path23a = common_path_sum + string("layer3_4_conv1_bias_32.txt");
    string path23_2 = common_path_mult + string("layer3_4_conv2_weight_32_32_3_3.txt");
    string path23_2a = common_path_sum + string("layer3_4_conv2_bias_32.txt");
    layer3_block4_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path23, path23a, path23_2, path23_2a, layer3_block3_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block4_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block3_out.clear();
    layer3_block3_out.shrink_to_fit();

    //////////RB24///////////////
    std::cout << "layer3_RB5 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block5_out;
    string path24 = common_path_mult + string("layer3_5_conv1_weight_32_32_3_3.txt");
    string path24a = common_path_sum + string("layer3_5_conv1_bias_32.txt");
    string path24_2 = common_path_mult + string("layer3_5_conv2_weight_32_32_3_3.txt");
    string path24_2a = common_path_sum + string("layer3_5_conv2_bias_32.txt");
    layer3_block5_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path24, path24a, path24_2, path24_2a, layer3_block4_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block5_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block4_out.clear();
    layer3_block4_out.shrink_to_fit();

     // result save.
    cout << "saving layer3_RB5 ctxt_result_bundel.." << endl;
    string savepath6 = "/app/result/layer3_RB5/";
    saveMsgBundle(dec, sk, layer3_block5_out,savepath6);    
    cout << "done" << endl;   

    //////////RB25///////////////
    std::cout << "layer3_RB6 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block6_out;
    string path25 = common_path_mult + string("layer3_6_conv1_weight_32_32_3_3.txt");
    string path25a = common_path_sum + string("layer3_6_conv1_bias_32.txt");
    string path25_2 = common_path_mult + string("layer3_6_conv2_weight_32_32_3_3.txt");
    string path25_2a = common_path_sum + string("layer3_6_conv2_bias_32.txt");
    layer3_block6_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path25, path25a, path25_2, path25_2a, layer3_block5_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block6_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block5_out.clear();
    layer3_block5_out.shrink_to_fit();

    //////////RB26///////////////
    std::cout << "layer3_RB7 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block7_out;
    string path26 = common_path_mult + string("layer3_7_conv1_weight_32_32_3_3.txt");
    string path26a = common_path_sum + string("layer3_7_conv1_bias_32.txt");
    string path26_2 = common_path_mult + string("layer3_7_conv2_weight_32_32_3_3.txt");
    string path26_2a = common_path_sum + string("layer3_7_conv2_bias_32.txt");
    layer3_block7_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path26, path26a, path26_2, path26_2a, layer3_block6_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block7_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block6_out.clear();
    layer3_block6_out.shrink_to_fit();

     //////////RB27///////////////
    std::cout << "layer3_RB8 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block8_out;
    string path27 = common_path_mult + string("layer3_8_conv1_weight_32_32_3_3.txt");
    string path27a = common_path_sum + string("layer3_8_conv1_bias_32.txt");
    string path27_2 = common_path_mult + string("layer3_8_conv2_weight_32_32_3_3.txt");
    string path27_2a = common_path_sum + string("layer3_8_conv2_bias_32.txt");
    layer3_block8_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path27, path27a, path27_2, path27_2a, layer3_block7_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block8_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block7_out.clear();
    layer3_block7_out.shrink_to_fit();

     //////////RB28///////////////
    std::cout << "layer3_RB9.. " << std::endl;
    vector<vector<Ciphertext>> layer3_block9_out;
    string path28 = common_path_mult + string("layer3_9_conv1_weight_32_32_3_3.txt");
    string path28a = common_path_sum + string("layer3_9_conv1_bias_32.txt");
    string path28_2 = common_path_mult + string("layer3_9_conv2_weight_32_32_3_3.txt");
    string path28_2a = common_path_sum + string("layer3_9_conv2_bias_32.txt");
    layer3_block9_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path28, path28a, path28_2, path28_2a, layer3_block8_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block9_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block8_out.clear();
    layer3_block8_out.shrink_to_fit();

     //////////RB29///////////////
    std::cout << "layer3_RB10 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block10_out;
    string path29 = common_path_mult + string("layer3_10_conv1_weight_32_32_3_3.txt");
    string path29a = common_path_sum + string("layer3_10_conv1_bias_32.txt");
    string path29_2 = common_path_mult + string("layer3_10_conv2_weight_32_32_3_3.txt");
    string path29_2a = common_path_sum + string("layer3_10_conv2_bias_32.txt");
    layer3_block10_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path29, path29a, path29_2, path29_2a, layer3_block9_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block10_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block9_out.clear();
    layer3_block9_out.shrink_to_fit();

     // result save.
    cout << "saving layer3_RB10 ctxt_result_bundel.." << endl;
    string savepath7 = "/app/result/layer3_RB10/";
    saveMsgBundle(dec, sk, layer3_block10_out,savepath7);    
    cout << "done" << endl; 

     //////////RB30///////////////
    std::cout << "layer3_RB11 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block11_out;
    string path30 = common_path_mult + string("layer3_11_conv1_weight_32_32_3_3.txt");
    string path30a = common_path_sum + string("layer3_11_conv1_bias_32.txt");
    string path30_2 = common_path_mult + string("layer3_11_conv2_weight_32_32_3_3.txt");
    string path30_2a = common_path_sum + string("layer3_11_conv2_bias_32.txt");
    layer3_block11_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path30, path30a, path30_2, path30_2a, layer3_block10_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block11_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block10_out.clear();
    layer3_block10_out.shrink_to_fit();

     //////////RB31///////////////
    std::cout << "layer3_RB12 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block12_out;
    string path31 = common_path_mult + string("layer3_12_conv1_weight_32_32_3_3.txt");
    string path31a = common_path_sum + string("layer3_12_conv1_bias_32.txt");
    string path31_2 = common_path_mult + string("layer3_12_conv2_weight_32_32_3_3.txt");
    string path31_2a = common_path_sum + string("layer3_12_conv2_bias_32.txt");
    layer3_block12_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path31, path31a, path31_2, path31_2a, layer3_block11_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block12_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block11_out.clear();
    layer3_block11_out.shrink_to_fit();

      //////////RB32///////////////
    std::cout << "layer3_RB13 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block13_out;
    string path32 = common_path_mult + string("layer3_13_conv1_weight_32_32_3_3.txt");
    string path32a = common_path_sum + string("layer3_13_conv1_bias_32.txt");
    string path32_2 = common_path_mult + string("layer3_13_conv2_weight_32_32_3_3.txt");
    string path32_2a = common_path_sum + string("layer3_13_conv2_bias_32.txt");
    layer3_block13_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path32, path32a, path32_2, path32_2a, layer3_block12_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block13_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block12_out.clear();
    layer3_block12_out.shrink_to_fit();

      //////////RB33///////////////
    std::cout << "layer3_RB14 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block14_out;
    string path33 = common_path_mult + string("layer3_14_conv1_weight_32_32_3_3.txt");
    string path33a = common_path_sum + string("layer3_14_conv1_bias_32.txt");
    string path33_2 = common_path_mult + string("layer3_14_conv2_weight_32_32_3_3.txt");
    string path33_2a = common_path_sum + string("layer3_14_conv2_bias_32.txt");
    layer3_block14_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path33, path33a, path33_2, path33_2a, layer3_block13_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block14_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block13_out.clear();
    layer3_block13_out.shrink_to_fit();

      //////////RB34///////////////
    std::cout << "layer3_RB15 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block15_out;
    string path34 = common_path_mult + string("layer3_15_conv1_weight_32_32_3_3.txt");
    string path34a = common_path_sum + string("layer3_15_conv1_bias_32.txt");
    string path34_2 = common_path_mult + string("layer3_15_conv2_weight_32_32_3_3.txt");
    string path34_2a = common_path_sum + string("layer3_15_conv2_bias_32.txt");
    layer3_block15_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path34, path34a, path34_2, path34_2a, layer3_block14_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block15_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block14_out.clear();
    layer3_block14_out.shrink_to_fit();

     // result save.
    cout << "saving layer3_RB15 ctxt_result_bundel.." << endl;
    string savepath8 = "/app/result/layer3_RB15/";
    saveMsgBundle(dec, sk, layer3_block15_out,savepath8);    
    cout << "done" << endl; 

      //////////RB35///////////////
    std::cout << "layer3_RB16 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block16_out;
    string path35 = common_path_mult + string("layer3_16_conv1_weight_32_32_3_3.txt");
    string path35a = common_path_sum + string("layer3_16_conv1_bias_32.txt");
    string path35_2 = common_path_mult + string("layer3_16_conv2_weight_32_32_3_3.txt");
    string path35_2a = common_path_sum + string("layer3_16_conv2_bias_32.txt");
    layer3_block16_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path35, path35a, path35_2, path35_2a, layer3_block15_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block16_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block15_out.clear();
    layer3_block15_out.shrink_to_fit();

      //////////RB36///////////////
    std::cout << "layer3_RB17 .. " << std::endl;
    vector<vector<Ciphertext>> layer3_block17_out;
    string path36 = common_path_mult + string("layer3_17_conv1_weight_32_32_3_3.txt");
    string path36a = common_path_sum + string("layer3_17_conv1_bias_32.txt");
    string path36_2 = common_path_mult + string("layer3_17_conv2_weight_32_32_3_3.txt");
    string path36_2a = common_path_sum + string("layer3_17_conv2_bias_32.txt");
    layer3_block17_out = RB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path36, path36a, path36_2, path36_2a, layer3_block16_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer3_block17_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer3_block16_out.clear();
    layer3_block16_out.shrink_to_fit();

      // result save.
    cout << "saving layer3_RB17 ctxt_result_bundel.." << endl;
    string savepath9 = "/app/result/layer3_RB17/";
    saveMsgBundle(dec, sk, layer3_block17_out,savepath9);    
    cout << "done" << endl; 

    ////////////////////////
    /////// DSB 2 //////////
    /////////////////////////
    
     ///////////////////// Residual flow ////////////////////////////
    // Convolution

    
    cout << "uploading for DSB2_conv_onebyone ...\n\n";
    timer.start(" * ");
    vector<double> temp14;
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1(64, vector<vector<Plaintext>>(32, vector<Plaintext>(1, ptxt_init)));
    string path37 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("layer4_0_downsample_weight_64_32_1_1.txt");
    txtreader(temp14, path37);
    kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 5, 2, 2, 64, 32, 1, ecd);
    temp14.clear();
    temp14.shrink_to_fit();

    vector<Plaintext> block7conv_onebyone_summands64(64, ptxt_init);
    vector<double> temp14a;
    string path37a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("layer4_0_downsample_bias_64.txt");
    Scaletxtreader(temp14a, path37a, cnst);

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp14a[i]);
        block7conv_onebyone_summands64[i]=ecd.encode(msg, 4, 0);
    }
    temp14a.clear();
    temp14a.shrink_to_fit();
    
    timer.end();

    cout << "DSB2_conv_onebyone ..." << endl;
    timer.start(" DSB2_conv_onebyone .. ");
    cout << "convolution ...\n\n";
    vector<vector<Ciphertext>> ctxt_block7conv_onebyone_out(4, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 4; ++i) { // 서로 다른 img
        #pragma omp parallel num_threads(10)
        {
            ctxt_block7conv_onebyone_out[i] = Conv(context, pack, eval, 32, 2, 2, 32, 64, layer3_block17_out[i], block7conv_onebyone_multiplicands64_32_1_1);
        }
    }


    block7conv_onebyone_multiplicands64_32_1_1.clear();
    block7conv_onebyone_multiplicands64_32_1_1.shrink_to_fit();

    cout << "Done!! \n" << "\n";
    // cout << "and decrypted message is ... " << "\n";
    // dec.decrypt(ctxt_block7conv_onebyone_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    // MPP input bundle making
    cout << "MPPacking ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block7MPP1_in(1, vector<vector<Ciphertext>>(64, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block7MPP1_in[i][ch][k] = ctxt_block7conv_onebyone_out[4 * i + k][ch];
            }
        }
    }
    ctxt_block7conv_onebyone_out.clear();
    ctxt_block7conv_onebyone_out.shrink_to_fit();
    
    
    // dec.decrypt(ctxt_block7MPP1_in[0][0][0], sk, dmsg);
    // printMessage(dmsg);
    
    // MPP
    vector<vector<Ciphertext>> ctxt_block7MPP1_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            ctxt_block7MPP1_out[i][ch] = MPPacking2(context, pack, eval, 32, ctxt_block7MPP1_in[i][ch]);
        }
    }

    ctxt_block7MPP1_in.clear();
    ctxt_block7MPP1_in.shrink_to_fit();
    
    
    // dec.decrypt(ctxt_block7MPP1_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    addBNsummands(context, eval, ctxt_block7MPP1_out, block7conv_onebyone_summands64, 1, 64);
    timer.end();

    block7conv_onebyone_summands64.clear();
    block7conv_onebyone_summands64.shrink_to_fit();

    cout << "Done!! \n" << "\n";
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block7MPP1_out[0][0], sk, dmsg);
    // printMessage(dmsg);
    
    
    
    


    ///////////////////////// Main flow /////////////////////////////////////////
   
    
    cout << "uploading for DSB2_conv0 ...\n\n";
    timer.start(" * ");

    vector<double> temp15;
    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path37_1 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("layer4_0_conv1_weight_64_32_3_3.txt");
    txtreader(temp15, path37_1);
    kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 5, 2, 2, 64, 32, 3, ecd);
    temp15.clear();
    temp15.shrink_to_fit();


    vector<Plaintext> block7conv0summands64(64, ptxt_init);
    vector<double> temp15a;
    string path37_1a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("layer4_0_conv1_bias_64.txt");
    Scaletxtreader(temp15a, path37_1a, cnst);
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp15a[i]);
        block7conv0summands64[i]=ecd.encode(msg, 4, 0);
    }
    temp15a.clear();
    temp15a.shrink_to_fit();
    
    timer.end();

    cout << "DSB2_conv0 ..." << endl;
    timer.start(" DSB2_conv0 ");
    cout << "convolution ... \n\n";
    vector<vector<Ciphertext>> ctxt_block7conv0_out(4, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 4; ++i) { // 서로 다른 img
        ctxt_block7conv0_out[i] = Conv_parallel(context, pack, eval, 32, 2, 2, 32, 64, layer3_block17_out[i], block7conv0multiplicands64_32_3_3);
    }

    layer3_block17_out.clear();
    layer3_block17_out.shrink_to_fit();

    block7conv0multiplicands64_32_3_3.clear();
    block7conv0multiplicands64_32_3_3.shrink_to_fit();

    // cout << "Done!! level of ctxt is " << ctxt_block7conv0_out[0][0].getLevel() << "\n";
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block7conv0_out[0][0], sk, dmsg);
    // printMessage(dmsg);


    // MPP input bundle making
    cout << "MPpacking ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block7MPP0_in(1, vector<vector<Ciphertext>>(64, vector<Ciphertext>(4, ctxt_init)));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block7MPP0_in[i][ch][k] = ctxt_block7conv0_out[4 * i + k][ch];
            }
        }
    }
    
    
    // dec.decrypt(ctxt_block7MPP0_in[0][0][0], sk, dmsg);
    // printMessage(dmsg);
    

    ctxt_block7conv0_out.clear();
    ctxt_block7conv0_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block7MPP0_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            ctxt_block7MPP0_out[i][ch] = MPPacking2(context, pack, eval, 32, ctxt_block7MPP0_in[i][ch]);
        }
    }

    ctxt_block7MPP0_in.clear();
    ctxt_block7MPP0_in.shrink_to_fit();
    
    // dec.decrypt(ctxt_block7MPP0_out[0][0], sk, dmsg);
    // printMessage(dmsg);
    
    addBNsummands(context, eval, ctxt_block7MPP0_out, block7conv0summands64, 1, 64);
    timer.end();

    block7conv0summands64.clear();
    block7conv0summands64.shrink_to_fit();

    cout << "Done!! \n\n";
    // cout << "and decrypted messagee is ... " << "\n";
    // dec.decrypt(ctxt_block7MPP0_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    // ctxt_block7MPP0_out 첫번째 : 서로 다른 img, 두번째 : ch.

    // AppReLU
    cout << "DSB2_relu0 ..." << endl;
    timer.start(" DSB2_relu0 ");
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

    // dec.decrypt(ctxt_block7relu0_out[0][0], sk, dmsg);
    // printMessage(dmsg);

    
    
 // Second convolution
    
    
    cout << "uploading for DSB2_conv1 ...\n\n";
    timer.start(" * ");

    vector<double> temp16;
    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path37_2 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("layer4_0_conv2_weight_64_64_3_3.txt");
    txtreader(temp16, path37_2);
    kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp16.clear();
    temp16.shrink_to_fit();

    vector<Plaintext> block7conv1summands64(64, ptxt_init);
    vector<double> temp16a;
    string path37_2a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("layer4_0_conv2_bias_64.txt");
    Scaletxtreader(temp16a, path37_2a, cnst);
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp16a[i]);
        block7conv1summands64[i]=ecd.encode(msg, 4, 0);
    }

    temp16a.clear();
    temp16a.shrink_to_fit();
    
    timer.end();
    cout << "DSB2_conv1 ..." << endl;
    timer.start(" DSB2_conv1 ");
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

    cout << "Done!! \n\n";
    // cout << "and decrypted message is ... " << "\n";
    // dec.decrypt(ctxt_block7conv1_out[0][0], sk, dmsg);
    // printMessage(dmsg);


    


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "DSB2_add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block7add_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2)
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
    cout << "DSB2_relu1 ..." << endl;
    timer.start(" DSB2_relu1 ");
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
    cout << "DONE!, decrypted message is ...\n\n";
    dec.decrypt(ctxt_block7relu1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout << "downsampling DSB2 DONE!" << "\n";

      // result save.
    cout << "saving DSB2 ctxt_result_bundel.." << endl;
    string savepath10 = "/app/result/DSB2/";
    saveMsgBundle(dec, sk, ctxt_block7relu1_out,savepath10);    
    cout << "done" << endl; 


    //////////RB38///////////////
    std::cout << "layer4_RB1 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block1_out;
    string path38 = common_path_mult + string("layer4_1_conv1_weight_64_64_3_3.txt");
    string path38a = common_path_sum + string("layer4_1_conv1_bias_64.txt");
    string path38_2 = common_path_mult + string("layer4_1_conv2_weight_64_64_3_3.txt");
    string path38_2a = common_path_sum + string("layer4_1_conv2_bias_64.txt");
    layer4_block1_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path38, path38a, path38_2, path38_2a, ctxt_block7relu1_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    ctxt_block7relu1_out.clear();
    ctxt_block7relu1_out.shrink_to_fit();

     //////////RB39///////////////
    std::cout << "layer4_RB2 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block2_out;
    string path39 = common_path_mult + string("layer4_2_conv1_weight_64_64_3_3.txt");
    string path39a = common_path_sum + string("layer4_2_conv1_bias_64.txt");
    string path39_2 = common_path_mult + string("layer4_2_conv2_weight_64_64_3_3.txt");
    string path39_2a = common_path_sum + string("layer4_2_conv2_bias_64.txt");
    layer4_block2_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path39, path39a, path39_2, path39_2a, layer4_block1_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block2_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block1_out.clear();
    layer4_block1_out.shrink_to_fit();

     //////////RB40///////////////
    std::cout << "layer4_RB3 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block3_out;
    string path40 = common_path_mult + string("layer4_3_conv1_weight_64_64_3_3.txt");
    string path40a = common_path_sum + string("layer4_3_conv1_bias_64.txt");
    string path40_2 = common_path_mult + string("layer4_3_conv2_weight_64_64_3_3.txt");
    string path40_2a = common_path_sum + string("layer4_3_conv2_bias_64.txt");
    layer4_block3_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path40, path40a, path40_2, path40_2a, layer4_block2_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block3_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block2_out.clear();
    layer4_block2_out.shrink_to_fit();

     //////////RB41///////////////
    std::cout << "layer4_RB4 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block4_out;
    string path41 = common_path_mult + string("layer4_4_conv1_weight_64_64_3_3.txt");
    string path41a = common_path_sum + string("layer4_4_conv1_bias_64.txt");
    string path41_2 = common_path_mult + string("layer4_4_conv2_weight_64_64_3_3.txt");
    string path41_2a = common_path_sum + string("layer4_4_conv2_bias_64.txt");
    layer4_block4_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path41, path41a, path41_2, path41_2a, layer4_block3_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block4_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block3_out.clear();
    layer4_block3_out.shrink_to_fit();

     //////////RB42///////////////
    std::cout << "layer4_RB5 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block5_out;
    string path42 = common_path_mult + string("layer4_5_conv1_weight_64_64_3_3.txt");
    string path42a = common_path_sum + string("layer4_5_conv1_bias_64.txt");
    string path42_2 = common_path_mult + string("layer4_5_conv2_weight_64_64_3_3.txt");
    string path42_2a = common_path_sum + string("layer4_5_conv2_bias_64.txt");
    layer4_block5_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path42, path42a, path42_2, path42_2a, layer4_block4_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block5_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block4_out.clear();
    layer4_block4_out.shrink_to_fit();

      // result save.
    cout << "saving layer4_RB5 ctxt_result_bundel.." << endl;
    string savepath11 = "/app/result/layer4_RB5/";
    saveMsgBundle(dec, sk, layer4_block5_out,savepath11);    
    cout << "done" << endl; 

     //////////RB43///////////////
    std::cout << "layer4_RB6 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block6_out;
    string path43 = common_path_mult + string("layer4_6_conv1_weight_64_64_3_3.txt");
    string path43a = common_path_sum + string("layer4_6_conv1_bias_64.txt");
    string path43_2 = common_path_mult + string("layer4_6_conv2_weight_64_64_3_3.txt");
    string path43_2a = common_path_sum + string("layer4_6_conv2_bias_64.txt");
    layer4_block6_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path43, path43a, path43_2, path43_2a, layer4_block5_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block6_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block5_out.clear();
    layer4_block5_out.shrink_to_fit();

      //////////RB44///////////////
    std::cout << "layer4_RB7 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block7_out;
    string path44 = common_path_mult + string("layer4_7_conv1_weight_64_64_3_3.txt");
    string path44a = common_path_sum + string("layer4_7_conv1_bias_64.txt");
    string path44_2 = common_path_mult + string("layer4_7_conv2_weight_64_64_3_3.txt");
    string path44_2a = common_path_sum + string("layer4_7_conv2_bias_64.txt");
    layer4_block7_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path44, path44a, path44_2, path44_2a, layer4_block6_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block7_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block6_out.clear();
    layer4_block6_out.shrink_to_fit();

      //////////RB45///////////////
    std::cout << "layer4_RB8 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block8_out;
    string path45 = common_path_mult + string("layer4_8_conv1_weight_64_64_3_3.txt");
    string path45a = common_path_sum + string("layer4_8_conv1_bias_64.txt");
    string path45_2 = common_path_mult + string("layer4_8_conv2_weight_64_64_3_3.txt");
    string path45_2a = common_path_sum + string("layer4_8_conv2_bias_64.txt");
    layer4_block8_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path45, path45a, path45_2, path45_2a, layer4_block7_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block8_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block7_out.clear();
    layer4_block7_out.shrink_to_fit();

     //////////RB46///////////////
    std::cout << "layer4_RB9 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block9_out;
    string path46 = common_path_mult + string("layer4_9_conv1_weight_64_64_3_3.txt");
    string path46a = common_path_sum + string("layer4_9_conv1_bias_64.txt");
    string path46_2 = common_path_mult + string("layer4_9_conv2_weight_64_64_3_3.txt");
    string path46_2a = common_path_sum + string("layer4_9_conv2_bias_64.txt");
    layer4_block9_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path46, path46a, path46_2, path46_2a, layer4_block8_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block9_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block8_out.clear();
    layer4_block8_out.shrink_to_fit();

    ///////////RB47///////////////
    std::cout << "layer4_RB10 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block10_out;
    string path47 = common_path_mult + string("layer4_10_conv1_weight_64_64_3_3.txt");
    string path47a = common_path_sum + string("layer4_10_conv1_bias_64.txt");
    string path47_2 = common_path_mult + string("layer4_10_conv2_weight_64_64_3_3.txt");
    string path47_2a = common_path_sum + string("layer4_10_conv2_bias_64.txt");
    layer4_block10_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path47, path47a, path47_2, path47_2a, layer4_block9_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block10_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block9_out.clear();
    layer4_block9_out.shrink_to_fit();

      // result save.
    cout << "saving layer4_RB10 ctxt_result_bundel.." << endl;
    string savepath12 = "/app/result/layer4_RB10/";
    saveMsgBundle(dec, sk, layer4_block10_out,savepath12);    
    cout << "done" << endl;

     //////////RB48///////////////
    std::cout << "layer4_RB11 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block11_out;
    string path48 = common_path_mult + string("layer4_11_conv1_weight_64_64_3_3.txt");
    string path48a = common_path_sum + string("layer4_11_conv1_bias_64.txt");
    string path48_2 = common_path_mult + string("layer4_11_conv2_weight_64_64_3_3.txt");
    string path48_2a = common_path_sum + string("layer4_11_conv2_bias_64.txt");
    layer4_block11_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path48, path48a, path48_2, path48_2a, layer4_block10_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block11_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block10_out.clear();
    layer4_block10_out.shrink_to_fit();

    //////////RB49///////////////
    std::cout << "layer4_RB12 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block12_out;
    string path49 = common_path_mult + string("layer4_12_conv1_weight_64_64_3_3.txt");
    string path49a = common_path_sum + string("layer4_12_conv1_bias_64.txt");
    string path49_2 = common_path_mult + string("layer4_12_conv2_weight_64_64_3_3.txt");
    string path49_2a = common_path_sum + string("layer4_12_conv2_bias_64.txt");
    layer4_block12_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path49, path49a, path49_2, path49_2a, layer4_block11_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block12_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block11_out.clear();
    layer4_block11_out.shrink_to_fit();

    //////////RB50///////////////
    std::cout << "layer4_RB13 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block13_out;
    string path50 = common_path_mult + string("layer4_13_conv1_weight_64_64_3_3.txt");
    string path50a = common_path_sum + string("layer4_13_conv1_bias_64.txt");
    string path50_2 = common_path_mult + string("layer4_13_conv2_weight_64_64_3_3.txt");
    string path50_2a = common_path_sum + string("layer4_13_conv2_bias_64.txt");
    layer4_block13_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path50, path50a, path50_2, path50_2a, layer4_block12_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block13_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block12_out.clear();
    layer4_block12_out.shrink_to_fit();

    //////////RB51///////////////
    std::cout << "layer4_RB14.. " << std::endl;
    vector<vector<Ciphertext>> layer4_block14_out;
    string path51 = common_path_mult + string("layer4_14_conv1_weight_64_64_3_3.txt");
    string path51a = common_path_sum + string("layer4_14_conv1_bias_64.txt");
    string path51_2 = common_path_mult + string("layer4_14_conv2_weight_64_64_3_3.txt");
    string path51_2a = common_path_sum + string("layer4_14_conv2_bias_64.txt");
    layer4_block14_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path51, path51a, path51_2, path51_2a, layer4_block13_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block14_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block13_out.clear();
    layer4_block13_out.shrink_to_fit();

     //////////RB52///////////////
    std::cout << "layer4_RB15 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block15_out;
    string path52 = common_path_mult + string("layer4_15_conv1_weight_64_64_3_3.txt");
    string path52a = common_path_sum + string("layer4_15_conv1_bias_64.txt");
    string path52_2 = common_path_mult + string("layer4_15_conv2_weight_64_64_3_3.txt");
    string path52_2a = common_path_sum + string("layer4_15_conv2_bias_64.txt");
    layer4_block15_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path52, path52a, path52_2, path52_2a, layer4_block14_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block15_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block14_out.clear();
    layer4_block14_out.shrink_to_fit();


      // result save.
    cout << "saving layer4_RB15 ctxt_result_bundel.." << endl;
    string savepath13 = "/app/result/layer4_RB15/";
    saveMsgBundle(dec, sk, layer4_block15_out,savepath13);    
    cout << "done" << endl;
    

     //////////RB53///////////////
    std::cout << "layer4_RB16 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block16_out;
    string path53 = common_path_mult + string("layer4_16_conv1_weight_64_64_3_3.txt");
    string path53a = common_path_sum + string("layer4_16_conv1_bias_64.txt");
    string path53_2 = common_path_mult + string("layer4_16_conv2_weight_64_64_3_3.txt");
    string path53_2a = common_path_sum + string("layer4_16_conv2_bias_64.txt");
    layer4_block16_out = RB3(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path53, path53a, path53_2, path53_2a, layer4_block15_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block16_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block15_out.clear();
    layer4_block15_out.shrink_to_fit();


     //////////RB54///////////////
    std::cout << "layer4_RB17 .. " << std::endl;
    vector<vector<Ciphertext>> layer4_block17_out;
    string path54 = common_path_mult + string("layer4_17_conv1_weight_64_64_3_3.txt");
    string path54a = common_path_sum + string("layer4_17_conv1_bias_64.txt");
    string path54_2 = common_path_mult + string("layer4_17_conv2_weight_64_64_3_3.txt");
    string path54_2a = common_path_sum + string("layer4_17_conv2_bias_64.txt");
    layer4_block17_out = RB3last(timer, context, pack, eval, ecd, ctxt_init, ptxt_init , cnst, log_slots,
        path54, path54a, path54_2, path54_2a, layer4_block16_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(layer4_block17_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";
    
    layer4_block16_out.clear();
    layer4_block16_out.shrink_to_fit();

      // result save.
    cout << "saving layer4_RB17 ctxt_result_bundel.." << endl;
    string savepath14 = "/app/result/layer4_RB17/";
    saveMsgBundle(dec, sk, layer4_block17_out,savepath14);    
    cout << "done" << endl;
    

    


    // Avg Pool
    cout << "evaluating Avgpool" << "\n";
    vector<Ciphertext> ctxt_avgp_out;
    timer.start(" avgpool * ");
    ctxt_avgp_out = Avgpool(context, pack, eval, layer4_block17_out[0]);
    timer.end();
    
    // std::cout << "AvgPool result" << std::endl;
    // dec.decrypt(ctxt_avgp_out[0], sk, dmsg);
    // printMessage(dmsg);

    layer4_block17_out.clear();
    layer4_block17_out.shrink_to_fit();
    
    //FC64 setup...
    
    
    cout << "uploading for FC64 layer ...\n\n";
    timer.start(" FC64 layer * ");
    vector<double> temp21;
    vector<vector<Plaintext>> fclayermultiplicands10_64(10, vector<Plaintext>(64, ptxt_init));
    string path56 = "/app/HEAAN-ResNet-110/resnet110/multiplicands/" + string("fc_weight_10_64.txt");
    double cnst2 = (double)(1.0 / 64.0);
    Scaletxtreader(temp21, path56,cnst2);
    
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            Message msg(log_slots, temp21[64 * i + j]);
            fclayermultiplicands10_64[i][j] = ecd.encode(msg, 1, 0);
        }
    }

    temp21.clear();
    temp21.shrink_to_fit();

    vector<double> temp21a;
    vector<Plaintext> fclayersummands10(10, ptxt_init);
    string path56a = "/app/HEAAN-ResNet-110/resnet110/summands/" + string("fc_bias_10.txt");

    double cnst1 = (double)(1.0 / 40.0);
    Scaletxtreader(temp21a, path56a, cnst1);
    
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 10; ++i) {
        #pragma omp parallel num_threads(4)
        {
        Message msg(log_slots, temp21a[i]);
        fclayersummands10[i] = ecd.encode(msg, 0, 0);
        }
    }

    temp21a.clear();
    temp21a.shrink_to_fit();
    
    timer.end();
    
     // FC64
    cout << "evaluating FC64 layer\n" << endl;

    vector<Ciphertext> ctxt_result;
    timer.start(" FC64 layer * ");
    ctxt_result = FC64(context, pack, eval, ctxt_avgp_out, fclayermultiplicands10_64, fclayersummands10);
    timer.end();

    std::cout << "FC64 result..." << std::endl;
    dec.decrypt(ctxt_result[0], sk, dmsg);
    printMessage(dmsg);

    fclayermultiplicands10_64.clear();
    fclayermultiplicands10_64.shrink_to_fit();
    fclayersummands10.clear();
    fclayersummands10.shrink_to_fit();

    timer.end();
    
    // Last Step; enumerating
    vector<vector<double>> orderVec(512, vector<double>(10, 0));
    vector<int> idx_table = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
    
    for (int t = 0; t < 10; ++t) {
        dec.decrypt(ctxt_result[t], sk, dmsg);

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 32; ++k) {
                    orderVec[32* idx_table[4*i+j]+k][t] = dmsg[1024 * k + 32 * i + j].real();
                }
            }
        }
    }
    
    
    cout << "Finaly, DONE!!!" << "\n\n";
    
    
    ///////////////////////////////////////
    ///////////// save file //////////////
    //////////////////////////////////////
    
    
    cout << "[ ";
    
    string savelabel = string("/app/output110/bundle")+to_string(num)+".txt";
    ofstream filesave(savelabel);
    
    for (int i = 0; i < 512; ++i) {
        int max_index = max_element(orderVec[i].begin(), orderVec[i].end()) - orderVec[i].begin();
        
        if (i%16 == 15) {
            cout << max_index << ",\n";
            filesave << max_index <<",\n";
        }
        else{
            cout << max_index << ", ";
            filesave << max_index <<", ";
        }
    }
    
    filesave.close();
    cout << "]\n";

    cout <<" end...\n";
    
    return 0;

}

