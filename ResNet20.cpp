#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

#include "HEaaN/heaan.hpp" 
#include "Conv.hpp"
#include "AppReLU.hpp"
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
    // SetUp
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
    // for 64 x 64 test , change enc level 5
    //enc.encrypt(msg_zero, pack, ctxt_init, 0, 0);
    enc.encrypt(msg_zero, pack, ctxt_init, 5, 0);
    Plaintext ptxt_init(context);
    // for 64 x 64 test, change enc level 5
    //ptxt_init = ecd.encode(msg_zero, 0, 0);
    ptxt_init = ecd.encode(msg_zero, 5, 0);




    ////////////////////////////////////////////////////////////
    ///////////// 10000 test image Encoding ///////////////////
    ////////////////////////////////////////////////////////////

    cout << "10000 test images encoding ... \n";

    int num;

    cout << "Choose one of bundle from 1 to 20 \n";
    cin >> num;

    cout << "\n Image Loading ..." << "\n";
    
    vector<vector<Ciphertext>> imageVec(16, vector<Ciphertext>(3, ctxt_zero));

    #pragma omp parallel for
    for (int i = (num-1) * 16; i < num*16; ++i) { // 313
        int ind = i+1;
        string str = "/app/HEAAN-ResNet-20/image/image_" + to_string(ind) + string(".txt");
        vector<double> temp;
        txtreader(temp, str);
        imageCompiler(context, pack, enc, 5, temp, imageVec[(i%16)]);

    }


    cout << "DONE, test for image encode ..." << "\n";

    Message dmsg;
    dec.decrypt(imageVec[0][0], sk, dmsg);
    printMessage(dmsg);

    cout << "DONE\n" << "\n";


    
    ////////////////////////////common path///////////////////////////////////
    string common_path_mult = "/app/HEAAN-ResNet-20/kernel/multiplicands/";
    string common_path_sum = "/app/HEAAN-ResNet-20/kernel/summands/";
    //////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////test///////////////////////////////////////////////////////
    // test for the 64 x 64 case
    cout << "test start" << endl;
    vector<vector<Ciphertext>> ctxt_conv0_out;
    vector<Ciphertext> convtemp(64, ctxt_init);
    
    vector<double> kernel_info1;
    string pathmult1 = common_path_mult + string("layer4_1_conv1_weight_64_64_3_3.txt");
    txtreader(kernel_info1, pathmult1);
    cout << "kernel vector length = " << kernel_info1.size() << endl;
    vector<vector<Ciphertext>> input(1, vector<Ciphertext>(64, ctxt_init));

    timer.start("temp case start...");
    
    convtemp = newConv(context, pack, eval, ecd, 32, 4, 1, 64, 64, input[0], kernel_info1, 3);
    ctxt_conv0_out.push_back(convtemp);
    
    timer.end();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////


    
    // 0st conv
    string path0 = common_path_mult + string("layer1_weight_16_3_3_3.txt");
    string path0a = common_path_sum + string("layer1_bias_16.txt");

    cout << " layer1 conv ... " << endl;
    timer.start(" * ");

    vector<vector<Ciphertext>> ctxt_block0conv0_out;
    vector<Ciphertext> convtemp0(16, ctxt_init);

    vector<double> kernel_info;
    txtreader(kernel_info, path0);


    for (int i=0; i<16; ++i){
        convtemp0 = newConv(context, pack, eval, ecd, 32, 1, 1, 3, 16, imageVec[i], kernel_info, 3);
        ctxt_block0conv0_out.push_back(convtemp0);
    }

    kernel_info.clear();
    kernel_info.shrink_to_fit();
    
    vector<Plaintext> bias0(16, ptxt_init);
    vector<double> temp0a;
    Scaletxtreader(temp0a, path0a, cnst);
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp0a[i]);
        bias0[i]=ecd.encode(msg, 4, 0);
    }

    temp0a.clear();
    temp0a.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block0conv0_out, bias0, 16, 16);
    timer.end();
    cout << "DONE!\n";

    imageVec.clear();
    imageVec.shrink_to_fit();

    bias0.clear();
    bias0.shrink_to_fit();

    ///////////////추가///////////////////
    cout << "first conv result test..." << endl;
    dec.decrypt(ctxt_block0conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);
    //////////////////////////////////////

    // // AppReLU
    cout << "layer1 relu ...\n\n";
    timer.start(" block0relu0 ");
    vector<vector<Ciphertext>> ctxt_block0relu0_out(16, vector<Ciphertext>(16, ctxt_init)); //초기화부분 추가
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[i / 5][i % 5], ctxt_block0relu0_out[i / 5][i % 5]);
    }
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[8+(i /5)][(i%5)], ctxt_block0relu0_out[8+(i /5)][(i%5)]);
    }
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[i / 5][5+(i%5)], ctxt_block0relu0_out[i / 5][5+(i%5)]);
    }
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[8+(i /5)][5+(i%5)], ctxt_block0relu0_out[8+(i /5)][5+(i%5)]);
    }
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[i / 5][10+(i%5)], ctxt_block0relu0_out[i / 5][10+(i%5)]);
    }
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 40; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[8+(i /5)][10+(i%5)], ctxt_block0relu0_out[8+(i /5)][10+(i%5)]);
    }
    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 16; ++i) {
        ApproxReLU(context, eval, ctxt_block0conv0_out[i][15], ctxt_block0relu0_out[i][15]);
    }
    timer.end();

    
    ctxt_block0conv0_out.clear();
    ctxt_block0conv0_out.shrink_to_fit();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block0relu0_out[0][0], sk, dmsg);
    printMessage(dmsg);

    cout << "layer1 DONE!\n" << "\n";


    ////////////////////
    /////// RB1 ////////
    ////////////////////
    std::cout << "layer2_RB1 ... " << std::endl;
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

    ////////////////////
    /////// RB2 ////////
    ////////////////////
    std::cout << "layer2_RB2 ... " << std::endl;
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

    ////////////////////
    /////// RB3 ////////
    ////////////////////
    std::cout << "layer2_RB3 ... " << std::endl;
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

    ////////////////////
    /////// DSB1 ///////
    ////////////////////
    std::cout << "layer3_DSB1 .. " << std::endl;
    vector<vector<Ciphertext>> ctxt_block4relu1_out;
    ctxt_block4relu1_out = DSB1(timer, context, pack, eval, ecd, ctxt_init, ptxt_init, cnst, log_slots, layer2_block2_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block4relu1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";



    ////////////////////
    /////// RB4 ////////
    ////////////////////
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

    ////////////////////
    /////// RB5 ////////
    ////////////////////
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

    ////////////////////
    /////// DSB2 ///////
    ////////////////////
    // block화 시키기...
    // 으로 retrun: ctxt_block7relu1_out
    std::cout << "layer4_DSB2 .. " << std::endl;
    vector<vector<Ciphertext>> ctxt_block7relu1_out;
    ctxt_block7relu1_out = DSB2(timer, context, pack, eval, ecd, ctxt_init, ptxt_init, cnst, log_slots, layer3_block2_out);
    
    cout << "DONE!, decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block7relu1_out[0][0], sk, dmsg);
    printMessage(dmsg);
    cout <<"\n";


    ////////////////////
    /////// RB6 ////////
    ////////////////////
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


    ////////////////////
    /////// RB7 ////////
    ////////////////////
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

    /////////////////////////
    /////// Avg Pool ////////
    /////////////////////////
    cout << "evaluating Avgpool" << "\n";
    vector<Ciphertext> ctxt_avgp_out;
    timer.start(" avgpool * ");
    ctxt_avgp_out = Avgpool(context, pack, eval, layer4_block2_out[0]);
    timer.end();
    
    // std::cout << "AvgPool result" << std::endl;
    // dec.decrypt(ctxt_avgp_out[0], sk, dmsg);
    // printMessage(dmsg);

    layer4_block2_out.clear();
    layer4_block2_out.shrink_to_fit();
    
    //FC64 setup...
    
    
    cout << "uploading for FC64 layer ...\n\n";
    timer.start(" FC64 layer * ");
    vector<double> temp21;
    vector<vector<Plaintext>> fclayermultiplicands10_64(10, vector<Plaintext>(64, ptxt_init));
    string path56 = "/app/HEAAN-ResNet-20/resnet20/multiplicands/" + string("fc_weight_10_64.txt");
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
    string path56a = "/app/HEAAN-ResNet-20/resnet20/summands/" + string("fc_bias_10.txt");

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
    
    string savelabel = string("/app/output20/bundle")+to_string(num)+".txt";
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