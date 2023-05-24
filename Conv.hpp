#pragma once
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include "rotsum.hpp"

namespace {
    using namespace HEaaN;
    using namespace std;
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                     //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed Withno the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

vector<Plaintext> maskBundle128(EnDecoder ecd, HomEvaluator eval, Context context) {
    int level = 4;
    Message v(15);
    vector<Message> msg(4, v);
    Plaintext ptxt(context);
    vector<Plaintext> out(4, ptxt);

    for (size_t j = 0; j < 32768; ++j) {
        msg[0][j].real(0.0);
        msg[0][j].imag(0.0);
    }
    for (size_t i = 1; i < 4; ++i) {
        msg[i] = msg[0];
    }

    for (size_t j = 0; j < 32768; ++j) {
        if ((j%4)<2 && (j%128)<64) {
            msg[0][j].real(1.0);
        }
        else if ((j%4)>=2 && (j%128)<64) {
            msg[1][j].real(1.0);
        }
        else if ((j%4)<2 && (j%128)>=64) {
            msg[2][j].real(1.0);
        }
        else if ((j%4)>=2 && (j%128)>=64) {
            msg[3][j].real(1.0);
        }
    }
    
    for (size_t j = 0; j < 4; ++j) {
        out[j] = ecd.encode(msg[j], level, 0);
    }
    return out;
}

vector<Plaintext> maskBundle32_1(EnDecoder ecd, HomEvaluator eval, Context context) {
    int level = 4;
    Message v(15);
    vector<Message> msg(4, v);
    Plaintext ptxt(context);
    vector<Plaintext> out(4, ptxt);

    for (size_t j = 0; j < 32768; ++j) {
        msg[0][j].real(0.0);
        msg[0][j].imag(0.0);
    }
    for (size_t i = 1; i < 4; ++i) {
        msg[i] = msg[0];
    }

    for (size_t j = 0; j < 32768; ++j) {
        if ((j%2)<1 && (j%64)<32) {
            msg[0][j].real(1.0);
        }
        else if ((j%2)>=1 && (j%64)<32) {
            msg[1][j].real(1.0);
        }
        else if ((j%2)<1 && (j%64)>=32) {
            msg[2][j].real(1.0);
        }
        else if ((j%2)>=1 && (j%64)>=32) {
            msg[3][j].real(1.0);
        }
    }

    for (size_t j = 0; j < 4; ++j) {
        out[j] = ecd.encode(msg[j], level, 0);
    }
    return out;
}

vector<Plaintext> maskBundle32_2(EnDecoder ecd, HomEvaluator eval, Context context) {
    int level = 4;
    Message v(15);
    vector<Message> msg(16, v);
    Plaintext ptxt(context);
    vector<Plaintext> out(16, ptxt);
    for (size_t j = 0; j < 32768; ++j) {
        msg[0][j].real(0.0);
        msg[0][j].imag(0.0);
    }
    for (size_t i = 1; i < 16; ++i) {
        msg[i] = msg[0];
    }
    
    for (size_t j = 0; j < 32768; ++j) {
        if (j%4==0 && (j/32)%4==0) {
            msg[0][j].real(1.0);
        }
        else if (j%4==1 && (j/32)%4==0) {
            msg[1][j].real(1.0);
        }
        else if (j%4==0 && (j/32)%4==1) {
            msg[2][j].real(1.0);
        }
        else if (j%4==1 && (j/32)%4==1) {
            msg[3][j].real(1.0);
        }
        else if (j%4==2 && (j/32)%4==0) {
            msg[4][j].real(1.0);
        }
        else if (j%4==3 && (j/32)%4==0) {
            msg[5][j].real(1.0);
        }
        else if (j%4==2 && (j/32)%4==1) {
            msg[6][j].real(1.0);
        }
        else if (j%4==3 && (j/32)%4==1) {
            msg[7][j].real(1.0);
        }
        else if (j%4==0 && (j/32)%4==2) {
            msg[8][j].real(1.0);
        }
        else if (j%4==1 && (j/32)%4==2) {
            msg[9][j].real(1.0);
        }
        else if (j%4==0 && (j/32)%4==3) {
            msg[10][j].real(1.0);
        }
        else if (j%4==1 && (j/32)%4==3) {
            msg[11][j].real(1.0);
        }
        else if (j%4==2 && (j/32)%4==2) {
            msg[12][j].real(1.0);
        }
        else if (j%4==3 && (j/32)%4==2) {
            msg[13][j].real(1.0);
        }
        else if (j%4==2 && (j/32)%4==3) {
            msg[14][j].real(1.0);
        }
        else if (j%4==3 && (j/32)%4==3) {
            msg[15][j].real(1.0);
        }
    }
    
    for (size_t j = 0; j < 16; ++j) {
        out[j] = ecd.encode(msg[j], level, 0);
    }
    return out;
}

Ciphertext auxiliaryFtn1(HomEvaluator eval, Context context, vector<Ciphertext>& ctxtVec, vector<vector<Plaintext>>& ptxtVec, int input_channel) {

    Ciphertext ctxt_out(context);

    for (int inputid = 0; inputid < input_channel; ++inputid) {

        HEaaN::Ciphertext ctxt_out_cache(context);

        int level1 = ctxtVec[inputid].getLevel();
        int level2 = ptxtVec[inputid][0].getLevel();
        if (level1 > level2) {
            eval.levelDown(ctxtVec[inputid], level2, ctxtVec[inputid]);
        }
        else if (level1 < level2) {
            ptxtVec[inputid][0].setLevel(level1);
        }
        eval.multWithoutRescale(ctxtVec[inputid], ptxtVec[inputid][0], ctxt_out_cache);

        if (inputid == 0) {
            ctxt_out = ctxt_out_cache;
        }

        else {
            eval.add(ctxt_out, ctxt_out_cache, ctxt_out);
        }
    }
    eval.rescale(ctxt_out);

    return ctxt_out;
}


Ciphertext auxiliaryFtn9(HomEvaluator eval, Context context, vector<vector<Ciphertext>>& ctxtVec, vector<vector<Plaintext>>& ptxtVec, int input_channel) {

    Ciphertext ctxt_out(context);

    for (int inputid = 0; inputid < input_channel; ++inputid) {

        Ciphertext ctxt_out_cache(context);



        int level1 = ctxtVec[inputid][0].getLevel();
        int level2 = ptxtVec[inputid][0].getLevel();
        if (level1 > level2) {
            eval.levelDown(ctxtVec[inputid][0], level2, ctxtVec[inputid][0]);
        }
        else if (level1 < level2) {
            ptxtVec[inputid][0].setLevel(level1);
        }

        eval.multWithoutRescale(ctxtVec[inputid][0], ptxtVec[inputid][0], ctxt_out_cache);

        Ciphertext mult_cache(context);
        for (int i = 1; i < 9; ++i) {
            int level1 = ctxtVec[inputid][i].getLevel();
            int level2 = ptxtVec[inputid][i].getLevel();
            if (level1 > level2) {
                eval.levelDown(ctxtVec[inputid][i], level2, ctxtVec[inputid][i]);
            }
            else if (level1 < level2) {
                ptxtVec[inputid][i].setLevel(level1);
            }
            eval.multWithoutRescale(ctxtVec[inputid][i], ptxtVec[inputid][i], mult_cache);
            eval.add(ctxt_out_cache, mult_cache, ctxt_out_cache);
        }

        if (inputid == 0) {
            ctxt_out = ctxt_out_cache;
        }

        else {
            eval.add(ctxt_out, ctxt_out_cache, ctxt_out);
        }
    }
    eval.rescale(ctxt_out);

    return ctxt_out;
}


std::vector<HEaaN::Ciphertext> Conv(HEaaN::Context context, HEaaN::KeyPack pack,
    HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, int input_channel, int output_channel,
    std::vector<HEaaN::Ciphertext>& ctxt_bundle,
    std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_o) {

    int kernelsize = kernel_o[0][0].size();

    HEaaN::Ciphertext ctxt_init(context);

    std::vector<HEaaN::Ciphertext> ctxt_out_bundle(output_channel, ctxt_init);


    // Convolution

    if (kernelsize == 9) {

        std::vector<std::vector<HEaaN::Ciphertext>> rotated_ctxts_bundle(input_channel, std::vector<HEaaN::Ciphertext>(9, ctxt_init));

        //#pragma omp parallel for collapse(3)
        for (int inputid = 0; inputid < (input_channel); ++inputid) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    eval.leftRotate(ctxt_bundle[inputid], (i - 1) * imgsize * gap + (j - 1) * (gap), rotated_ctxts_bundle[inputid][3 * i + j]);
                }
            }
        }

        //#pragma omp parallel for
        for (int outputid = 0; outputid < output_channel; ++outputid) {
            ctxt_out_bundle[outputid] = auxiliaryFtn9(eval, context, rotated_ctxts_bundle, kernel_o[outputid], input_channel);
        }

        rotated_ctxts_bundle.clear();
        rotated_ctxts_bundle.shrink_to_fit();

    }

    else {

        for (int outputid = 0; outputid < output_channel; ++outputid) {
            ctxt_out_bundle[outputid] = auxiliaryFtn1(eval, context, ctxt_bundle, kernel_o[outputid], input_channel);
        }
    }

    return ctxt_out_bundle;

}


std::vector<HEaaN::Ciphertext> Conv_parallel(HEaaN::Context context, HEaaN::KeyPack pack,
    HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, int input_channel, int output_channel,
    std::vector<HEaaN::Ciphertext>& ctxt_bundle,
    std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_o) {

    int kernelsize = kernel_o[0][0].size();

    HEaaN::Ciphertext ctxt_init(context);

    std::vector<HEaaN::Ciphertext> ctxt_out_bundle(output_channel, ctxt_init);


    // Convolution

    if (kernelsize == 9) {

        std::vector<std::vector<HEaaN::Ciphertext>> rotated_ctxts_bundle(input_channel, std::vector<HEaaN::Ciphertext>(9, ctxt_init));

        #pragma omp parallel for collapse(3)
        for (int inputid = 0; inputid < (input_channel); ++inputid) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    eval.leftRotate(ctxt_bundle[inputid], (i - 1) * imgsize * gap + (j - 1) * gap, rotated_ctxts_bundle[inputid][3 * i + j]);
                }
            }
        }

        #pragma omp parallel for
        for (int outputid = 0; outputid < output_channel; ++outputid) {
            ctxt_out_bundle[outputid] = auxiliaryFtn9(eval, context, rotated_ctxts_bundle, kernel_o[outputid], input_channel);
        }

        rotated_ctxts_bundle.clear();
        rotated_ctxts_bundle.shrink_to_fit();

    }

    else {

        #pragma omp parallel for
        for (int outputid = 0; outputid < output_channel; ++outputid) {
            ctxt_out_bundle[outputid] = auxiliaryFtn1(eval, context, ctxt_bundle, kernel_o[outputid], input_channel);
        }
    }

    return ctxt_out_bundle;

}
