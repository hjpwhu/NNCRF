/*
 * SparseGatedClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseGatedClassifier_H_
#define SRC_SparseGatedClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "NRMat.h"
#include "MyLib.h"
#include "tensor.h"

#include "SparseUniHidderLayer.h"
#include "UniHidderLayer.h"
#include "BiHidderLayer.h"
#include "Utiltensor.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseGatedClassifier {
public:
  SparseGatedClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~SparseGatedClassifier() {

  }

public:
  Tensor<xpu, 2, double> _wordEmb;
  Tensor<xpu, 2, double> _grad_wordEmb;
  Tensor<xpu, 2, double> _eg2_wordEmb;
  Tensor<xpu, 2, double> _ft_wordEmb;
  hash_set<int> _indexers;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;
  bool _b_wordEmb_finetune;

  int _inputsize, _token_representation_size;
  int _inputwindow;
  UniHidderLayer<xpu> _olayer_linear;

  //Gated Recursive Unit
  UniHidderLayer<xpu> _atom_reset_input;
  UniHidderLayer<xpu> _atom_gate_input;
  BiHidderLayer<xpu> _atom_hidden_input;

  int _atom_composition_layer_num;

  int _labelSize;

  Metric _eval;

  double _dropOut;

  SparseUniHidderLayer<xpu> _sparselayer_linear;
  int _linearfeatSize;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, int labelSize, int atom_composition_layer_num, int linearfeatSize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = labelSize;
    _atom_composition_layer_num = atom_composition_layer_num;
    if (atom_composition_layer_num > _wordcontext)
      _atom_composition_layer_num = _wordcontext;

    _wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _grad_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _eg2_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _ft_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 1.0);
    assign(_wordEmb, wordEmb);
    for (int idx = 0; idx < _wordSize; idx++) {
      norm2one(_wordEmb, idx);
    }

    _token_representation_size = _wordDim;
    _inputsize = _token_representation_size * _wordwindow;
    _inputwindow = _wordwindow;

    for (int idx = 1; idx < _atom_composition_layer_num; idx++) {
      _inputsize += _token_representation_size * (_wordwindow - idx);
      _inputwindow = _inputwindow + (_wordwindow - idx);
    }
    
    _atom_reset_input.initial(2 * _token_representation_size, 2 * _token_representation_size, false, 101, 1);
    _atom_gate_input.initial(3 * _token_representation_size, 3 * _token_representation_size, false, 102, 2);
    _atom_hidden_input.initial(_token_representation_size, _token_representation_size, _token_representation_size, false, 103, 0);

    _olayer_linear.initial(_labelSize, _inputsize, true, 4, 2);

    _linearfeatSize = linearfeatSize;
    _sparselayer_linear.initial(_labelSize, _linearfeatSize, false, 5, 2);

  }

  inline void release() {
    FreeSpace(&_wordEmb);
    FreeSpace(&_grad_wordEmb);
    FreeSpace(&_eg2_wordEmb);
    FreeSpace(&_ft_wordEmb);
    _olayer_linear.release();
    _sparselayer_linear.release();

    _atom_reset_input.release();
    _atom_gate_input.release();
    _atom_hidden_input.release();

  }

  inline double process(const vector<Example>& examples, int iter) {
    _eval.reset();
    _indexers.clear();

    int example_num = examples.size();
    double cost = 0.0;
    int offset = 0;
    int gru_end = _inputwindow;
    int curlayer, curlayerSize, leftchild, rightchild;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      vector<vector<int> > linear_features(seq_size);

      Tensor<xpu, 2, double> input[seq_size], inputLoss[seq_size];
      Tensor<xpu, 2, double> denseout[seq_size], denseoutLoss[seq_size];
      Tensor<xpu, 2, double> sparseout[seq_size], sparseoutLoss[seq_size];
      Tensor<xpu, 2, double> output[seq_size], outputLoss[seq_size];
      Tensor<xpu, 2, double> scores[seq_size];

      Tensor<xpu, 2, double> wordprime[seq_size], wordprimeLoss[seq_size], wordprimeMask[seq_size];
      Tensor<xpu, 2, double> wordrepresent[seq_size], wordrepresentLoss[seq_size];
      
      //GRU
      Tensor<xpu, 2, double> inputcontext[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontextLoss[seq_size][_inputwindow];

      Tensor<xpu, 2, double> inputcontext_reset_left[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_reset_leftLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_reset_right[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_reset_rightLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_reset[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_resetLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_afterreset_left[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_afterreset_leftLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_afterreset_right[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_afterreset_rightLoss[seq_size][_inputwindow];

      Tensor<xpu, 2, double> inputcontext_gate_left[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_leftLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_right[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_rightLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_current[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_currentLoss[seq_size][_inputwindow];

      Tensor<xpu, 2, double> inputcontext_gate_tmp[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_tmpLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gateLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_pool[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_poolLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_norm[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_gate_normLoss[seq_size][_inputwindow];

      Tensor<xpu, 2, double> inputcontext_current[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_currentLoss[seq_size][_inputwindow];

      Tensor<xpu, 2, double> inputcontext_lr[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_lrLoss[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_lrc[seq_size][_inputwindow];
      Tensor<xpu, 2, double> inputcontext_lrcLoss[seq_size][_inputwindow];
      //end gru

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        for (int idy = 0; idy < _inputwindow; idy++) {
          inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontextLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

          inputcontext_current[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_currentLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_lr[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
          inputcontext_lrLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
          inputcontext_lrc[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_lrcLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);

          inputcontext_reset[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
          inputcontext_resetLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
          inputcontext_reset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_reset_leftLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_reset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_reset_rightLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

          inputcontext_afterreset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_afterreset_leftLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_afterreset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_afterreset_rightLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

          inputcontext_gate_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_gate_leftLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_gate_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_gate_rightLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_gate_current[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontext_gate_currentLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

          inputcontext_gate_tmp[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gate_tmpLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gate[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gateLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gate_pool[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gate_poolLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gate_norm[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
          inputcontext_gate_normLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        }
        wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        wordprimeMask[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 1.0);
        wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        wordrepresentLoss[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);    
        input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        sparseoutLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        denseoutLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        scores[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      }

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);
        
        linear_features[idx].clear();
        for (int idy = 0; idy < feature.linear_features.size(); idy++) {
          if (1.0 * rand() / RAND_MAX >= _dropOut) {
            linear_features[idx].push_back(feature.linear_features[idy]);
          }
        }
        
        const vector<int>& words = feature.words;
        offset = words[0];
        wordprime[idx][0] = _wordEmb[offset] / _ft_wordEmb[offset];   
        
        //dropout
        for (int j = 0; j < _wordDim; j++) {
          if (1.0 * rand() / RAND_MAX >= _dropOut) {
            wordprimeMask[idx][0][j] = 1.0;
          } else {
            wordprimeMask[idx][0][j] = 0.0;
          }
        }        
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];            
      }
      
      
      
      for (int idx = 0; idx < seq_size; idx++) {
      	wordrepresent[idx] += wordprime[idx];
      }


      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = -_wordcontext; i <= _wordcontext; i++) {
          if(idx+i >= 0 && idx+i < seq_size)
          	inputcontext[idx][i+_wordcontext] += wordrepresent[idx+i];
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {
        //gru
        curlayer = 1;
        curlayerSize = _wordwindow - 1;
        offset = _wordwindow;
        while (curlayer < _atom_composition_layer_num) {
          for (int i = 0; i < curlayerSize; i++) {
            leftchild = offset - curlayerSize - 1;
            rightchild = leftchild + 1;
            //reset
            concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_lr[idx][offset]);
            _atom_reset_input.ComputeForwardScore(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset]);
            unconcat(inputcontext_reset_left[idx][offset], inputcontext_reset_right[idx][offset], inputcontext_reset[idx][offset]);
            //current input
            inputcontext_afterreset_left[idx][offset] = inputcontext_reset_left[idx][offset] * inputcontext[idx][leftchild];
            inputcontext_afterreset_right[idx][offset] = inputcontext_reset_right[idx][offset] * inputcontext[idx][rightchild];
            _atom_hidden_input.ComputeForwardScore(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
                inputcontext_current[idx][offset]);

            //gate
            concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_current[idx][offset], inputcontext_lrc[idx][offset]);
            //gateleft
            _atom_gate_input.ComputeForwardScore(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset]);
            inputcontext_gate[idx][offset] = F<nl_exp>(inputcontext_gate_tmp[idx][offset]);
            for (int j = 0; j < _wordDim; j++) {
              double sum = inputcontext_gate[idx][offset][0][j] + inputcontext_gate[idx][offset][0][_wordDim + j]
                  + inputcontext_gate[idx][offset][0][2 * _wordDim + j];
              sum = 1.0 / sum;
              inputcontext_gate_pool[idx][offset][0][j] = sum;
              inputcontext_gate_pool[idx][offset][0][_wordDim + j] = sum;
              inputcontext_gate_pool[idx][offset][0][2 * _wordDim + j] = sum;
            }
            inputcontext_gate_norm[idx][offset] = inputcontext_gate_pool[idx][offset] * inputcontext_gate[idx][offset];
            unconcat(inputcontext_gate_left[idx][offset], inputcontext_gate_right[idx][offset], inputcontext_gate_current[idx][offset],
                inputcontext_gate_norm[idx][offset]);

            //for(int j = 0; j <_wordDim; j++)
            //{
            //std::cout << inputcontext_gate_left[idx][offset][0][j] << " " << inputcontext_gate_right[idx][offset][0][j] << " " << inputcontext_gate_current[idx][offset][0][j] << std::endl;
            //}
            //std::cout << std::endl;

            //current hidden
            inputcontext[idx][offset] = inputcontext_gate_left[idx][offset] * inputcontext[idx][leftchild]
                + inputcontext_gate_right[idx][offset] * inputcontext[idx][rightchild]
                + inputcontext_gate_current[idx][offset] * inputcontext_current[idx][offset];
            offset++;
          }
          curlayer++;
          curlayerSize--;
        }
        //end gru
        if (offset != gru_end) {
          std::cout << "error forward computation here" << std::endl;
        }

        offset = 0;
        for (int i = 0; i < _inputwindow; i++) {
          for (int j = 0; j < _wordDim; j++) {
            input[idx][0][offset] = inputcontext[idx][i][0][j];
            offset++;
          }
        }

      }

      for (int idx = 0; idx < seq_size; idx++) {
        _olayer_linear.ComputeForwardScore(input[idx], denseout[idx]);
        _sparselayer_linear.ComputeForwardScore(linear_features[idx], sparseout[idx]);
        output[idx] = denseout[idx] + sparseout[idx];
      }

      // get delta for each output
      for (int idx = 0; idx < seq_size; idx++) {
        int optLabel = -1;
        for (int i = 0; i < _labelSize; ++i) {
          if (example.m_labels[idx][i] >= 0) {
            if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
              optLabel = i;
          }
        }

        double sum1 = 0.0;
        double sum2 = 0.0;
        double maxScore = output[idx][0][optLabel];
        for (int i = 0; i < _labelSize; ++i) {
          scores[idx][0][i] = -1e10;
          if (example.m_labels[idx][i] >= 0) {
            scores[idx][0][i] = exp(output[idx][0][i] - maxScore);
            if (example.m_labels[idx][i] == 1)
              sum1 += scores[idx][0][i];
            sum2 += scores[idx][0][i];
          }
        }
        cost += (log(sum2) - log(sum1)) / (example_num * seq_size);
        if (example.m_labels[idx][optLabel] == 1)
          _eval.correct_label_count++;
        _eval.overall_label_count++;

        for (int i = 0; i < _labelSize; ++i) {
          outputLoss[idx][0][i] = 0.0;
          if (example.m_labels[idx][i] >= 0) {
            outputLoss[idx][0][i] = (scores[idx][0][i] / sum2 - example.m_labels[idx][i]) / (example_num * seq_size);
          }
        }
      }

      // loss backward propagation
      for (int idx = 0; idx < seq_size; idx++) {
        _olayer_linear.ComputeBackwardLoss(input[idx], denseout[idx], outputLoss[idx], inputLoss[idx]);
        _sparselayer_linear.ComputeBackwardLoss(linear_features[idx], sparseout[idx], outputLoss[idx]);

        offset = 0;
        for (int i = 0; i < _inputwindow; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            inputcontextLoss[idx][i][0][j] = inputLoss[idx][0][offset];
            offset++;
          }
        }

        //gru back-propagation
        curlayer = _atom_composition_layer_num - 1;
        curlayerSize = _wordwindow + 1 - _atom_composition_layer_num;
        offset = gru_end - 1;
        while (curlayer > 0) {
          for (int i = curlayerSize - 1; i >= 0; i--) {
            leftchild = offset - curlayerSize - 1;
            rightchild = leftchild + 1;
            //current hidden
            inputcontextLoss[idx][leftchild] += inputcontextLoss[idx][offset] * inputcontext_gate_left[idx][offset];
            inputcontextLoss[idx][rightchild] += inputcontextLoss[idx][offset] * inputcontext_gate_right[idx][offset];
            inputcontext_currentLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext_gate_current[idx][offset];

            inputcontext_gate_leftLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext[idx][leftchild];
            inputcontext_gate_rightLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext[idx][rightchild];
            inputcontext_gate_currentLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext_current[idx][offset];

            //gate
            concat(inputcontext_gate_leftLoss[idx][offset], inputcontext_gate_rightLoss[idx][offset], inputcontext_gate_currentLoss[idx][offset],
                inputcontext_gate_normLoss[idx][offset]);
            inputcontext_gate_poolLoss[idx][offset] += inputcontext_gate_normLoss[idx][offset] * inputcontext_gate[idx][offset];
            inputcontext_gateLoss[idx][offset] += inputcontext_gate_normLoss[idx][offset] * inputcontext_gate_pool[idx][offset];
            for (int j = 0; j < _wordDim; j++) {
              double sumLoss = inputcontext_gate_poolLoss[idx][offset][0][j] + inputcontext_gate_poolLoss[idx][offset][0][_wordDim + j]
                  + inputcontext_gate_poolLoss[idx][offset][0][2 * _wordDim + j];
              sumLoss = -sumLoss * inputcontext_gate_pool[idx][offset][0][j] * inputcontext_gate_pool[idx][offset][0][j];
              inputcontext_gateLoss[idx][offset][0][j] += sumLoss;
              inputcontext_gateLoss[idx][offset][0][_wordDim + j] += sumLoss;
              inputcontext_gateLoss[idx][offset][0][2 * _wordDim + j] += sumLoss;
            }
            inputcontext_gate_tmpLoss[idx][offset] += inputcontext_gateLoss[idx][offset] * inputcontext_gate[idx][offset];
            _atom_gate_input.ComputeBackwardLoss(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset], inputcontext_gate_tmpLoss[idx][offset],
                inputcontext_lrcLoss[idx][offset]);
            unconcat(inputcontextLoss[idx][leftchild], inputcontextLoss[idx][rightchild], inputcontext_currentLoss[idx][offset],
                inputcontext_lrcLoss[idx][offset]);

            //current input
            _atom_hidden_input.ComputeBackwardLoss(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
                inputcontext_current[idx][offset], inputcontext_currentLoss[idx][offset], inputcontext_afterreset_leftLoss[idx][offset],
                inputcontext_afterreset_rightLoss[idx][offset]);

            inputcontext_reset_rightLoss[idx][offset] += inputcontext_afterreset_rightLoss[idx][offset] * inputcontext[idx][rightchild];
            inputcontextLoss[idx][rightchild] += inputcontext_afterreset_rightLoss[idx][offset] * inputcontext_reset_right[idx][offset];
            inputcontext_reset_leftLoss[idx][offset] += inputcontext_afterreset_leftLoss[idx][offset] * inputcontext[idx][leftchild];
            inputcontextLoss[idx][leftchild] += inputcontext_afterreset_leftLoss[idx][offset] * inputcontext_reset_left[idx][offset];

            //reset
            concat(inputcontext_reset_leftLoss[idx][offset], inputcontext_reset_rightLoss[idx][offset], inputcontext_resetLoss[idx][offset]);
            _atom_reset_input.ComputeBackwardLoss(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset], inputcontext_resetLoss[idx][offset],
                inputcontext_lrLoss[idx][offset]);
            unconcat(inputcontextLoss[idx][leftchild], inputcontextLoss[idx][rightchild], inputcontext_lrLoss[idx][offset]);

            offset--;
          }
          curlayer--;
          curlayerSize++;
        }

        //end gru
        if (offset != _wordwindow - 1) {
          std::cout << "error back-propagation here" << std::endl;
        }

      }
      
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = -_wordcontext; i <= _wordcontext; i++) {
          if(idx+i >= 0 && idx+i < seq_size)
          	wordrepresentLoss[idx+i] += inputcontextLoss[idx][i+_wordcontext];
        }
      }
      
      for (int idx = 0; idx < seq_size; idx++) {
      	wordprimeLoss[idx] += wordrepresentLoss[idx];
      }

      if (_b_wordEmb_finetune) {
      	for (int idx = 0; idx < seq_size; idx++) {      	        
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          offset = words[0];
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _grad_wordEmb[offset] += wordprimeLoss[idx][0];
          _indexers.insert(offset);
        }
      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
        for (int idy = 0; idy < _inputwindow; idy++) {
          FreeSpace(&(inputcontext[idx][idy]));
          FreeSpace(&(inputcontextLoss[idx][idy]));

          FreeSpace(&(inputcontext_current[idx][idy]));
          FreeSpace(&(inputcontext_currentLoss[idx][idy]));
          FreeSpace(&(inputcontext_lr[idx][idy]));
          FreeSpace(&(inputcontext_lrLoss[idx][idy]));
          FreeSpace(&(inputcontext_lrc[idx][idy]));
          FreeSpace(&(inputcontext_lrcLoss[idx][idy]));

          FreeSpace(&(inputcontext_reset[idx][idy]));
          FreeSpace(&(inputcontext_resetLoss[idx][idy]));
          FreeSpace(&(inputcontext_reset_left[idx][idy]));
          FreeSpace(&(inputcontext_reset_leftLoss[idx][idy]));
          FreeSpace(&(inputcontext_reset_right[idx][idy]));
          FreeSpace(&(inputcontext_reset_rightLoss[idx][idy]));

          FreeSpace(&(inputcontext_afterreset_left[idx][idy]));
          FreeSpace(&(inputcontext_afterreset_leftLoss[idx][idy]));
          FreeSpace(&(inputcontext_afterreset_right[idx][idy]));
          FreeSpace(&(inputcontext_afterreset_rightLoss[idx][idy]));

          FreeSpace(&(inputcontext_gate_left[idx][idy]));
          FreeSpace(&(inputcontext_gate_leftLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_right[idx][idy]));
          FreeSpace(&(inputcontext_gate_rightLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_current[idx][idy]));
          FreeSpace(&(inputcontext_gate_currentLoss[idx][idy]));

          FreeSpace(&(inputcontext_gate_tmp[idx][idy]));
          FreeSpace(&(inputcontext_gate_tmpLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate[idx][idy]));
          FreeSpace(&(inputcontext_gateLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_pool[idx][idy]));
          FreeSpace(&(inputcontext_gate_poolLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_norm[idx][idy]));
          FreeSpace(&(inputcontext_gate_normLoss[idx][idy]));
        }
        FreeSpace(&(wordprime[idx]));
        FreeSpace(&(wordprimeLoss[idx]));
        FreeSpace(&(wordprimeMask[idx]));
        FreeSpace(&(wordrepresent[idx]));
        FreeSpace(&(wordrepresentLoss[idx]));

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(sparseout[idx]));
        FreeSpace(&(sparseoutLoss[idx]));
        FreeSpace(&(denseout[idx]));
        FreeSpace(&(denseoutLoss[idx]));
        FreeSpace(&(output[idx]));
        FreeSpace(&(outputLoss[idx]));
        FreeSpace(&(scores[idx]));
      }
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, vector<int>& results) {
    int seq_size = features.size();
    int offset = 0;
    int gru_end = _inputwindow;
    int curlayer, curlayerSize, leftchild, rightchild;

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> sparseout[seq_size];
    Tensor<xpu, 2, double> denseout[seq_size];
    Tensor<xpu, 2, double> output[seq_size];
    Tensor<xpu, 2, double> wordprime[seq_size];
    Tensor<xpu, 2, double> wordrepresent[seq_size];
    
    //GRU
    Tensor<xpu, 2, double> inputcontext[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_reset_left[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_reset_right[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_reset[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_afterreset_left[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_afterreset_right[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_gate_left[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_right[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_current[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_gate_tmp[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_pool[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_norm[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_current[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_lr[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_lrc[seq_size][_inputwindow];
    //end gru

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputwindow; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_current[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_lr[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
        inputcontext_lrc[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);

        inputcontext_reset[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
        inputcontext_reset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_reset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_afterreset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_afterreset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_gate_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_gate_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_gate_current[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_gate_tmp[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        inputcontext_gate[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        inputcontext_gate_pool[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        inputcontext_gate_norm[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
      }
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

        const vector<int>& words = feature.words;
        offset = words[0];
        wordprime[idx][0] = _wordEmb[offset] / _ft_wordEmb[offset];         
      }
            
      for (int idx = 0; idx < seq_size; idx++) {
      	wordrepresent[idx] += wordprime[idx];
      }


      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = -_wordcontext; i <= _wordcontext; i++) {
          if(idx+i >= 0 && idx+i < seq_size)
          	inputcontext[idx][i+_wordcontext] += wordrepresent[idx+i];
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {

      //gru
      curlayer = 1;
      curlayerSize = _wordwindow - 1;
      offset = _wordwindow;
      while (curlayer < _atom_composition_layer_num) {
        for (int i = 0; i < curlayerSize; i++) {
          leftchild = offset - curlayerSize - 1;
          rightchild = leftchild + 1;
          //reset
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_lr[idx][offset]);
          _atom_reset_input.ComputeForwardScore(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset]);
          unconcat(inputcontext_reset_left[idx][offset], inputcontext_reset_right[idx][offset], inputcontext_reset[idx][offset]);
          //current input
          inputcontext_afterreset_left[idx][offset] = inputcontext_reset_left[idx][offset] * inputcontext[idx][leftchild];
          inputcontext_afterreset_right[idx][offset] = inputcontext_reset_right[idx][offset] * inputcontext[idx][rightchild];
          _atom_hidden_input.ComputeForwardScore(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
              inputcontext_current[idx][offset]);

          //gate
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_current[idx][offset], inputcontext_lrc[idx][offset]);
          //gateleft
          _atom_gate_input.ComputeForwardScore(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset]);
          inputcontext_gate[idx][offset] = F<nl_exp>(inputcontext_gate_tmp[idx][offset]);
          for (int j = 0; j < _wordDim; j++) {
            double sum = inputcontext_gate[idx][offset][0][j] + inputcontext_gate[idx][offset][0][_wordDim + j]
                + inputcontext_gate[idx][offset][0][2 * _wordDim + j];
            sum = 1.0 / sum;
            inputcontext_gate_pool[idx][offset][0][j] = sum;
            inputcontext_gate_pool[idx][offset][0][_wordDim + j] = sum;
            inputcontext_gate_pool[idx][offset][0][2 * _wordDim + j] = sum;
          }
          inputcontext_gate_norm[idx][offset] = inputcontext_gate_pool[idx][offset] * inputcontext_gate[idx][offset];
          unconcat(inputcontext_gate_left[idx][offset], inputcontext_gate_right[idx][offset], inputcontext_gate_current[idx][offset],
              inputcontext_gate_norm[idx][offset]);

          //current hidden
          inputcontext[idx][offset] = inputcontext_gate_left[idx][offset] * inputcontext[idx][leftchild]
              + inputcontext_gate_right[idx][offset] * inputcontext[idx][rightchild]
              + inputcontext_gate_current[idx][offset] * inputcontext_current[idx][offset];
          offset++;
        }
        curlayer++;
        curlayerSize--;
      }
      //end gru

      offset = 0;
      for (int i = 0; i < _inputwindow; i++) {
        for (int j = 0; j < _token_representation_size; j++) {
          input[idx][0][offset] = inputcontext[idx][i][0][j];
          offset++;
        }
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      _olayer_linear.ComputeForwardScore(input[idx], denseout[idx]);
      _sparselayer_linear.ComputeForwardScore(features[idx].linear_features, sparseout[idx]);
      output[idx] = denseout[idx] + sparseout[idx];
    }

    // decode algorithm
    results.resize(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      int optLabel = -1;
      for (int i = 0; i < _labelSize; ++i) {
        if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
          optLabel = i;
      }
      results[idx] = optLabel;
    }

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputwindow; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));

        FreeSpace(&(inputcontext_current[idx][idy]));
        FreeSpace(&(inputcontext_lr[idx][idy]));
        FreeSpace(&(inputcontext_lrc[idx][idy]));

        FreeSpace(&(inputcontext_reset[idx][idy]));
        FreeSpace(&(inputcontext_reset_left[idx][idy]));
        FreeSpace(&(inputcontext_reset_right[idx][idy]));

        FreeSpace(&(inputcontext_afterreset_left[idx][idy]));
        FreeSpace(&(inputcontext_afterreset_right[idx][idy]));

        FreeSpace(&(inputcontext_gate_left[idx][idy]));
        FreeSpace(&(inputcontext_gate_right[idx][idy]));
        FreeSpace(&(inputcontext_gate_current[idx][idy]));

        FreeSpace(&(inputcontext_gate_tmp[idx][idy]));
        FreeSpace(&(inputcontext_gate[idx][idy]));
        FreeSpace(&(inputcontext_gate_pool[idx][idy]));
        FreeSpace(&(inputcontext_gate_norm[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
      FreeSpace(&(output[idx]));
    }
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;
    int gru_end = _inputwindow;
    int curlayer, curlayerSize, leftchild, rightchild;

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> sparseout[seq_size];
    Tensor<xpu, 2, double> denseout[seq_size];
    Tensor<xpu, 2, double> output[seq_size];
    Tensor<xpu, 2, double> scores[seq_size];
    Tensor<xpu, 2, double> wordprime[seq_size];
    Tensor<xpu, 2, double> wordrepresent[seq_size];
    
    //GRU
    Tensor<xpu, 2, double> inputcontext[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_reset_left[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_reset_right[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_reset[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_afterreset_left[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_afterreset_right[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_gate_left[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_right[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_current[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_gate_tmp[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_pool[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_gate_norm[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_current[seq_size][_inputwindow];

    Tensor<xpu, 2, double> inputcontext_lr[seq_size][_inputwindow];
    Tensor<xpu, 2, double> inputcontext_lrc[seq_size][_inputwindow];
    //end gru

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputwindow; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_current[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_lr[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
        inputcontext_lrc[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);

        inputcontext_reset[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _token_representation_size), 0.0);
        inputcontext_reset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_reset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_afterreset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_afterreset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_gate_left[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_gate_right[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        inputcontext_gate_current[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

        inputcontext_gate_tmp[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        inputcontext_gate[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        inputcontext_gate_pool[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
        inputcontext_gate_norm[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _token_representation_size), 0.0);
      }
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      scores[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    //input setting, and linear setting
   for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      //linear features should not be dropped out

        const vector<int>& words = feature.words;
        offset = words[0];
        wordprime[idx][0] = _wordEmb[offset] / _ft_wordEmb[offset];         
      }
            
      for (int idx = 0; idx < seq_size; idx++) {
      	wordrepresent[idx] += wordprime[idx];
      }


      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = -_wordcontext; i <= _wordcontext; i++) {
          if(idx+i >= 0 && idx+i < seq_size)
          	inputcontext[idx][i+_wordcontext] += wordrepresent[idx+i];
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {

      //gru
      curlayer = 1;
      curlayerSize = _wordwindow - 1;
      offset = _wordwindow;

      while (curlayer < _atom_composition_layer_num) {
        for (int i = 0; i < curlayerSize; i++) {
          leftchild = offset - curlayerSize - 1;
          rightchild = leftchild + 1;
          //reset
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_lr[idx][offset]);
          _atom_reset_input.ComputeForwardScore(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset]);
          unconcat(inputcontext_reset_left[idx][offset], inputcontext_reset_right[idx][offset], inputcontext_reset[idx][offset]);
          //current input
          inputcontext_afterreset_left[idx][offset] = inputcontext_reset_left[idx][offset] * inputcontext[idx][leftchild];
          inputcontext_afterreset_right[idx][offset] = inputcontext_reset_right[idx][offset] * inputcontext[idx][rightchild];
          _atom_hidden_input.ComputeForwardScore(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
              inputcontext_current[idx][offset]);

          //gate
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_current[idx][offset], inputcontext_lrc[idx][offset]);
          //gateleft
          _atom_gate_input.ComputeForwardScore(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset]);
          inputcontext_gate[idx][offset] = F<nl_exp>(inputcontext_gate_tmp[idx][offset]);
          for (int j = 0; j < _wordDim; j++) {
            double sum = inputcontext_gate[idx][offset][0][j] + inputcontext_gate[idx][offset][0][_wordDim + j]
                + inputcontext_gate[idx][offset][0][2 * _wordDim + j];
            sum = 1.0 / sum;
            inputcontext_gate_pool[idx][offset][0][j] = sum;
            inputcontext_gate_pool[idx][offset][0][_wordDim + j] = sum;
            inputcontext_gate_pool[idx][offset][0][2 * _wordDim + j] = sum;
          }
          inputcontext_gate_norm[idx][offset] = inputcontext_gate_pool[idx][offset] * inputcontext_gate[idx][offset];
          unconcat(inputcontext_gate_left[idx][offset], inputcontext_gate_right[idx][offset], inputcontext_gate_current[idx][offset],
              inputcontext_gate_norm[idx][offset]);

          //current hidden
          inputcontext[idx][offset] = inputcontext_gate_left[idx][offset] * inputcontext[idx][leftchild]
              + inputcontext_gate_right[idx][offset] * inputcontext[idx][rightchild]
              + inputcontext_gate_current[idx][offset] * inputcontext_current[idx][offset];
          offset++;
        }
        curlayer++;
        curlayerSize--;
      }
      //end gru

      offset = 0;
      for (int i = 0; i < _inputwindow; i++) {
        for (int j = 0; j < _token_representation_size; j++) {
          input[idx][0][offset] = inputcontext[idx][i][0][j];
          offset++;
        }
      }

    }

    for (int idx = 0; idx < seq_size; idx++) {
      _olayer_linear.ComputeForwardScore(input[idx], denseout[idx]);
      _sparselayer_linear.ComputeForwardScore(example.m_features[idx].linear_features, sparseout[idx]);
      output[idx] = denseout[idx] + sparseout[idx];
    }

    // get delta for each output
    double cost = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      int optLabel = -1;
      for (int i = 0; i < _labelSize; ++i) {
        if (example.m_labels[idx][i] >= 0) {
          if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
            optLabel = i;
        }
      }

      double sum1 = 0.0;
      double sum2 = 0.0;
      double maxScore = output[idx][0][optLabel];
      for (int i = 0; i < _labelSize; ++i) {
        scores[idx][0][i] = -1e10;
        if (example.m_labels[idx][i] >= 0) {
          scores[idx][0][i] = exp(output[idx][0][i] - maxScore);
          if (example.m_labels[idx][i] == 1)
            sum1 += scores[idx][0][i];
          sum2 += scores[idx][0][i];
        }
      }
      //std::cout << sum2 << " " << sum1 << std::endl;
      cost += (log(sum2) - log(sum1));
    }

    cost = cost / seq_size;

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputwindow; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));

        FreeSpace(&(inputcontext_current[idx][idy]));
        FreeSpace(&(inputcontext_lr[idx][idy]));
        FreeSpace(&(inputcontext_lrc[idx][idy]));

        FreeSpace(&(inputcontext_reset[idx][idy]));
        FreeSpace(&(inputcontext_reset_left[idx][idy]));
        FreeSpace(&(inputcontext_reset_right[idx][idy]));

        FreeSpace(&(inputcontext_afterreset_left[idx][idy]));
        FreeSpace(&(inputcontext_afterreset_right[idx][idy]));

        FreeSpace(&(inputcontext_gate_left[idx][idy]));
        FreeSpace(&(inputcontext_gate_right[idx][idy]));
        FreeSpace(&(inputcontext_gate_current[idx][idy]));

        FreeSpace(&(inputcontext_gate_tmp[idx][idy]));
        FreeSpace(&(inputcontext_gate[idx][idy]));
        FreeSpace(&(inputcontext_gate_pool[idx][idy]));
        FreeSpace(&(inputcontext_gate_norm[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
      FreeSpace(&(output[idx]));
      FreeSpace(&(scores[idx]));
    }
    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sparselayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _atom_reset_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _atom_gate_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _atom_hidden_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    if (_b_wordEmb_finetune) {
      static hash_set<int>::iterator it;
      Tensor<xpu, 1, double> _grad_wordEmb_ij = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> tmp_normaize_alpha = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> tmp_alpha = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> _ft_wordEmb_ij = NewTensor<xpu>(Shape1(_wordDim), 0.0);

      for (it = _indexers.begin(); it != _indexers.end(); ++it) {
        int index = *it;
        _grad_wordEmb_ij = _grad_wordEmb[index] + nnRegular * _wordEmb[index] / _ft_wordEmb[index];
        _eg2_wordEmb[index] += _grad_wordEmb_ij * _grad_wordEmb_ij;
        tmp_normaize_alpha = F<nl_sqrt>(_eg2_wordEmb[index] + adaEps);
        tmp_alpha = adaAlpha / tmp_normaize_alpha;
        _ft_wordEmb_ij = _ft_wordEmb[index] * tmp_alpha * nnRegular;
        _ft_wordEmb[index] -= _ft_wordEmb_ij;
        _wordEmb[index] -= tmp_alpha * _grad_wordEmb[index] / _ft_wordEmb[index];
        _grad_wordEmb[index] = 0.0;
      }

      FreeSpace(&_grad_wordEmb_ij);
      FreeSpace(&tmp_normaize_alpha);
      FreeSpace(&tmp_alpha);
      FreeSpace(&_ft_wordEmb_ij);
    }
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double> Wd, Tensor<xpu, 2, double> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    double orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    double lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    double lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    double mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    double computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double> Wd, Tensor<xpu, 2, double> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    double orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    double lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    double lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    double mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    double computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(examples, _sparselayer_linear._W, _sparselayer_linear._gradW, "_sparselayer_linear._W", iter, _sparselayer_linear._indexers, false);
    checkgrad(examples, _sparselayer_linear._b, _sparselayer_linear._gradb, "_sparselayer_linear._b", iter);

    checkgrad(examples, _atom_reset_input._W, _atom_reset_input._gradW, "_atom_reset_input._W", iter);
    checkgrad(examples, _atom_reset_input._b, _atom_reset_input._gradb, "_atom_reset_input._b", iter);

    checkgrad(examples, _atom_gate_input._W, _atom_gate_input._gradW, "_atom_gate_input._W", iter);
    checkgrad(examples, _atom_gate_input._b, _atom_gate_input._gradb, "_atom_gate_input._b", iter);

    checkgrad(examples, _atom_hidden_input._WL, _atom_hidden_input._gradWL, "_atom_hidden_input._WL", iter);
    checkgrad(examples, _atom_hidden_input._WR, _atom_hidden_input._gradWR, "_atom_hidden_input._WR", iter);
    checkgrad(examples, _atom_hidden_input._b, _atom_hidden_input._gradb, "_atom_hidden_input._b", iter);

    checkgrad(examples, _wordEmb, _grad_wordEmb, "_wordEmb", iter, _indexers);

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(double dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _b_wordEmb_finetune = b_wordEmb_finetune;
  }

};

#endif /* SRC_SparseGatedClassifier_H_ */
