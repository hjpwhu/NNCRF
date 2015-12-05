/*
 * SparseLSTMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseLSTMClassifier_H_
#define SRC_SparseLSTMClassifier_H_

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
#include "TriHidderLayer.h"
#include "Utiltensor.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseLSTMClassifier {
public:
  SparseLSTMClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~SparseLSTMClassifier() {

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

  TriHidderLayer<xpu> _lstm_left_output;
  TriHidderLayer<xpu> _lstm_left_input;
  TriHidderLayer<xpu> _lstm_left_forget;
  BiHidderLayer<xpu> _lstm_left_cell;

  TriHidderLayer<xpu> _lstm_right_output;
  TriHidderLayer<xpu> _lstm_right_input;
  TriHidderLayer<xpu> _lstm_right_forget;
  BiHidderLayer<xpu> _lstm_right_cell;

  vector<TriHidderLayer<xpu> > _lstm_middle_left_output;
  vector<TriHidderLayer<xpu> > _lstm_middle_left_input;
  vector<TriHidderLayer<xpu> > _lstm_middle_left_forget;
  vector<BiHidderLayer<xpu> > _lstm_middle_left_cell;

  vector<TriHidderLayer<xpu> > _lstm_middle_right_output;
  vector<TriHidderLayer<xpu> > _lstm_middle_right_input;
  vector<TriHidderLayer<xpu> > _lstm_middle_right_forget;
  vector<BiHidderLayer<xpu> > _lstm_middle_right_cell;

  int _rnnHiddenSize;
  int _rnnMidLayers;

  int _hiddensize;
  int _inputsize, _token_representation_size;
  UniHidderLayer<xpu> _olayer_linear;
  UniHidderLayer<xpu> _tanh_project;

  int _labelSize;

  Metric _eval;

  double _dropOut;

  SparseUniHidderLayer<xpu> _sparselayer_linear;
  int _linearfeatSize;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, int labelSize, int hiddensize, int rnnHiddenSize, int rnnMidLayers, int linearfeatSize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = labelSize;
    _hiddensize = hiddensize;
    _token_representation_size = _wordDim;
    _inputsize = _wordwindow * _token_representation_size;

    _wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _grad_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _eg2_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _ft_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 1.0);
    assign(_wordEmb, wordEmb);
    for (int idx = 0; idx < _wordSize; idx++) {
      norm2one(_wordEmb, idx);
    }

    _rnnHiddenSize = rnnHiddenSize;
    _rnnMidLayers = rnnMidLayers;
    assert(_rnnMidLayers >= 0);

    _lstm_left_output.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);
    _lstm_left_input.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);
    _lstm_left_forget.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);
    _lstm_left_cell.initial(_rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);

    _lstm_right_output.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);
    _lstm_right_input.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);
    _lstm_right_forget.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);
    _lstm_right_cell.initial(_rnnHiddenSize, _rnnHiddenSize, _inputsize, true, 3, 0);

    for (int idx = 0; idx < _rnnMidLayers; idx++) {
      TriHidderLayer<xpu> lstm_middle_left_output, lstm_middle_left_input, lstm_middle_left_forget;
      BiHidderLayer<xpu> lstm_middle_left_cell;
      lstm_middle_left_output.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      lstm_middle_left_input.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      lstm_middle_left_forget.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      lstm_middle_left_cell.initial(_rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      _lstm_middle_left_output.push_back(lstm_middle_left_output);
      _lstm_middle_left_input.push_back(lstm_middle_left_input);
      _lstm_middle_left_forget.push_back(lstm_middle_left_forget);
      _lstm_middle_left_cell.push_back(lstm_middle_left_cell);

      TriHidderLayer<xpu> lstm_middle_right_output, lstm_middle_right_input, lstm_middle_right_forget;
      BiHidderLayer<xpu> lstm_middle_right_cell;
      lstm_middle_right_output.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      lstm_middle_right_input.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      lstm_middle_right_forget.initial(_rnnHiddenSize, _rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      lstm_middle_right_cell.initial(_rnnHiddenSize, _rnnHiddenSize, 2 * _rnnHiddenSize, true, 3, 0);
      _lstm_middle_right_output.push_back(lstm_middle_right_output);
      _lstm_middle_right_input.push_back(lstm_middle_right_input);
      _lstm_middle_right_forget.push_back(lstm_middle_right_forget);
      _lstm_middle_right_cell.push_back(lstm_middle_right_cell);
    }

    _tanh_project.initial(_hiddensize, 2 * _rnnHiddenSize, true, 3, 0);
    _olayer_linear.initial(_labelSize, _hiddensize, true, 4, 2);

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
    _tanh_project.release();

    _lstm_left_output.release();
    _lstm_left_input.release();
    _lstm_left_forget.release();
    _lstm_left_cell.release();

    _lstm_right_output.release();
    _lstm_right_input.release();
    _lstm_right_forget.release();
    _lstm_right_cell.release();

    for (int idx = 0; idx < _rnnMidLayers; idx++) {
      _lstm_middle_left_output[idx].release();
      _lstm_middle_left_input[idx].release();
      _lstm_middle_left_forget[idx].release();
      _lstm_middle_left_cell[idx].release();

      _lstm_middle_right_output[idx].release();
      _lstm_middle_right_input[idx].release();
      _lstm_middle_right_forget[idx].release();
      _lstm_middle_right_cell[idx].release();
    }

  }

  inline double process(const vector<Example>& examples, int iter) {
    _eval.reset();
    _indexers.clear();

    int example_num = examples.size();
    double cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      vector<vector<int> > linear_features(seq_size);

      Tensor<xpu, 2, double> input[seq_size], inputLoss[seq_size];

      Tensor<xpu, 2, double> lstm_hidden_left_final[seq_size][_rnnMidLayers + 1], lstm_hidden_left_finalLoss[seq_size][_rnnMidLayers + 1],
          lstm_hidden_left_finalFLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_left_cell[seq_size][_rnnMidLayers + 1], lstm_hidden_left_cellLoss[seq_size][_rnnMidLayers + 1],
          lstm_hidden_left_cellFLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_left_finalcache[seq_size][_rnnMidLayers + 1], lstm_hidden_left_finalcacheLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_left_cellcache[seq_size][_rnnMidLayers + 1], lstm_hidden_left_cellcacheLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_left_input[seq_size][_rnnMidLayers + 1], lstm_hidden_left_inputLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_left_output[seq_size][_rnnMidLayers + 1], lstm_hidden_left_outputLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_left_forget[seq_size][_rnnMidLayers + 1], lstm_hidden_left_forgetLoss[seq_size][_rnnMidLayers + 1];

      Tensor<xpu, 2, double> lstm_hidden_right_final[seq_size][_rnnMidLayers + 1], lstm_hidden_right_finalLoss[seq_size][_rnnMidLayers + 1],
          lstm_hidden_right_finalFLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_right_cell[seq_size][_rnnMidLayers + 1], lstm_hidden_right_cellLoss[seq_size][_rnnMidLayers + 1],
          lstm_hidden_right_cellFLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_right_finalcache[seq_size][_rnnMidLayers + 1], lstm_hidden_right_finalcacheLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_right_cellcache[seq_size][_rnnMidLayers + 1], lstm_hidden_right_cellcacheLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_right_input[seq_size][_rnnMidLayers + 1], lstm_hidden_right_inputLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_right_output[seq_size][_rnnMidLayers + 1], lstm_hidden_right_outputLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> lstm_hidden_right_forget[seq_size][_rnnMidLayers + 1], lstm_hidden_right_forgetLoss[seq_size][_rnnMidLayers + 1];

      Tensor<xpu, 2, double> lstm_hidden_merge[seq_size][_rnnMidLayers + 1], lstm_hidden_mergeLoss[seq_size][_rnnMidLayers + 1];
      Tensor<xpu, 2, double> inputLossTmp = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      Tensor<xpu, 2, double> lstm_hidden_mergeLossTmp = NewTensor<xpu>(Shape2(1, 2 * _rnnHiddenSize), 0.0);
      Tensor<xpu, 2, double> lstm_hidden_null1 = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
      Tensor<xpu, 2, double> lstm_hidden_null1Loss = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
      Tensor<xpu, 2, double> lstm_hidden_null2 = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
      Tensor<xpu, 2, double> lstm_hidden_null2Loss = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

      Tensor<xpu, 2, double> project[seq_size], projectLoss[seq_size];
      Tensor<xpu, 2, double> denseout[seq_size], denseoutLoss[seq_size];
      Tensor<xpu, 2, double> sparseout[seq_size], sparseoutLoss[seq_size];
      Tensor<xpu, 2, double> output[seq_size], outputLoss[seq_size];
      Tensor<xpu, 2, double> scores[seq_size];

      Tensor<xpu, 2, double> wordprime[seq_size], wordprimeLoss[seq_size], wordprimeMask[seq_size];
      Tensor<xpu, 2, double> wordrepresent[seq_size], wordrepresentLoss[seq_size];
      Tensor<xpu, 2, double> inputcontext[seq_size][_wordwindow];
      Tensor<xpu, 2, double> inputcontextLoss[seq_size][_wordwindow];

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        for (int idy = 0; idy < _wordwindow; idy++) {
          inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
          inputcontextLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        }
        wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        wordprimeMask[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 1.0);
        wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        wordrepresentLoss[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
        for (int idy = 0; idy <= _rnnMidLayers; idy++) {
          lstm_hidden_left_final[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_finalLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_finalFLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_cell[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_cellLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_cellFLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_finalcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_finalcacheLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_cellcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_cellcacheLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_output[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_outputLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_input[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_inputLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_forget[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_left_forgetLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

          lstm_hidden_right_final[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_finalLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_finalFLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_cell[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_cellLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_cellFLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_finalcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_finalcacheLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_cellcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_cellcacheLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_output[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_outputLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_input[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_inputLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_forget[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
          lstm_hidden_right_forgetLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

          lstm_hidden_merge[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _rnnHiddenSize), 0.0);
          lstm_hidden_mergeLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _rnnHiddenSize), 0.0);
        }

        input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        projectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
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
        offset = 0;
        for (int i = 0; i < _wordwindow; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            input[idx][0][offset] = inputcontext[idx][i][0][j];
            offset++;
          }
        }
      }

      // left rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _lstm_left_input.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_left_input[idx][0]);
          _lstm_left_cell.ComputeForwardScore(lstm_hidden_null1, input[idx], lstm_hidden_left_cellcache[idx][0]);
          lstm_hidden_left_cell[idx][0] = lstm_hidden_left_cellcache[idx][0] * lstm_hidden_left_input[idx][0];
          _lstm_left_output.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0]);
          lstm_hidden_left_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_left_cell[idx][0]);
          lstm_hidden_left_final[idx][0] = lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_output[idx][0];
        } else {
          _lstm_left_input.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx],
              lstm_hidden_left_input[idx][0]);
          _lstm_left_forget.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx],
              lstm_hidden_left_forget[idx][0]);
          _lstm_left_cell.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], input[idx], lstm_hidden_left_cellcache[idx][0]);
          lstm_hidden_left_cell[idx][0] = lstm_hidden_left_cellcache[idx][0] * lstm_hidden_left_input[idx][0]
              + lstm_hidden_left_cell[idx - 1][0] * lstm_hidden_left_forget[idx][0];
          _lstm_left_output.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0]);
          lstm_hidden_left_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_left_cell[idx][0]);
          lstm_hidden_left_final[idx][0] = lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_output[idx][0];
        }
      }

      // right rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _lstm_right_input.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_right_input[idx][0]);
          _lstm_right_cell.ComputeForwardScore(lstm_hidden_null1, input[idx], lstm_hidden_right_cellcache[idx][0]);
          lstm_hidden_right_cell[idx][0] = lstm_hidden_right_cellcache[idx][0] * lstm_hidden_right_input[idx][0];
          _lstm_right_output.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_right_cell[idx][0], input[idx], lstm_hidden_right_output[idx][0]);
          lstm_hidden_right_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_right_cell[idx][0]);
          lstm_hidden_right_final[idx][0] = lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_output[idx][0];
        } else {
          _lstm_right_input.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
              lstm_hidden_right_input[idx][0]);
          _lstm_right_forget.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
              lstm_hidden_right_forget[idx][0]);
          _lstm_right_cell.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], input[idx], lstm_hidden_right_cellcache[idx][0]);
          lstm_hidden_right_cell[idx][0] = lstm_hidden_right_cellcache[idx][0] * lstm_hidden_right_input[idx][0]
              + lstm_hidden_right_cell[idx + 1][0] * lstm_hidden_right_forget[idx][0];
          _lstm_right_output.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx][0], input[idx],
              lstm_hidden_right_output[idx][0]);
          lstm_hidden_right_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_right_cell[idx][0]);
          lstm_hidden_right_final[idx][0] = lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_output[idx][0];
        }
      }

      for (int idLayer = 0; idLayer < _rnnMidLayers; idLayer++) {
        for (int idx = 0; idx < seq_size; idx++) {
          concat(lstm_hidden_left_final[idx][idLayer], lstm_hidden_right_final[idx][idLayer], lstm_hidden_merge[idx][idLayer]);
        }
        // left rnn
        for (int idx = 0; idx < seq_size; idx++) {
          if (idx == 0) {
            _lstm_middle_left_input[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_input[idx][idLayer + 1]);
            _lstm_middle_left_cell[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_cellcache[idx][idLayer + 1]);
            lstm_hidden_left_cell[idx][idLayer + 1] = lstm_hidden_left_cellcache[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1];
            _lstm_middle_left_output[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_left_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_output[idx][idLayer + 1]);
            lstm_hidden_left_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_left_cell[idx][idLayer + 1]);
            lstm_hidden_left_final[idx][idLayer + 1] = lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
          } else {
            _lstm_middle_left_input[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_left_input[idx][idLayer + 1]);
            _lstm_middle_left_forget[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_left_forget[idx][idLayer + 1]);
            _lstm_middle_left_cell[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_cellcache[idx][idLayer + 1]);
            lstm_hidden_left_cell[idx][idLayer + 1] = lstm_hidden_left_cellcache[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1]
                + lstm_hidden_left_cell[idx - 1][idLayer + 1] * lstm_hidden_left_forget[idx][idLayer + 1];
            _lstm_middle_left_output[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_left_output[idx][idLayer + 1]);
            lstm_hidden_left_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_left_cell[idx][idLayer + 1]);
            lstm_hidden_left_final[idx][idLayer + 1] = lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
          }
        }

        // right rnn
        for (int idx = seq_size - 1; idx >= 0; idx--) {
          if (idx == seq_size - 1) {
            _lstm_middle_right_input[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_input[idx][idLayer + 1]);
            _lstm_middle_right_cell[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_cellcache[idx][idLayer + 1]);
            lstm_hidden_right_cell[idx][idLayer + 1] = lstm_hidden_right_cellcache[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1];
            _lstm_middle_right_output[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_right_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_output[idx][idLayer + 1]);
            lstm_hidden_right_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_right_cell[idx][idLayer + 1]);
            lstm_hidden_right_final[idx][idLayer + 1] = lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
          } else {
            _lstm_middle_right_input[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_right_input[idx][idLayer + 1]);
            _lstm_middle_right_forget[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_right_forget[idx][idLayer + 1]);
            _lstm_middle_right_cell[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_cellcache[idx][idLayer + 1]);
            lstm_hidden_right_cell[idx][idLayer + 1] = lstm_hidden_right_cellcache[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1]
                + lstm_hidden_right_cell[idx + 1][idLayer + 1] * lstm_hidden_right_forget[idx][idLayer + 1];
            _lstm_middle_right_output[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_right_output[idx][idLayer + 1]);
            lstm_hidden_right_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_right_cell[idx][idLayer + 1]);
            lstm_hidden_right_final[idx][idLayer + 1] = lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
          }
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {
        concat(lstm_hidden_left_final[idx][_rnnMidLayers], lstm_hidden_right_final[idx][_rnnMidLayers], lstm_hidden_merge[idx][_rnnMidLayers]);
        _tanh_project.ComputeForwardScore(lstm_hidden_merge[idx][_rnnMidLayers], project[idx]);
        _olayer_linear.ComputeForwardScore(project[idx], denseout[idx]);
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
        _olayer_linear.ComputeBackwardLoss(project[idx], denseout[idx], outputLoss[idx], projectLoss[idx]);
        _sparselayer_linear.ComputeBackwardLoss(linear_features[idx], sparseout[idx], outputLoss[idx]);
        _tanh_project.ComputeBackwardLoss(lstm_hidden_merge[idx][_rnnMidLayers], project[idx], projectLoss[idx], lstm_hidden_mergeLoss[idx][_rnnMidLayers]);
        unconcat(lstm_hidden_left_finalLoss[idx][_rnnMidLayers], lstm_hidden_right_finalLoss[idx][_rnnMidLayers], lstm_hidden_mergeLoss[idx][_rnnMidLayers]);
      }

      for (int idLayer = _rnnMidLayers - 1; idLayer >= 0; idLayer--) {
        //left rnn
        for (int idx = seq_size - 1; idx >= 0; idx--) {
          if (idx < seq_size - 1)
            lstm_hidden_left_finalLoss[idx][idLayer + 1] = lstm_hidden_left_finalLoss[idx][idLayer + 1] + lstm_hidden_left_finalFLoss[idx][idLayer + 1];

          lstm_hidden_left_finalcacheLoss[idx][idLayer + 1] = lstm_hidden_left_finalLoss[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
          lstm_hidden_left_outputLoss[idx][idLayer + 1] = lstm_hidden_left_finalLoss[idx][idLayer + 1] * lstm_hidden_left_finalcache[idx][idLayer + 1];
          if (idx < seq_size - 1) {
            lstm_hidden_left_cellLoss[idx][idLayer + 1] = lstm_hidden_left_finalcacheLoss[idx][idLayer + 1]
                * (1.0 - lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_finalcache[idx][idLayer + 1])
                + lstm_hidden_left_cellFLoss[idx][idLayer + 1];
          } else {
            lstm_hidden_left_cellLoss[idx][idLayer + 1] = lstm_hidden_left_finalcacheLoss[idx][idLayer + 1]
                * (1.0 - lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_finalcache[idx][idLayer + 1]);
          }

          if (idx == 0) {
            _lstm_middle_left_output[idLayer].ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_left_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_output[idx][idLayer + 1], lstm_hidden_left_outputLoss[idx][idLayer + 1], lstm_hidden_null1Loss, lstm_hidden_null2Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_left_cellLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] + lstm_hidden_null2Loss;

            lstm_hidden_left_cellcacheLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1];
            lstm_hidden_left_inputLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] * lstm_hidden_left_cellcache[idx][idLayer + 1];

            _lstm_middle_left_cell[idLayer].ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_cellcache[idx][idLayer + 1], lstm_hidden_left_cellcacheLoss[idx][idLayer + 1], lstm_hidden_null1Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;

            _lstm_middle_left_input[idLayer].ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_input[idx][idLayer + 1], lstm_hidden_left_inputLoss[idx][idLayer + 1], lstm_hidden_null1Loss, lstm_hidden_null2Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
          } else {
            _lstm_middle_left_output[idLayer].ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_left_output[idx][idLayer + 1], lstm_hidden_left_outputLoss[idx][idLayer + 1],
                lstm_hidden_null1Loss, lstm_hidden_null2Loss, lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_left_cellLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] + lstm_hidden_null2Loss;
            lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] + lstm_hidden_null1Loss;

            lstm_hidden_left_cellcacheLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1];
            lstm_hidden_left_inputLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] * lstm_hidden_left_cellcache[idx][idLayer + 1];
            lstm_hidden_left_cellFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] * lstm_hidden_left_forget[idx][idLayer + 1];
            lstm_hidden_left_forgetLoss[idx][idLayer + 1] = lstm_hidden_left_cellLoss[idx][idLayer + 1] * lstm_hidden_left_cell[idx - 1][idLayer + 1];

            _lstm_middle_left_cell[idLayer].ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_left_cellcache[idx][idLayer + 1], lstm_hidden_left_cellcacheLoss[idx][idLayer + 1], lstm_hidden_null1Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] + lstm_hidden_null1Loss;

            _lstm_middle_left_forget[idLayer].ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_left_forget[idx][idLayer + 1], lstm_hidden_left_forgetLoss[idx][idLayer + 1],
                lstm_hidden_null1Loss, lstm_hidden_null2Loss, lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_left_cellFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_cellFLoss[idx - 1][idLayer + 1] + lstm_hidden_null2Loss;
            lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] + lstm_hidden_null1Loss;

            _lstm_middle_left_input[idLayer].ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_left_input[idx][idLayer + 1], lstm_hidden_left_inputLoss[idx][idLayer + 1], lstm_hidden_null1Loss,
                lstm_hidden_null2Loss, lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_left_cellFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_cellFLoss[idx - 1][idLayer + 1] + lstm_hidden_null2Loss;
            lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] = lstm_hidden_left_finalFLoss[idx - 1][idLayer + 1] + lstm_hidden_null1Loss;
          }
        }

        // right rnn
        for (int idx = 0; idx < seq_size; idx++) {
          if (idx > 0)
            lstm_hidden_right_finalLoss[idx][idLayer + 1] = lstm_hidden_right_finalLoss[idx][idLayer + 1] + lstm_hidden_right_finalFLoss[idx][idLayer + 1];

          lstm_hidden_right_finalcacheLoss[idx][idLayer + 1] = lstm_hidden_right_finalLoss[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
          lstm_hidden_right_outputLoss[idx][idLayer + 1] = lstm_hidden_right_finalLoss[idx][idLayer + 1] * lstm_hidden_right_finalcache[idx][idLayer + 1];
          if (idx > 0) {
            lstm_hidden_right_cellLoss[idx][idLayer + 1] = lstm_hidden_right_finalcacheLoss[idx][idLayer + 1]
                * (1.0 - lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_finalcache[idx][idLayer + 1])
                + lstm_hidden_right_cellFLoss[idx][idLayer + 1];
          } else {
            lstm_hidden_right_cellLoss[idx][idLayer + 1] = lstm_hidden_right_finalcacheLoss[idx][idLayer + 1]
                * (1.0 - lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_finalcache[idx][idLayer + 1]);
          }

          if (idx == seq_size - 1) {
            _lstm_middle_right_output[idLayer].ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_right_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_output[idx][idLayer + 1], lstm_hidden_right_outputLoss[idx][idLayer + 1], lstm_hidden_null1Loss, lstm_hidden_null2Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_right_cellLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] + lstm_hidden_null2Loss;

            lstm_hidden_right_cellcacheLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1];
            lstm_hidden_right_inputLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] * lstm_hidden_right_cellcache[idx][idLayer + 1];

            _lstm_middle_right_cell[idLayer].ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_cellcache[idx][idLayer + 1], lstm_hidden_right_cellcacheLoss[idx][idLayer + 1], lstm_hidden_null1Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;

            _lstm_middle_right_input[idLayer].ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_input[idx][idLayer + 1], lstm_hidden_right_inputLoss[idx][idLayer + 1], lstm_hidden_null1Loss, lstm_hidden_null2Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
          } else {
            _lstm_middle_right_output[idLayer].ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_right_output[idx][idLayer + 1], lstm_hidden_right_outputLoss[idx][idLayer + 1],
                lstm_hidden_null1Loss, lstm_hidden_null2Loss, lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_right_cellLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] + lstm_hidden_null2Loss;
            lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] + lstm_hidden_null1Loss;

            lstm_hidden_right_cellcacheLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1];
            lstm_hidden_right_inputLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] * lstm_hidden_right_cellcache[idx][idLayer + 1];
            lstm_hidden_right_cellFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] * lstm_hidden_right_forget[idx][idLayer + 1];
            lstm_hidden_right_forgetLoss[idx][idLayer + 1] = lstm_hidden_right_cellLoss[idx][idLayer + 1] * lstm_hidden_right_cell[idx + 1][idLayer + 1];

            _lstm_middle_right_cell[idLayer].ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
                lstm_hidden_right_cellcache[idx][idLayer + 1], lstm_hidden_right_cellcacheLoss[idx][idLayer + 1], lstm_hidden_null1Loss,
                lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] + lstm_hidden_null1Loss;

            _lstm_middle_right_forget[idLayer].ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_right_forget[idx][idLayer + 1], lstm_hidden_right_forgetLoss[idx][idLayer + 1],
                lstm_hidden_null1Loss, lstm_hidden_null2Loss, lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_right_cellFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_cellFLoss[idx + 1][idLayer + 1] + lstm_hidden_null2Loss;
            lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] + lstm_hidden_null1Loss;

            _lstm_middle_right_input[idLayer].ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
                lstm_hidden_merge[idx][idLayer], lstm_hidden_right_input[idx][idLayer + 1], lstm_hidden_right_inputLoss[idx][idLayer + 1],
                lstm_hidden_null1Loss, lstm_hidden_null2Loss, lstm_hidden_mergeLossTmp);
            lstm_hidden_mergeLoss[idx][idLayer] = lstm_hidden_mergeLoss[idx][idLayer] + lstm_hidden_mergeLossTmp;
            lstm_hidden_right_cellFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_cellFLoss[idx + 1][idLayer + 1] + lstm_hidden_null2Loss;
            lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] = lstm_hidden_right_finalFLoss[idx + 1][idLayer + 1] + lstm_hidden_null1Loss;
          }
        }

        for (int idx = 0; idx < seq_size; idx++) {
          unconcat(lstm_hidden_left_finalLoss[idx][idLayer], lstm_hidden_right_finalLoss[idx][idLayer], lstm_hidden_mergeLoss[idx][idLayer]);
        }
      }

      //left rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          lstm_hidden_left_finalLoss[idx][0] = lstm_hidden_left_finalLoss[idx][0] + lstm_hidden_left_finalFLoss[idx][0];

        lstm_hidden_left_finalcacheLoss[idx][0] = lstm_hidden_left_finalLoss[idx][0] * lstm_hidden_left_output[idx][0];
        lstm_hidden_left_outputLoss[idx][0] = lstm_hidden_left_finalLoss[idx][0] * lstm_hidden_left_finalcache[idx][0];
        if (idx < seq_size - 1) {
          lstm_hidden_left_cellLoss[idx][0] = lstm_hidden_left_finalcacheLoss[idx][0]
              * (1.0 - lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_finalcache[idx][0]) + lstm_hidden_left_cellFLoss[idx][0];
        } else {
          lstm_hidden_left_cellLoss[idx][0] = lstm_hidden_left_finalcacheLoss[idx][0]
              * (1.0 - lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_finalcache[idx][0]);
        }

        if (idx == 0) {
          _lstm_left_output.ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0],
              lstm_hidden_left_outputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_left_cellLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] + lstm_hidden_null2Loss;

          lstm_hidden_left_cellcacheLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] * lstm_hidden_left_input[idx][0];
          lstm_hidden_left_inputLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] * lstm_hidden_left_cellcache[idx][0];

          _lstm_left_cell.ComputeBackwardLoss(lstm_hidden_null1, input[idx], lstm_hidden_left_cellcache[idx][0], lstm_hidden_left_cellcacheLoss[idx][0],
              lstm_hidden_null1Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;

          _lstm_left_input.ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_left_input[idx][0],
              lstm_hidden_left_inputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
        } else {

          _lstm_left_output.ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0],
              lstm_hidden_left_outputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_left_cellLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] + lstm_hidden_null2Loss;
          lstm_hidden_left_finalFLoss[idx - 1][0] = lstm_hidden_left_finalFLoss[idx - 1][0] + lstm_hidden_null1Loss;

          lstm_hidden_left_cellcacheLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] * lstm_hidden_left_input[idx][0];
          lstm_hidden_left_inputLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] * lstm_hidden_left_cellcache[idx][0];
          lstm_hidden_left_cellFLoss[idx - 1][0] = lstm_hidden_left_cellLoss[idx][0] * lstm_hidden_left_forget[idx][0];
          lstm_hidden_left_forgetLoss[idx][0] = lstm_hidden_left_cellLoss[idx][0] * lstm_hidden_left_cell[idx - 1][0];

          _lstm_left_cell.ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][0], input[idx], lstm_hidden_left_cellcache[idx][0],
              lstm_hidden_left_cellcacheLoss[idx][0], lstm_hidden_null1Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_left_finalFLoss[idx - 1][0] = lstm_hidden_left_finalFLoss[idx - 1][0] + lstm_hidden_null1Loss;

          _lstm_left_forget.ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx],
              lstm_hidden_left_forget[idx][0], lstm_hidden_left_forgetLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_left_cellFLoss[idx - 1][0] = lstm_hidden_left_cellFLoss[idx - 1][0] + lstm_hidden_null2Loss;
          lstm_hidden_left_finalFLoss[idx - 1][0] = lstm_hidden_left_finalFLoss[idx - 1][0] + lstm_hidden_null1Loss;

          _lstm_left_input.ComputeBackwardLoss(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx],
              lstm_hidden_left_input[idx][0], lstm_hidden_left_inputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_left_cellFLoss[idx - 1][0] = lstm_hidden_left_cellFLoss[idx - 1][0] + lstm_hidden_null2Loss;
          lstm_hidden_left_finalFLoss[idx - 1][0] = lstm_hidden_left_finalFLoss[idx - 1][0] + lstm_hidden_null1Loss;
        }
      }

      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          lstm_hidden_right_finalLoss[idx][0] = lstm_hidden_right_finalLoss[idx][0] + lstm_hidden_right_finalFLoss[idx][0];

        lstm_hidden_right_finalcacheLoss[idx][0] = lstm_hidden_right_finalLoss[idx][0] * lstm_hidden_right_output[idx][0];
        lstm_hidden_right_outputLoss[idx][0] = lstm_hidden_right_finalLoss[idx][0] * lstm_hidden_right_finalcache[idx][0];
        if (idx > 0) {
          lstm_hidden_right_cellLoss[idx][0] = lstm_hidden_right_finalcacheLoss[idx][0]
              * (1.0 - lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_finalcache[idx][0]) + lstm_hidden_right_cellFLoss[idx][0];
        } else {
          lstm_hidden_right_cellLoss[idx][0] = lstm_hidden_right_finalcacheLoss[idx][0]
              * (1.0 - lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_finalcache[idx][0]);
        }

        if (idx == seq_size - 1) {
          _lstm_right_output.ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_right_cell[idx][0], input[idx], lstm_hidden_right_output[idx][0],
              lstm_hidden_right_outputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_right_cellLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] + lstm_hidden_null2Loss;

          lstm_hidden_right_cellcacheLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] * lstm_hidden_right_input[idx][0];
          lstm_hidden_right_inputLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] * lstm_hidden_right_cellcache[idx][0];

          _lstm_right_cell.ComputeBackwardLoss(lstm_hidden_null1, input[idx], lstm_hidden_right_cellcache[idx][0], lstm_hidden_right_cellcacheLoss[idx][0],
              lstm_hidden_null1Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;

          _lstm_right_input.ComputeBackwardLoss(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_right_input[idx][0],
              lstm_hidden_right_inputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
        } else {
          _lstm_right_output.ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx][0], input[idx],
              lstm_hidden_right_output[idx][0], lstm_hidden_right_outputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_right_cellLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] + lstm_hidden_null2Loss;
          lstm_hidden_right_finalFLoss[idx + 1][0] = lstm_hidden_right_finalFLoss[idx + 1][0] + lstm_hidden_null1Loss;

          lstm_hidden_right_cellcacheLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] * lstm_hidden_right_input[idx][0];
          lstm_hidden_right_inputLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] * lstm_hidden_right_cellcache[idx][0];
          lstm_hidden_right_cellFLoss[idx + 1][0] = lstm_hidden_right_cellLoss[idx][0] * lstm_hidden_right_forget[idx][0];
          lstm_hidden_right_forgetLoss[idx][0] = lstm_hidden_right_cellLoss[idx][0] * lstm_hidden_right_cell[idx + 1][0];

          _lstm_right_cell.ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][0], input[idx], lstm_hidden_right_cellcache[idx][0],
              lstm_hidden_right_cellcacheLoss[idx][0], lstm_hidden_null1Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_right_finalFLoss[idx + 1][0] = lstm_hidden_right_finalFLoss[idx + 1][0] + lstm_hidden_null1Loss;

          _lstm_right_forget.ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
              lstm_hidden_right_forget[idx][0], lstm_hidden_right_forgetLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_right_cellFLoss[idx + 1][0] = lstm_hidden_right_cellFLoss[idx + 1][0] + lstm_hidden_null2Loss;
          lstm_hidden_right_finalFLoss[idx + 1][0] = lstm_hidden_right_finalFLoss[idx + 1][0] + lstm_hidden_null1Loss;

          _lstm_right_input.ComputeBackwardLoss(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
              lstm_hidden_right_input[idx][0], lstm_hidden_right_inputLoss[idx][0], lstm_hidden_null1Loss, lstm_hidden_null2Loss, inputLossTmp);
          inputLoss[idx] = inputLoss[idx] + inputLossTmp;
          lstm_hidden_right_cellFLoss[idx + 1][0] = lstm_hidden_right_cellFLoss[idx + 1][0] + lstm_hidden_null2Loss;
          lstm_hidden_right_finalFLoss[idx + 1][0] = lstm_hidden_right_finalFLoss[idx + 1][0] + lstm_hidden_null1Loss;
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {
        offset = 0;
        for (int i = 0; i < _wordwindow; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            inputcontextLoss[idx][i][0][j] = inputLoss[idx][0][offset];
            offset++;
          }
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
        for (int idy = 0; idy < _wordwindow; idy++) {
          FreeSpace(&(inputcontext[idx][idy]));
          FreeSpace(&(inputcontextLoss[idx][idy]));
        }
        FreeSpace(&(wordprime[idx]));
        FreeSpace(&(wordprimeLoss[idx]));
        FreeSpace(&(wordprimeMask[idx]));
        FreeSpace(&(wordrepresent[idx]));
        FreeSpace(&(wordrepresentLoss[idx]));

        for (int idy = 0; idy <= _rnnMidLayers; idy++) {
          FreeSpace(&(lstm_hidden_left_final[idx][idy]));
          FreeSpace(&(lstm_hidden_left_finalLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_finalFLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_cell[idx][idy]));
          FreeSpace(&(lstm_hidden_left_cellLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_cellFLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_finalcache[idx][idy]));
          FreeSpace(&(lstm_hidden_left_finalcacheLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_cellcache[idx][idy]));
          FreeSpace(&(lstm_hidden_left_cellcacheLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_output[idx][idy]));
          FreeSpace(&(lstm_hidden_left_outputLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_input[idx][idy]));
          FreeSpace(&(lstm_hidden_left_inputLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_left_forget[idx][idy]));
          FreeSpace(&(lstm_hidden_left_forgetLoss[idx][idy]));

          FreeSpace(&(lstm_hidden_right_final[idx][idy]));
          FreeSpace(&(lstm_hidden_right_finalLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_finalFLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_cell[idx][idy]));
          FreeSpace(&(lstm_hidden_right_cellLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_cellFLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_finalcache[idx][idy]));
          FreeSpace(&(lstm_hidden_right_finalcacheLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_cellcache[idx][idy]));
          FreeSpace(&(lstm_hidden_right_cellcacheLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_output[idx][idy]));
          FreeSpace(&(lstm_hidden_right_outputLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_input[idx][idy]));
          FreeSpace(&(lstm_hidden_right_inputLoss[idx][idy]));
          FreeSpace(&(lstm_hidden_right_forget[idx][idy]));
          FreeSpace(&(lstm_hidden_right_forgetLoss[idx][idy]));

          FreeSpace(&(lstm_hidden_merge[idx][idy]));
          FreeSpace(&(lstm_hidden_mergeLoss[idx][idy]));
        }

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(project[idx]));
        FreeSpace(&(projectLoss[idx]));
        FreeSpace(&(sparseout[idx]));
        FreeSpace(&(sparseoutLoss[idx]));
        FreeSpace(&(denseout[idx]));
        FreeSpace(&(denseoutLoss[idx]));
        FreeSpace(&(output[idx]));
        FreeSpace(&(outputLoss[idx]));
        FreeSpace(&(scores[idx]));
      }

      FreeSpace(&inputLossTmp);
      FreeSpace(&lstm_hidden_mergeLossTmp);
      FreeSpace(&lstm_hidden_null1);
      FreeSpace(&lstm_hidden_null1Loss);
      FreeSpace(&lstm_hidden_null2);
      FreeSpace(&lstm_hidden_null2Loss);
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, vector<int>& results) {
    int seq_size = features.size();
    int offset = 0;

    Tensor<xpu, 2, double> input[seq_size];

    Tensor<xpu, 2, double> lstm_hidden_left_final[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_cell[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_finalcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_cellcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_input[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_output[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_forget[seq_size][_rnnMidLayers + 1];

    Tensor<xpu, 2, double> lstm_hidden_right_final[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_cell[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_finalcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_cellcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_input[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_output[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_forget[seq_size][_rnnMidLayers + 1];

    Tensor<xpu, 2, double> lstm_hidden_merge[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_null1 = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
    Tensor<xpu, 2, double> lstm_hidden_null2 = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

    Tensor<xpu, 2, double> project[seq_size];
    Tensor<xpu, 2, double> sparseout[seq_size];
    Tensor<xpu, 2, double> denseout[seq_size];
    Tensor<xpu, 2, double> output[seq_size];

    Tensor<xpu, 2, double> wordprime[seq_size];
    Tensor<xpu, 2, double> wordrepresent[seq_size];
    Tensor<xpu, 2, double> inputcontext[seq_size][_wordwindow];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordwindow; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
      }
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

      for (int idy = 0; idy <= _rnnMidLayers; idy++) {
        lstm_hidden_left_final[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_cell[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_finalcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_cellcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_output[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_input[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_forget[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

        lstm_hidden_right_final[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_cell[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_finalcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_cellcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_output[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_input[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_forget[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

        lstm_hidden_merge[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _rnnHiddenSize), 0.0);
      }

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
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
        offset = 0;
        for (int i = 0; i < _wordwindow; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            input[idx][0][offset] = inputcontext[idx][i][0][j];
            offset++;
          }
        }
      }

    // left rnn
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0) {
        _lstm_left_input.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_left_input[idx][0]);
        _lstm_left_cell.ComputeForwardScore(lstm_hidden_null1, input[idx], lstm_hidden_left_cellcache[idx][0]);
        lstm_hidden_left_cell[idx][0] = lstm_hidden_left_cellcache[idx][0] * lstm_hidden_left_input[idx][0];
        _lstm_left_output.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0]);
        lstm_hidden_left_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_left_cell[idx][0]);
        lstm_hidden_left_final[idx][0] = lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_output[idx][0];
      } else {
        _lstm_left_input.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx], lstm_hidden_left_input[idx][0]);
        _lstm_left_forget.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx],
            lstm_hidden_left_forget[idx][0]);
        _lstm_left_cell.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], input[idx], lstm_hidden_left_cellcache[idx][0]);
        lstm_hidden_left_cell[idx][0] = lstm_hidden_left_cellcache[idx][0] * lstm_hidden_left_input[idx][0]
            + lstm_hidden_left_cell[idx - 1][0] * lstm_hidden_left_forget[idx][0];
        _lstm_left_output.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0]);
        lstm_hidden_left_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_left_cell[idx][0]);
        lstm_hidden_left_final[idx][0] = lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_output[idx][0];
      }
    }

    // right rnn
    for (int idx = seq_size - 1; idx >= 0; idx--) {
      if (idx == seq_size - 1) {
        _lstm_right_input.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_right_input[idx][0]);
        _lstm_right_cell.ComputeForwardScore(lstm_hidden_null1, input[idx], lstm_hidden_right_cellcache[idx][0]);
        lstm_hidden_right_cell[idx][0] = lstm_hidden_right_cellcache[idx][0] * lstm_hidden_right_input[idx][0];
        _lstm_right_output.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_right_cell[idx][0], input[idx], lstm_hidden_right_output[idx][0]);
        lstm_hidden_right_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_right_cell[idx][0]);
        lstm_hidden_right_final[idx][0] = lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_output[idx][0];
      } else {
        _lstm_right_input.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
            lstm_hidden_right_input[idx][0]);
        _lstm_right_forget.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
            lstm_hidden_right_forget[idx][0]);
        _lstm_right_cell.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], input[idx], lstm_hidden_right_cellcache[idx][0]);
        lstm_hidden_right_cell[idx][0] = lstm_hidden_right_cellcache[idx][0] * lstm_hidden_right_input[idx][0]
            + lstm_hidden_right_cell[idx + 1][0] * lstm_hidden_right_forget[idx][0];
        _lstm_right_output.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx][0], input[idx],
            lstm_hidden_right_output[idx][0]);
        lstm_hidden_right_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_right_cell[idx][0]);
        lstm_hidden_right_final[idx][0] = lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_output[idx][0];
      }
    }

    for (int idLayer = 0; idLayer < _rnnMidLayers; idLayer++) {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(lstm_hidden_left_final[idx][idLayer], lstm_hidden_right_final[idx][idLayer], lstm_hidden_merge[idx][idLayer]);
      }
      // left rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _lstm_middle_left_input[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
              lstm_hidden_left_input[idx][idLayer + 1]);
          _lstm_middle_left_cell[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer], lstm_hidden_left_cellcache[idx][idLayer + 1]);
          lstm_hidden_left_cell[idx][idLayer + 1] = lstm_hidden_left_cellcache[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1];
          _lstm_middle_left_output[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_left_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_left_output[idx][idLayer + 1]);
          lstm_hidden_left_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_left_cell[idx][idLayer + 1]);
          lstm_hidden_left_final[idx][idLayer + 1] = lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
        } else {
          _lstm_middle_left_input[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_left_input[idx][idLayer + 1]);
          _lstm_middle_left_forget[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_left_forget[idx][idLayer + 1]);
          _lstm_middle_left_cell[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_left_cellcache[idx][idLayer + 1]);
          lstm_hidden_left_cell[idx][idLayer + 1] = lstm_hidden_left_cellcache[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1]
              + lstm_hidden_left_cell[idx - 1][idLayer + 1] * lstm_hidden_left_forget[idx][idLayer + 1];
          _lstm_middle_left_output[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_left_output[idx][idLayer + 1]);
          lstm_hidden_left_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_left_cell[idx][idLayer + 1]);
          lstm_hidden_left_final[idx][idLayer + 1] = lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
        }
      }

      // right rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _lstm_middle_right_input[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_input[idx][idLayer + 1]);
          _lstm_middle_right_cell[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_cellcache[idx][idLayer + 1]);
          lstm_hidden_right_cell[idx][idLayer + 1] = lstm_hidden_right_cellcache[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1];
          _lstm_middle_right_output[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_right_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_output[idx][idLayer + 1]);
          lstm_hidden_right_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_right_cell[idx][idLayer + 1]);
          lstm_hidden_right_final[idx][idLayer + 1] = lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
        } else {
          _lstm_middle_right_input[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_right_input[idx][idLayer + 1]);
          _lstm_middle_right_forget[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_right_forget[idx][idLayer + 1]);
          _lstm_middle_right_cell[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_cellcache[idx][idLayer + 1]);
          lstm_hidden_right_cell[idx][idLayer + 1] = lstm_hidden_right_cellcache[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1]
              + lstm_hidden_right_cell[idx + 1][idLayer + 1] * lstm_hidden_right_forget[idx][idLayer + 1];
          _lstm_middle_right_output[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_right_output[idx][idLayer + 1]);
          lstm_hidden_right_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_right_cell[idx][idLayer + 1]);
          lstm_hidden_right_final[idx][idLayer + 1] = lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
        }
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      concat(lstm_hidden_left_final[idx][_rnnMidLayers], lstm_hidden_right_final[idx][_rnnMidLayers], lstm_hidden_merge[idx][_rnnMidLayers]);
      _tanh_project.ComputeForwardScore(lstm_hidden_merge[idx][_rnnMidLayers], project[idx]);
      _olayer_linear.ComputeForwardScore(project[idx], denseout[idx]);
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
      for (int idy = 0; idy < _wordwindow; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      for (int idy = 0; idy <= _rnnMidLayers; idy++) {
        FreeSpace(&(lstm_hidden_left_final[idx][idy]));
        FreeSpace(&(lstm_hidden_left_cell[idx][idy]));
        FreeSpace(&(lstm_hidden_left_finalcache[idx][idy]));
        FreeSpace(&(lstm_hidden_left_cellcache[idx][idy]));
        FreeSpace(&(lstm_hidden_left_output[idx][idy]));
        FreeSpace(&(lstm_hidden_left_input[idx][idy]));
        FreeSpace(&(lstm_hidden_left_forget[idx][idy]));

        FreeSpace(&(lstm_hidden_right_final[idx][idy]));
        FreeSpace(&(lstm_hidden_right_cell[idx][idy]));
        FreeSpace(&(lstm_hidden_right_finalcache[idx][idy]));
        FreeSpace(&(lstm_hidden_right_cellcache[idx][idy]));
        FreeSpace(&(lstm_hidden_right_output[idx][idy]));
        FreeSpace(&(lstm_hidden_right_input[idx][idy]));
        FreeSpace(&(lstm_hidden_right_forget[idx][idy]));

        FreeSpace(&(lstm_hidden_merge[idx][idy]));
      }
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
      FreeSpace(&(output[idx]));
    }
    FreeSpace(&lstm_hidden_null1);
    FreeSpace(&lstm_hidden_null2);
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    Tensor<xpu, 2, double> input[seq_size];

    Tensor<xpu, 2, double> lstm_hidden_left_final[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_cell[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_finalcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_cellcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_input[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_output[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_left_forget[seq_size][_rnnMidLayers + 1];

    Tensor<xpu, 2, double> lstm_hidden_right_final[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_cell[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_finalcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_cellcache[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_input[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_output[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_right_forget[seq_size][_rnnMidLayers + 1];

    Tensor<xpu, 2, double> lstm_hidden_merge[seq_size][_rnnMidLayers + 1];
    Tensor<xpu, 2, double> lstm_hidden_null1 = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
    Tensor<xpu, 2, double> lstm_hidden_null2 = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

    Tensor<xpu, 2, double> project[seq_size];
    Tensor<xpu, 2, double> sparseout[seq_size];
    Tensor<xpu, 2, double> denseout[seq_size];
    Tensor<xpu, 2, double> output[seq_size];
    Tensor<xpu, 2, double> scores[seq_size];

    Tensor<xpu, 2, double> wordprime[seq_size];
    Tensor<xpu, 2, double> wordrepresent[seq_size];
    Tensor<xpu, 2, double> inputcontext[seq_size][_wordwindow];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordwindow; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);
      }
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), 0.0);

      for (int idy = 0; idy <= _rnnMidLayers; idy++) {
        lstm_hidden_left_final[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_cell[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_finalcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_cellcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_output[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_input[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_left_forget[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

        lstm_hidden_right_final[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_cell[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_finalcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_cellcache[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_output[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_input[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);
        lstm_hidden_right_forget[idx][idy] = NewTensor<xpu>(Shape2(1, _rnnHiddenSize), 0.0);

        lstm_hidden_merge[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _rnnHiddenSize), 0.0);
      }

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
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
        offset = 0;
        for (int i = 0; i < _wordwindow; i++) {
          for (int j = 0; j < _token_representation_size; j++) {
            input[idx][0][offset] = inputcontext[idx][i][0][j];
            offset++;
          }
        }
      }

    // left rnn
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0) {
        _lstm_left_input.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_left_input[idx][0]);
        _lstm_left_cell.ComputeForwardScore(lstm_hidden_null1, input[idx], lstm_hidden_left_cellcache[idx][0]);
        lstm_hidden_left_cell[idx][0] = lstm_hidden_left_cellcache[idx][0] * lstm_hidden_left_input[idx][0];
        _lstm_left_output.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0]);
        lstm_hidden_left_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_left_cell[idx][0]);
        lstm_hidden_left_final[idx][0] = lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_output[idx][0];
      } else {
        _lstm_left_input.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx], lstm_hidden_left_input[idx][0]);
        _lstm_left_forget.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx - 1][0], input[idx],
            lstm_hidden_left_forget[idx][0]);
        _lstm_left_cell.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], input[idx], lstm_hidden_left_cellcache[idx][0]);
        lstm_hidden_left_cell[idx][0] = lstm_hidden_left_cellcache[idx][0] * lstm_hidden_left_input[idx][0]
            + lstm_hidden_left_cell[idx - 1][0] * lstm_hidden_left_forget[idx][0];
        _lstm_left_output.ComputeForwardScore(lstm_hidden_left_final[idx - 1][0], lstm_hidden_left_cell[idx][0], input[idx], lstm_hidden_left_output[idx][0]);
        lstm_hidden_left_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_left_cell[idx][0]);
        lstm_hidden_left_final[idx][0] = lstm_hidden_left_finalcache[idx][0] * lstm_hidden_left_output[idx][0];
      }
    }

    // right rnn
    for (int idx = seq_size - 1; idx >= 0; idx--) {
      if (idx == seq_size - 1) {
        _lstm_right_input.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, input[idx], lstm_hidden_right_input[idx][0]);
        _lstm_right_cell.ComputeForwardScore(lstm_hidden_null1, input[idx], lstm_hidden_right_cellcache[idx][0]);
        lstm_hidden_right_cell[idx][0] = lstm_hidden_right_cellcache[idx][0] * lstm_hidden_right_input[idx][0];
        _lstm_right_output.ComputeForwardScore(lstm_hidden_null1, lstm_hidden_right_cell[idx][0], input[idx], lstm_hidden_right_output[idx][0]);
        lstm_hidden_right_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_right_cell[idx][0]);
        lstm_hidden_right_final[idx][0] = lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_output[idx][0];
      } else {
        _lstm_right_input.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
            lstm_hidden_right_input[idx][0]);
        _lstm_right_forget.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx + 1][0], input[idx],
            lstm_hidden_right_forget[idx][0]);
        _lstm_right_cell.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], input[idx], lstm_hidden_right_cellcache[idx][0]);
        lstm_hidden_right_cell[idx][0] = lstm_hidden_right_cellcache[idx][0] * lstm_hidden_right_input[idx][0]
            + lstm_hidden_right_cell[idx + 1][0] * lstm_hidden_right_forget[idx][0];
        _lstm_right_output.ComputeForwardScore(lstm_hidden_right_final[idx + 1][0], lstm_hidden_right_cell[idx][0], input[idx],
            lstm_hidden_right_output[idx][0]);
        lstm_hidden_right_finalcache[idx][0] = F<nl_tanh>(lstm_hidden_right_cell[idx][0]);
        lstm_hidden_right_final[idx][0] = lstm_hidden_right_finalcache[idx][0] * lstm_hidden_right_output[idx][0];
      }
    }

    for (int idLayer = 0; idLayer < _rnnMidLayers; idLayer++) {
      for (int idx = 0; idx < seq_size; idx++) {
        concat(lstm_hidden_left_final[idx][idLayer], lstm_hidden_right_final[idx][idLayer], lstm_hidden_merge[idx][idLayer]);
      }
      // left rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _lstm_middle_left_input[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
              lstm_hidden_left_input[idx][idLayer + 1]);
          _lstm_middle_left_cell[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer], lstm_hidden_left_cellcache[idx][idLayer + 1]);
          lstm_hidden_left_cell[idx][idLayer + 1] = lstm_hidden_left_cellcache[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1];
          _lstm_middle_left_output[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_left_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_left_output[idx][idLayer + 1]);
          lstm_hidden_left_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_left_cell[idx][idLayer + 1]);
          lstm_hidden_left_final[idx][idLayer + 1] = lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
        } else {
          _lstm_middle_left_input[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_left_input[idx][idLayer + 1]);
          _lstm_middle_left_forget[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx - 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_left_forget[idx][idLayer + 1]);
          _lstm_middle_left_cell[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_left_cellcache[idx][idLayer + 1]);
          lstm_hidden_left_cell[idx][idLayer + 1] = lstm_hidden_left_cellcache[idx][idLayer + 1] * lstm_hidden_left_input[idx][idLayer + 1]
              + lstm_hidden_left_cell[idx - 1][idLayer + 1] * lstm_hidden_left_forget[idx][idLayer + 1];
          _lstm_middle_left_output[idLayer].ComputeForwardScore(lstm_hidden_left_final[idx - 1][idLayer + 1], lstm_hidden_left_cell[idx][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_left_output[idx][idLayer + 1]);
          lstm_hidden_left_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_left_cell[idx][idLayer + 1]);
          lstm_hidden_left_final[idx][idLayer + 1] = lstm_hidden_left_finalcache[idx][idLayer + 1] * lstm_hidden_left_output[idx][idLayer + 1];
        }
      }

      // right rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _lstm_middle_right_input[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_null2, lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_input[idx][idLayer + 1]);
          _lstm_middle_right_cell[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_cellcache[idx][idLayer + 1]);
          lstm_hidden_right_cell[idx][idLayer + 1] = lstm_hidden_right_cellcache[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1];
          _lstm_middle_right_output[idLayer].ComputeForwardScore(lstm_hidden_null1, lstm_hidden_right_cell[idx][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_output[idx][idLayer + 1]);
          lstm_hidden_right_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_right_cell[idx][idLayer + 1]);
          lstm_hidden_right_final[idx][idLayer + 1] = lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
        } else {
          _lstm_middle_right_input[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_right_input[idx][idLayer + 1]);
          _lstm_middle_right_forget[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx + 1][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_right_forget[idx][idLayer + 1]);
          _lstm_middle_right_cell[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_merge[idx][idLayer],
              lstm_hidden_right_cellcache[idx][idLayer + 1]);
          lstm_hidden_right_cell[idx][idLayer + 1] = lstm_hidden_right_cellcache[idx][idLayer + 1] * lstm_hidden_right_input[idx][idLayer + 1]
              + lstm_hidden_right_cell[idx + 1][idLayer + 1] * lstm_hidden_right_forget[idx][idLayer + 1];
          _lstm_middle_right_output[idLayer].ComputeForwardScore(lstm_hidden_right_final[idx + 1][idLayer + 1], lstm_hidden_right_cell[idx][idLayer + 1],
              lstm_hidden_merge[idx][idLayer], lstm_hidden_right_output[idx][idLayer + 1]);
          lstm_hidden_right_finalcache[idx][idLayer + 1] = F<nl_tanh>(lstm_hidden_right_cell[idx][idLayer + 1]);
          lstm_hidden_right_final[idx][idLayer + 1] = lstm_hidden_right_finalcache[idx][idLayer + 1] * lstm_hidden_right_output[idx][idLayer + 1];
        }
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      concat(lstm_hidden_left_final[idx][_rnnMidLayers], lstm_hidden_right_final[idx][_rnnMidLayers], lstm_hidden_merge[idx][_rnnMidLayers]);
      _tanh_project.ComputeForwardScore(lstm_hidden_merge[idx][_rnnMidLayers], project[idx]);
      _olayer_linear.ComputeForwardScore(project[idx], denseout[idx]);
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
      for (int idy = 0; idy < _wordwindow; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      for (int idy = 0; idy <= _rnnMidLayers; idy++) {
        FreeSpace(&(lstm_hidden_left_final[idx][idy]));
        FreeSpace(&(lstm_hidden_left_cell[idx][idy]));
        FreeSpace(&(lstm_hidden_left_finalcache[idx][idy]));
        FreeSpace(&(lstm_hidden_left_cellcache[idx][idy]));
        FreeSpace(&(lstm_hidden_left_output[idx][idy]));
        FreeSpace(&(lstm_hidden_left_input[idx][idy]));
        FreeSpace(&(lstm_hidden_left_forget[idx][idy]));

        FreeSpace(&(lstm_hidden_right_final[idx][idy]));
        FreeSpace(&(lstm_hidden_right_cell[idx][idy]));
        FreeSpace(&(lstm_hidden_right_finalcache[idx][idy]));
        FreeSpace(&(lstm_hidden_right_cellcache[idx][idy]));
        FreeSpace(&(lstm_hidden_right_output[idx][idy]));
        FreeSpace(&(lstm_hidden_right_input[idx][idy]));
        FreeSpace(&(lstm_hidden_right_forget[idx][idy]));

        FreeSpace(&(lstm_hidden_merge[idx][idy]));
      }
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
      FreeSpace(&(output[idx]));
      FreeSpace(&(scores[idx]));
    }
    FreeSpace(&lstm_hidden_null1);
    FreeSpace(&lstm_hidden_null2);

    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sparselayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _lstm_left_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _lstm_left_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _lstm_left_forget.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _lstm_left_cell.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _lstm_right_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _lstm_right_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _lstm_right_forget.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _lstm_right_cell.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    for (int idLayer = 0; idLayer < _rnnMidLayers; idLayer++) {
      _lstm_middle_left_output[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _lstm_middle_left_input[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _lstm_middle_left_forget[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _lstm_middle_left_cell[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);

      _lstm_middle_right_output[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _lstm_middle_right_input[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _lstm_middle_right_forget[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
      _lstm_middle_right_cell[idLayer].updateAdaGrad(nnRegular, adaAlpha, adaEps);
    }

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

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _lstm_left_input._W1, _lstm_left_input._gradW1, "_lstm_left_input._W1", iter);
    checkgrad(examples, _lstm_left_input._W2, _lstm_left_input._gradW2, "_lstm_left_input._W2", iter);
    checkgrad(examples, _lstm_left_input._W3, _lstm_left_input._gradW3, "_lstm_left_input._W3", iter);
    checkgrad(examples, _lstm_left_input._b, _lstm_left_input._gradb, "_lstm_left_input._b", iter);

    checkgrad(examples, _lstm_left_output._W1, _lstm_left_output._gradW1, "_lstm_left_output._W1", iter);
    checkgrad(examples, _lstm_left_output._W2, _lstm_left_output._gradW2, "_lstm_left_output._W2", iter);
    checkgrad(examples, _lstm_left_output._W3, _lstm_left_output._gradW3, "_lstm_left_output._W3", iter);
    checkgrad(examples, _lstm_left_output._b, _lstm_left_output._gradb, "_lstm_left_output._b", iter);

    checkgrad(examples, _lstm_left_forget._W1, _lstm_left_forget._gradW1, "_lstm_left_forget._W1", iter);
    checkgrad(examples, _lstm_left_forget._W2, _lstm_left_forget._gradW2, "_lstm_left_forget._W2", iter);
    checkgrad(examples, _lstm_left_forget._W3, _lstm_left_forget._gradW3, "_lstm_left_forget._W3", iter);
    checkgrad(examples, _lstm_left_forget._b, _lstm_left_forget._gradb, "_lstm_left_forget._b", iter);

    checkgrad(examples, _lstm_left_cell._WL, _lstm_left_cell._gradWL, "_lstm_left_cell._WL", iter);
    checkgrad(examples, _lstm_left_cell._WR, _lstm_left_cell._gradWR, "_lstm_left_cell._WR", iter);
    checkgrad(examples, _lstm_left_cell._b, _lstm_left_cell._gradb, "_lstm_left_cell._b", iter);

    checkgrad(examples, _lstm_right_input._W1, _lstm_right_input._gradW1, "_lstm_right_input._W1", iter);
    checkgrad(examples, _lstm_right_input._W2, _lstm_right_input._gradW2, "_lstm_right_input._W2", iter);
    checkgrad(examples, _lstm_right_input._W3, _lstm_right_input._gradW3, "_lstm_right_input._W3", iter);
    checkgrad(examples, _lstm_right_input._b, _lstm_right_input._gradb, "_lstm_right_input._b", iter);

    checkgrad(examples, _lstm_right_output._W1, _lstm_right_output._gradW1, "_lstm_right_output._W1", iter);
    checkgrad(examples, _lstm_right_output._W2, _lstm_right_output._gradW2, "_lstm_right_output._W2", iter);
    checkgrad(examples, _lstm_right_output._W3, _lstm_right_output._gradW3, "_lstm_right_output._W3", iter);
    checkgrad(examples, _lstm_right_output._b, _lstm_right_output._gradb, "_lstm_right_output._b", iter);

    checkgrad(examples, _lstm_right_forget._W1, _lstm_right_forget._gradW1, "_lstm_right_forget._W1", iter);
    checkgrad(examples, _lstm_right_forget._W2, _lstm_right_forget._gradW2, "_lstm_right_forget._W2", iter);
    checkgrad(examples, _lstm_right_forget._W3, _lstm_right_forget._gradW3, "_lstm_right_forget._W3", iter);
    checkgrad(examples, _lstm_right_forget._b, _lstm_right_forget._gradb, "_lstm_right_forget._b", iter);

    checkgrad(examples, _lstm_right_cell._WL, _lstm_right_cell._gradWL, "_lstm_right_cell._WL", iter);
    checkgrad(examples, _lstm_right_cell._WR, _lstm_right_cell._gradWR, "_lstm_right_cell._WR", iter);
    checkgrad(examples, _lstm_right_cell._b, _lstm_right_cell._gradb, "_lstm_right_cell._b", iter);

    for (int idLayer = 0; idLayer < _rnnMidLayers; idLayer++) {
      stringstream ssposition;
      ssposition << "[" << idLayer << "]";

      checkgrad(examples, _lstm_middle_left_input[idLayer]._W1, _lstm_middle_left_input[idLayer]._gradW1, "_lstm_middle_left_input" + ssposition.str() + "._W1",
          iter);
      checkgrad(examples, _lstm_middle_left_input[idLayer]._W2, _lstm_middle_left_input[idLayer]._gradW2, "_lstm_middle_left_input" + ssposition.str() + "._W2",
          iter);
      checkgrad(examples, _lstm_middle_left_input[idLayer]._W3, _lstm_middle_left_input[idLayer]._gradW3, "_lstm_middle_left_input" + ssposition.str() + "._W3",
          iter);
      checkgrad(examples, _lstm_middle_left_input[idLayer]._b, _lstm_middle_left_input[idLayer]._gradb, "_lstm_middle_left_input" + ssposition.str() + "._b",
          iter);

      checkgrad(examples, _lstm_middle_left_output[idLayer]._W1, _lstm_middle_left_output[idLayer]._gradW1,
          "_lstm_middle_left_output" + ssposition.str() + "._W1", iter);
      checkgrad(examples, _lstm_middle_left_output[idLayer]._W2, _lstm_middle_left_output[idLayer]._gradW2,
          "_lstm_middle_left_output" + ssposition.str() + "._W2", iter);
      checkgrad(examples, _lstm_middle_left_output[idLayer]._W3, _lstm_middle_left_output[idLayer]._gradW3,
          "_lstm_middle_left_output" + ssposition.str() + "._W3", iter);
      checkgrad(examples, _lstm_middle_left_output[idLayer]._b, _lstm_middle_left_output[idLayer]._gradb, "_lstm_middle_left_output" + ssposition.str() + "._b",
          iter);

      checkgrad(examples, _lstm_middle_left_forget[idLayer]._W1, _lstm_middle_left_forget[idLayer]._gradW1,
          "_lstm_middle_left_forget" + ssposition.str() + "._W1", iter);
      checkgrad(examples, _lstm_middle_left_forget[idLayer]._W2, _lstm_middle_left_forget[idLayer]._gradW2,
          "_lstm_middle_left_forget" + ssposition.str() + "._W2", iter);
      checkgrad(examples, _lstm_middle_left_forget[idLayer]._W3, _lstm_middle_left_forget[idLayer]._gradW3,
          "_lstm_middle_left_forget" + ssposition.str() + "._W3", iter);
      checkgrad(examples, _lstm_middle_left_forget[idLayer]._b, _lstm_middle_left_forget[idLayer]._gradb, "_lstm_middle_left_forget" + ssposition.str() + "._b",
          iter);

      checkgrad(examples, _lstm_middle_left_cell[idLayer]._WL, _lstm_middle_left_cell[idLayer]._gradWL, "_lstm_middle_left_cell" + ssposition.str() + "._WL",
          iter);
      checkgrad(examples, _lstm_middle_left_cell[idLayer]._WR, _lstm_middle_left_cell[idLayer]._gradWR, "_lstm_middle_left_cell" + ssposition.str() + "._WR",
          iter);
      checkgrad(examples, _lstm_middle_left_cell[idLayer]._b, _lstm_middle_left_cell[idLayer]._gradb, "_lstm_middle_left_cell" + ssposition.str() + "._b",
          iter);

      checkgrad(examples, _lstm_middle_right_input[idLayer]._W1, _lstm_middle_right_input[idLayer]._gradW1,
          "_lstm_middle_right_input" + ssposition.str() + "._W1", iter);
      checkgrad(examples, _lstm_middle_right_input[idLayer]._W2, _lstm_middle_right_input[idLayer]._gradW2,
          "_lstm_middle_right_input" + ssposition.str() + "._W2", iter);
      checkgrad(examples, _lstm_middle_right_input[idLayer]._W3, _lstm_middle_right_input[idLayer]._gradW3,
          "_lstm_middle_right_input" + ssposition.str() + "._W3", iter);
      checkgrad(examples, _lstm_middle_right_input[idLayer]._b, _lstm_middle_right_input[idLayer]._gradb, "_lstm_middle_right_input" + ssposition.str() + "._b",
          iter);

      checkgrad(examples, _lstm_middle_right_output[idLayer]._W1, _lstm_middle_right_output[idLayer]._gradW1,
          "_lstm_middle_right_output" + ssposition.str() + "._W1", iter);
      checkgrad(examples, _lstm_middle_right_output[idLayer]._W2, _lstm_middle_right_output[idLayer]._gradW2,
          "_lstm_middle_right_output" + ssposition.str() + "._W2", iter);
      checkgrad(examples, _lstm_middle_right_output[idLayer]._W3, _lstm_middle_right_output[idLayer]._gradW3,
          "_lstm_middle_right_output" + ssposition.str() + "._W3", iter);
      checkgrad(examples, _lstm_middle_right_output[idLayer]._b, _lstm_middle_right_output[idLayer]._gradb,
          "_lstm_middle_right_output" + ssposition.str() + "._b", iter);

      checkgrad(examples, _lstm_middle_right_forget[idLayer]._W1, _lstm_middle_right_forget[idLayer]._gradW1,
          "_lstm_middle_right_forget" + ssposition.str() + "._W1", iter);
      checkgrad(examples, _lstm_middle_right_forget[idLayer]._W2, _lstm_middle_right_forget[idLayer]._gradW2,
          "_lstm_middle_right_forget" + ssposition.str() + "._W2", iter);
      checkgrad(examples, _lstm_middle_right_forget[idLayer]._W3, _lstm_middle_right_forget[idLayer]._gradW3,
          "_lstm_middle_right_forget" + ssposition.str() + "._W3", iter);
      checkgrad(examples, _lstm_middle_right_forget[idLayer]._b, _lstm_middle_right_forget[idLayer]._gradb,
          "_lstm_middle_right_forget" + ssposition.str() + "._b", iter);

      checkgrad(examples, _lstm_middle_right_cell[idLayer]._WL, _lstm_middle_right_cell[idLayer]._gradWL, "_lstm_middle_right_cell" + ssposition.str() + "._WL",
          iter);
      checkgrad(examples, _lstm_middle_right_cell[idLayer]._WR, _lstm_middle_right_cell[idLayer]._gradWR, "_lstm_middle_right_cell" + ssposition.str() + "._WR",
          iter);
      checkgrad(examples, _lstm_middle_right_cell[idLayer]._b, _lstm_middle_right_cell[idLayer]._gradb, "_lstm_middle_right_cell" + ssposition.str() + "._b",
          iter);
    }

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

#endif /* SRC_SparseLSTMClassifier_H_ */
