/*
 * TNNCRFMLClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TNNCRFMLClassifier_H_
#define SRC_TNNCRFMLClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "NRMat.h"
#include "MyLib.h"
#include "tensor.h"

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
class TNNCRFMLClassifier {
public:
  TNNCRFMLClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~TNNCRFMLClassifier() {

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

  Tensor<xpu, 2, double> _tagBigram;
  Tensor<xpu, 2, double> _grad_tagBigram;
  Tensor<xpu, 2, double> _eg2_tagBigram;


  int _hiddensize;
  int _inputsize, _token_representation_size;
  UniHidderLayer<xpu> _olayer_linear;
  UniHidderLayer<xpu> _tanh_project;


  int _labelSize;

  Metric _eval;

  double _dropOut;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, int labelSize, int hiddensize) {
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

    _tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);
    _grad_tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);

    random(_tagBigram, -0.1, 0.1, 100);

    _tanh_project.initial(_hiddensize, _inputsize, true, 3, 0);
    _olayer_linear.initial(_labelSize, _hiddensize, false, 4, 2);

  }

  inline void release() {
    FreeSpace(&_wordEmb);
    FreeSpace(&_grad_wordEmb);
    FreeSpace(&_eg2_wordEmb);
    FreeSpace(&_ft_wordEmb);
    _olayer_linear.release();
    _tanh_project.release();

    FreeSpace(&_tagBigram);
    FreeSpace(&_grad_tagBigram);
    FreeSpace(&_eg2_tagBigram);

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

      Tensor<xpu, 2, double> input[seq_size], inputLoss[seq_size];
      Tensor<xpu, 2, double> project[seq_size], projectLoss[seq_size];
      Tensor<xpu, 2, double> output[seq_size], outputLoss[seq_size];

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
        input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        projectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      }

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);
        
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

      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeForwardScore(input[idx], project[idx]);
        _olayer_linear.ComputeForwardScore(project[idx], output[idx]);
      }

      // crf, forward-backward algorithm
      // forward
      double tmp_value = 0.0;
      NRMat<double> alpha(seq_size, _labelSize);
      NRMat<double> alpha_annotated(seq_size, _labelSize);
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = 0; i < _labelSize; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            alpha[idx][i] = output[idx][0][i];
            if (example.m_labels[idx][i] == 0) {
              alpha_annotated[idx][i] = minlogvalue;
            } else if (example.m_labels[idx][i] == 1) {
              alpha_annotated[idx][i] = output[idx][0][i];
            } else {
              cout << "error label set" << std::endl;
            }
          } else {
            double tmp[_labelSize];
            for (int j = 0; j < _labelSize; ++j) {
              tmp[j] = _tagBigram[j][i] + output[idx][0][i] + alpha[idx - 1][j];
            }
            alpha[idx][i] = logsumexp(tmp, _labelSize);

            if (example.m_labels[idx][i] == 0) {
              alpha_annotated[idx][i] = minlogvalue;
            } else if (example.m_labels[idx][i] == 1) {
              double tmp_annoteted[_labelSize];
              for (int j = 0; j < _labelSize; ++j) {
                if (example.m_labels[idx - 1][j] == 1) {
                  tmp_annoteted[j] = _tagBigram[j][i] + output[idx][0][i] + alpha_annotated[idx - 1][j];
                } else {
                  tmp_annoteted[j] = minlogvalue;
                }
              }
              alpha_annotated[idx][i] = logsumexp(tmp_annoteted, _labelSize);
            } else {
              cout << "error label set" << std::endl;
            }
          }
        }
      }

      // backward
      NRMat<double> belta(seq_size, _labelSize);
      NRMat<double> belta_annotated(seq_size, _labelSize);

      for (int idx = seq_size - 1; idx >= 0; idx--) {
        for (int i = 0; i < _labelSize; ++i) {
          if (idx == seq_size - 1) {
            belta[idx][i] = 0.0;
            if (example.m_labels[idx][i] == 0) {
              belta_annotated[idx][i] = minlogvalue;
            } else if (example.m_labels[idx][i] == 1) {
              belta_annotated[idx][i] = 0.0;
            } else {
              cout << "error label set" << std::endl;
            }
          } else {
            double tmp[_labelSize];
            for (int j = 0; j < _labelSize; ++j) {
              tmp[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta[idx + 1][j];
            }
            belta[idx][i] = logsumexp(tmp, _labelSize);

            if (example.m_labels[idx][i] == 0) {
              belta_annotated[idx][i] = minlogvalue;
            } else if (example.m_labels[idx][i] == 1) {
              double tmp_annoteted[_labelSize];
              for (int j = 0; j < _labelSize; ++j) {
                if (example.m_labels[idx + 1][j] == 1) {
                  tmp_annoteted[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta_annotated[idx + 1][j];
                } else {
                  tmp_annoteted[j] = minlogvalue;
                }
              }
              belta_annotated[idx][i] = logsumexp(tmp_annoteted, _labelSize);
            } else {
              cout << "error label set" << std::endl;
            }
          }
        }
      }

      double logZ = logsumexp(alpha[seq_size - 1], _labelSize);

      double logZAnnotated = logsumexp(alpha_annotated[seq_size - 1], _labelSize);
      cost += (logZ - logZAnnotated) / example_num;

      // compute free expectation
      NRMat<double> marginalProbXL(seq_size, _labelSize);
      NRMat3d<double> marginalProbLL(seq_size, _labelSize, _labelSize);

      for (int idx = 0; idx < seq_size; idx++) {
        double sum = 0.0;
        for (int i = 0; i < _labelSize; ++i) {
          marginalProbXL[idx][i] = 0.0;
          if (idx == 0) {
            tmp_value = alpha[idx][i] + belta[idx][i] - logZ;
            marginalProbXL[idx][i] = exp(tmp_value);
          } else {
            for (int j = 0; j < _labelSize; ++j) {
              tmp_value = alpha[idx - 1][j] + output[idx][0][i] + _tagBigram[j][i] + belta[idx][i] - logZ;
              marginalProbLL[idx][j][i] = exp(tmp_value);
              marginalProbXL[idx][i] += marginalProbLL[idx][j][i];
            }
            tmp_value = alpha[idx][i] + belta[idx][i] - logZ;
            double tmpprob = exp(tmp_value);
            if (abs(marginalProbXL[idx][i] - tmpprob) > 1e-20) {
              // System.err.println(String.format("diff: %.18f\t%.18f",
              // marginalProbXL[idx][i], tmpprob));
            }
          }
          sum += marginalProbXL[idx][i];
        }
        if (abs(sum - 1) > 1e-6)
          std::cout << "prob unconstrained sum: " << sum << std::endl;
      }

      // compute constrained expectation
      NRMat<double> marginalAnnotatedProbXL(seq_size, _labelSize);
      NRMat3d<double> marginalAnnotatedProbLL(seq_size, _labelSize, _labelSize);
      for (int idx = 0; idx < seq_size; idx++) {
        double sum = 0;
        for (int i = 0; i < _labelSize; ++i) {
          marginalAnnotatedProbXL[idx][i] = 0.0;
          if (idx == 0) {
            if (example.m_labels[idx][i] == 1) {
              tmp_value = alpha_annotated[idx][i] + belta_annotated[idx][i] - logZAnnotated;
              marginalAnnotatedProbXL[idx][i] = exp(tmp_value);
            }
          } else {
            for (int j = 0; j < _labelSize; ++j) {
              marginalAnnotatedProbLL[idx][j][i] = 0.0;
              if (example.m_labels[idx - 1][j] == 1 && example.m_labels[idx][i] == 1) {
                tmp_value = alpha_annotated[idx - 1][j] + output[idx][0][i] + _tagBigram[j][i] + belta_annotated[idx][i] - logZAnnotated;
                marginalAnnotatedProbLL[idx][j][i] = exp(tmp_value);
              }
              marginalAnnotatedProbXL[idx][i] += marginalAnnotatedProbLL[idx][j][i];
            }
          }
          sum += marginalAnnotatedProbXL[idx][i];
        }
        if (abs(sum - 1) > 1e-6)
          std::cout << "prob constrained sum: " << sum << std::endl;
      }

      // compute _tagBigram grad
      for (int idx = 1; idx < seq_size; idx++) {
        for (int i = 0; i < _labelSize; ++i) {
          for (int j = 0; j < _labelSize; ++j) {
            _grad_tagBigram[i][j] += marginalProbLL[idx][i][j] - marginalAnnotatedProbLL[idx][i][j];
          }
        }
      }

      // get delta for each output
      _eval.overall_label_count += seq_size;
      for (int idx = 0; idx < seq_size; idx++) {
        double predict_best = -1.0;
        int predict_labelid = -1;
        double annotated_best = -1.0;
        int annotated_labelid = -1;
        for (int i = 0; i < _labelSize; ++i) {
          outputLoss[idx][0][i] = (marginalProbXL[idx][i] - marginalAnnotatedProbXL[idx][i]) / example_num;
          if (marginalProbXL[idx][i] > predict_best) {
            predict_best = marginalProbXL[idx][i];
            predict_labelid = i;
          }
          if (marginalAnnotatedProbXL[idx][i] > annotated_best) {
            annotated_best = marginalAnnotatedProbXL[idx][i];
            annotated_labelid = i;
          }
        }

        if (annotated_labelid != -1 && annotated_labelid == predict_labelid)
          _eval.correct_label_count++;
        if (annotated_labelid == -1)
          std::cout << "error, please debug" << std::endl;

      }

      // loss backward propagation
      for (int idx = 0; idx < seq_size; idx++) {
        _olayer_linear.ComputeBackwardLoss(project[idx], output[idx], outputLoss[idx], projectLoss[idx]);
        _tanh_project.ComputeBackwardLoss(input[idx], project[idx], projectLoss[idx], inputLoss[idx]);

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

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(project[idx]));
        FreeSpace(&(projectLoss[idx]));
        FreeSpace(&(output[idx]));
        FreeSpace(&(outputLoss[idx]));
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

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> project[seq_size];
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
      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
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

    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(input[idx], project[idx]);
      _olayer_linear.ComputeForwardScore(project[idx], output[idx]);
    }

    // decode algorithm
    // viterbi algorithm
    NRMat<double> maxscores(seq_size, _labelSize);
    NRMat<int> maxlastlabels(seq_size, _labelSize);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _labelSize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram[0][i] + output[idx][0][i]
              + maxscores[idx - 1][0];
          for (int j = 1; j < _labelSize; ++j) {
            double curscore = _tagBigram[j][i] + output[idx][0][i]
                + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          maxlastlabels[idx][i] = maxlastlabel;
        }
      }
    }

    results.resize(seq_size);
    double maxFinalScore = maxscores[seq_size - 1][0];
    results[seq_size - 1] = 0;
    for (int i = 1; i < _labelSize; ++i) {
      if (maxscores[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscores[seq_size - 1][i];
        results[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      results[idx] = maxlastlabels[idx + 1][results[idx + 1]];
    }

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordwindow; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output[idx]));
    }
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> project[seq_size];
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

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
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

    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(input[idx], project[idx]);
      _olayer_linear.ComputeForwardScore(project[idx], output[idx]);
    }

    // crf, forward-backward algorithm
    // forward
    double tmp_value = 0.0;
    NRMat<double> alpha(seq_size, _labelSize);
    NRMat<double> alpha_annotated(seq_size, _labelSize);
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _labelSize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          alpha[idx][i] = output[idx][0][i];
          if (example.m_labels[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (example.m_labels[idx][i] == 1) {
            alpha_annotated[idx][i] = output[idx][0][i];
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          double tmp[_labelSize];
          for (int j = 0; j < _labelSize; ++j) {
            tmp[j] = _tagBigram[j][i] + output[idx][0][i] + alpha[idx - 1][j];
          }
          alpha[idx][i] = logsumexp(tmp, _labelSize);

          if (example.m_labels[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (example.m_labels[idx][i] == 1) {
            double tmp_annoteted[_labelSize];
            for (int j = 0; j < _labelSize; ++j) {
              if (example.m_labels[idx - 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[j][i] + output[idx][0][i] + alpha_annotated[idx - 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            alpha_annotated[idx][i] = logsumexp(tmp_annoteted, _labelSize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    // backward
    NRMat<double> belta(seq_size, _labelSize);
    NRMat<double> belta_annotated(seq_size, _labelSize);

    for (int idx = seq_size - 1; idx >= 0; idx--) {
      for (int i = 0; i < _labelSize; ++i) {
        if (idx == seq_size - 1) {
          belta[idx][i] = 0.0;
          if (example.m_labels[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (example.m_labels[idx][i] == 1) {
            belta_annotated[idx][i] = 0.0;
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          double tmp[_labelSize];
          for (int j = 0; j < _labelSize; ++j) {
            tmp[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta[idx + 1][j];
          }
          belta[idx][i] = logsumexp(tmp, _labelSize);

          if (example.m_labels[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (example.m_labels[idx][i] == 1) {
            double tmp_annoteted[_labelSize];
            for (int j = 0; j < _labelSize; ++j) {
              if (example.m_labels[idx + 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta_annotated[idx + 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            belta_annotated[idx][i] = logsumexp(tmp_annoteted, _labelSize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    double logZ = logsumexp(alpha[seq_size - 1], _labelSize);

    double logZAnnotated = logsumexp(alpha_annotated[seq_size - 1], _labelSize);
    double cost = logZ - logZAnnotated;

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordwindow; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));
      }
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output[idx]));
    }
    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _grad_tagBigram = _grad_tagBigram + _tagBigram * nnRegular;
    _eg2_tagBigram = _eg2_tagBigram + _grad_tagBigram * _grad_tagBigram;
    _tagBigram = _tagBigram - _grad_tagBigram * adaAlpha / F<nl_sqrt>(_eg2_tagBigram + adaEps);
    _grad_tagBigram = 0.0;

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

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

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

#endif /* SRC_TNNCRFMLClassifier_H_ */
