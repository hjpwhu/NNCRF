/*
 * SparseClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseClassifier_H_
#define SRC_SparseClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "NRMat.h"
#include "MyLib.h"
#include "tensor.h"

#include "SparseUniHidderLayer.h"
#include "Utiltensor.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseClassifier {
public:
  SparseClassifier() {
    _dropOut = 0.5;
  }
  ~SparseClassifier() {

  }

public:
  int _labelSize;
  int _linearfeatSize;

  double _dropOut;
  Metric _eval;

  SparseUniHidderLayer<xpu> _layer_linear;

public:

  inline void init(int labelSize, int linearfeatSize) {
    _labelSize = labelSize;
    _linearfeatSize = linearfeatSize;

    _layer_linear.initial(_labelSize, _linearfeatSize, false, 4, 2);
    _eval.reset();

  }

  inline void release() {
    _layer_linear.release();
  }

  inline double process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    double cost = 0.0;
    int offset = 0;

    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      Tensor<xpu, 2, double> output[seq_size], outputLoss[seq_size];
      Tensor<xpu, 2, double> scores[seq_size];

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        scores[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      }

      //forward propagation
      vector<vector<int> > linear_features(seq_size);
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        srand(iter * example_num + count * seq_size + idx);
        linear_features[idx].clear();
        for (int idy = 0; idy < feature.linear_features.size(); idy++) {
          if (1.0 * rand() / RAND_MAX >= _dropOut) {
            linear_features[idx].push_back(feature.linear_features[idy]);
          }
        }
        _layer_linear.ComputeForwardScore(linear_features[idx], output[idx]);
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
        _layer_linear.ComputeBackwardLoss(linear_features[idx], output[idx], outputLoss[idx]);
      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
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
    Tensor<xpu, 2, double> output[seq_size];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      _layer_linear.ComputeForwardScore(feature.linear_features, output[idx]);
    }
    //end gru


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
      FreeSpace(&(output[idx]));
    }
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();

    Tensor<xpu, 2, double> output[seq_size];
    Tensor<xpu, 2, double> scores[seq_size];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      scores[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      _layer_linear.ComputeForwardScore(feature.linear_features, output[idx]);
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
      FreeSpace(&(output[idx]));
      FreeSpace(&(scores[idx]));
    }
    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _layer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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
    checkgrad(examples, _layer_linear._W, _layer_linear._gradW, "_layer_linear._W", iter, _layer_linear._indexers, false);
    checkgrad(examples, _layer_linear._b, _layer_linear._gradb, "_layer_linear._b", iter);
  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(double dropOut) {
    _dropOut = dropOut;
  }


};

#endif /* SRC_SparseClassifier_H_ */
