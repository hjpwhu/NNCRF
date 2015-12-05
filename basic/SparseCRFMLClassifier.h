/*
 * SparseCRFMLClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseCRFMLClassifier_H_
#define SRC_SparseCRFMLClassifier_H_

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
class SparseCRFMLClassifier {
public:
  SparseCRFMLClassifier() {
    _dropOut = 0.5;
  }
  ~SparseCRFMLClassifier() {

  }

public:
  int _labelSize;
  int _linearfeatSize;

  double _dropOut;
  Metric _eval;

  Tensor<xpu, 2, double> _tagBigram;
  Tensor<xpu, 2, double> _grad_tagBigram;
  Tensor<xpu, 2, double> _eg2_tagBigram;

  SparseUniHidderLayer<xpu> _layer_linear;

public:

  inline void init(int labelSize, int linearfeatSize) {
    _labelSize = labelSize;
    _linearfeatSize = linearfeatSize;

    _tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);
    _grad_tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);

    random(_tagBigram, -0.1, 0.1, 100);

    _layer_linear.initial(_labelSize, _linearfeatSize, false, 4, 2);
    _eval.reset();

  }

  inline void release() {
    _layer_linear.release();

    FreeSpace(&_tagBigram);
    FreeSpace(&_grad_tagBigram);
    FreeSpace(&_eg2_tagBigram);
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

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
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
        _layer_linear.ComputeBackwardLoss(linear_features[idx], output[idx], outputLoss[idx]);
      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
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
      FreeSpace(&(output[idx]));
    }
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();

    Tensor<xpu, 2, double> output[seq_size];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      _layer_linear.ComputeForwardScore(feature.linear_features, output[idx]);
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
      FreeSpace(&(output[idx]));
    }
    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _layer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);


    _grad_tagBigram = _grad_tagBigram + _tagBigram * nnRegular;
    _eg2_tagBigram = _eg2_tagBigram + _grad_tagBigram * _grad_tagBigram;
    _tagBigram = _tagBigram - _grad_tagBigram * adaAlpha / F<nl_sqrt>(_eg2_tagBigram + adaEps);
    _grad_tagBigram = 0.0;

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

#endif /* SRC_SparseCRFMLClassifier_H_ */
