#include "ml_lib/models/logistic-regression.h"
#include <cmath>

const double e = std::exp(1.0);

LogisticRegression::LogisticRegression(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg)
    : LinearRegression(input_dim, loss, opt, reg), weights(input_dim, 1, 0.01), bias(1, 1, 0.0),
      grad_w(input_dim, 1, 0.0), grad_b(1, 1, 0.0) {}

// y^​=XW+b
Matrix LogisticRegression::forward(const Matrix &X)
{
    last_input = X;
    Matrix result = X.multiply(weights);

    for (int i = 0; i < result.rows(); i++) {
        result(i, 0) = 1 / (1 + pow(e, -(result(i, 0) + bias(0, 0))) );
    }

    last_output = result;
    return result;
}

void LogisticRegression::backward(const Matrix& y_true) {
    int m = last_input.rows();
    if (m == 0) return;

    // Compute gradient from loss function
    Matrix predictions = last_output;
    Matrix error = loss_func->gradient(predictions, y_true);
    Matrix reg_vals = regularizer->gradient(weights);

    // Compute gradients for weights and bias
    // grad_w = X^T * error + regularization
    grad_w = last_input.transpose().multiply(error).add(reg_vals);

    // grad_b = sum of error
    double grad_b_sum = 0.0;
    for (int j = 0; j < error.rows(); j++) {
        grad_b_sum += error(j, 0);
    }
    grad_b = Matrix(bias.rows(), bias.cols(), grad_b_sum);
}

void LogisticRegression::update() {
    // Update weights and bias using optimizer
    optimizer->step(weights, grad_w);
    optimizer->step(bias, grad_b);
}
