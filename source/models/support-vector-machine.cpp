#include "ml_lib/models/support-vector-machine.h"
#include <cmath>
#include <algorithm>
#include <limits>

SupportVectorMachine::SupportVectorMachine(
    double C,
    double gamma,
    KERNEL kernel,
    int degree,
    double tolerance,
    int max_iterations,
    double coef0
)
    : C(C), gamma(gamma), kernel_type(kernel), degree(degree),
      tolerance(tolerance), max_iteration(max_iterations), coef0(coef0), bias(0.0)
{}

double SupportVectorMachine::kernel(const Matrix &X1, const Matrix &X2) {
    switch(this->kernel_type) {
        case KERNEL::LINEAR:
            return X1.dot(X2);
        case KERNEL::POLYNOMIAL:
            return pow(this->gamma * X1.dot(X2) + this->coef0, this->degree);
        case KERNEL::RBF: {
            Matrix diff = X1.sub(X2);
            return exp(-this->gamma * diff.dot(diff));
        }
        case KERNEL::SIGMOID:
            return tanh(this->gamma * X1.dot(X2) + this->coef0);
        default:
            return X1.dot(X2);
    }
}

double SupportVectorMachine::decision(const Matrix& X) {
    double result = 0.0;
    int m = this->X_train.rows();

    for (int i = 0; i < m; i++) {
        if (this->alphas[i] > 0) {
            result += this->alphas[i] * this->Y_train(i, 0) * kernel(this->X_train.row(i), X);
        }
    }
    return result + bias;
}

int SupportVectorMachine::takeStep(int I1, int I2) {
    if (I1 == I2) return 0;

    double alpha1 = alphas[I1];
    double alpha2 = alphas[I2];
    double Y1 = Y_train(I1, 0);
    double Y2 = Y_train(I2, 0);
    double E1 = errors[I1];
    double E2 = errors[I2];

    // Bounds
    double L, H;
    if (Y1 != Y2) {
        L = std::max(0.0, alpha2 - alpha1);
        H = std::min(C, C + alpha2 - alpha1);
    } else {
        L = std::max(0.0, alpha2 + alpha1 - C);
        H = std::min(C, alpha2 + alpha1);
    }
    if (L == H) return 0; // No room to move

    double k11 = kernel(X_train.row(I1), X_train.row(I1)); // similarity of 1 to itself
    double k12 = kernel(X_train.row(I1), X_train.row(I2)); // similarity of 1 to 2
    double k22 = kernel(X_train.row(I2), X_train.row(I2)); // similarity of 2 to itself
    double eta = 2 * k12 - k11 - k22;

    if (eta >= 0) return 0; // flat/convex, unable to find max

    double alpha2_new = alpha2 - Y2 * (E1 - E2) / eta;
    alpha2_new = std::clamp(alpha2_new, L, H);

    if (std::abs(alpha2_new - alpha2) < tolerance) return 0; // Delta too small

    double alpha1_new = alpha1 + Y1 * Y2 * (alpha2 - alpha2_new);

    double bias1 = bias - E1 - Y1 * (alpha1_new - alpha1) * k11 - Y2 * (alpha2_new - alpha2) * k12;
    double bias2 = bias - E2 - Y1 * (alpha1_new - alpha1) * k12 - Y2 * (alpha2_new - alpha2) * k22;

    if (0 < alpha1_new && alpha1_new < C)
        bias = bias1;
    else if (0 < alpha2_new && alpha2_new < C)
        bias = bias2;
    else
        bias = (bias1 + bias2) / 2;

    alphas[I1] = alpha1_new;
    alphas[I2] = alpha2_new;

    int m = X_train.rows();
    for (int i = 0; i < m; i++) {
        errors[i] = decision(X_train.row(i)) - Y_train(i, 0);
    }

    return 1;
}

int SupportVectorMachine::examineExample(int I2) {
    double alpha2 = alphas[I2];
    double r2 = errors[I2] * Y_train(I2, 0);
    double E2 = errors[I2];

    // KKT Violations
    if ((r2 < -tolerance && alpha2 < C) || (r2 > tolerance && alpha2 > 0)) {
        int I = -1;
        double max_diff = 0;

        int m = X_train.rows();

        for (int i = 0; i < m; i++) { // Argmax
            if (alphas[i] > 0 && alphas[i] < C) {
                double diff = abs(errors[i] - E2);
                if (diff > max_diff)
                    I = i;
            }
        }

        if (I >= 0 && takeStep(I, I2)) return 1;

        for (int i = 0; i < m; i++) {
            if (takeStep(i, I2)) return 1;
        }

    }

    return 0;
}

void SupportVectorMachine::fit(const Matrix &X, const Matrix &Y)
{
    X_train = X;
    Y_train = Y;
    int m = X.rows();
    int n = X.cols();

    alphas = std::vector<double>(m, 0.0);
    errors = std::vector<double>(m);
    bias = 0.0;

    for (int i = 0; i < m; i++) {
        errors[i] = -Y(i, 0);
    }
    
    double error_sum = 0;
    for (int i = 0; i < errors.size(); i++) {
        error_sum += errors[i];
    }

    int iterations = 0;
    int num_changed = 0;
    bool examine_all = true; // examine all on first iteration

    while ((num_changed > 0 || examineExample) && iterations < max_iteration) {
        num_changed = 0;

        for (int i = 0; i < m; i++) {
            if (examine_all || (alphas[i] > 0 && alphas[i] < C))
            num_changed += examineExample(i);
        }

        if (examine_all)
            examine_all = false;
        else if (num_changed == 0) 
            examine_all = true;

        iterations++;
    }
    
}

Matrix SupportVectorMachine::predict(const Matrix &X)
{
    int m = X.rows();
    Matrix predictions(m, 1);

    for (int i = 0; i < m; i++) {
        double val = decision(X.row(i));
        predictions(i, 0) = (val >= 0) ? 1.0 : -1.0;
    }

    return predictions;
}
 