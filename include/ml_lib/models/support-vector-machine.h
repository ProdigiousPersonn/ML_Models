#pragma once
#include "../math/matrix.h"
#include <vector>

enum KERNEL {
    LINEAR,
    POLYNOMIAL,
    RBF,
    SIGMOID,
};

class SupportVectorMachine {
    private:
        double C;
        double gamma;
        KERNEL kernel_type;

        int degree;
        double tolerance;
        int max_iteration;
        double coef0;
        double bias;

        Matrix X_train;
        Matrix Y_train;
        std::vector<double> alphas;
        std::vector<double> errors;

        double kernel(const Matrix& X1, const Matrix& X2);
        double decision(const Matrix& X);
        int examineExample(int I2);
        int takeStep(int I1, int I2);

    public:
        SupportVectorMachine(
            double C = 1.0,
            double gamma = 0.1,
            KERNEL kernel = KERNEL::LINEAR,
            int degree = 3,
            double tolerance = 1e-3,
            int max_iterations = 1000,
            double coef0 = 0.0
        );

        void fit(const Matrix& X, const Matrix& y);
        Matrix predict(const Matrix& X);
};
