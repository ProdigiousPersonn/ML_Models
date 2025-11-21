#pragma once
#include "../math/matrix.h"

class Model {
    public:
        virtual ~Model() {}
};

class GradientModelInterface : public Model {
    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual void backward(const Matrix& y_true) = 0;
        virtual void update() = 0;

        virtual ~GradientModelInterface() {}
};

class FitPredictModel : public Model {
    public:
        virtual void fit(const Matrix& X, const Matrix& y) = 0;
        virtual Matrix predict(const Matrix& X) = 0;

        virtual ~FitPredictModel() {}
};
