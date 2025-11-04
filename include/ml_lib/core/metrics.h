#pragma once
#include "../math/matrix.h"

class Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const = 0;
        virtual ~Metric() {}
};

class R2Metric : public Metric { 
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override;
};
class AdjustedR2Metric : public Metric {
    private:
        int k;
    public:
        AdjustedR2Metric(int predictors) : k(predictors) {}
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};
class MSEMetric : public Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};
class RMSEMetric : public Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};
class MAEMetric : public Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};