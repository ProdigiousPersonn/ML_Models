#include "ml_lib/core/metrics.h"
#include <cmath>

double R2Metric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    const double rows = y_pred.rows();
    const double cols = y_pred.cols();

    double SSres = 0.0; // Residual sum of squares
    double SStot = 0.0; // Total sum of squares

    double mean = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mean += y_true(i, j);
        }
    }
    mean /= (rows * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff1 = y_pred(i, j) - y_true(i, j);
            double diff2 = y_true(i, j) - mean;
            SSres += diff1 * diff1;
            SStot += diff2 * diff2;
        }
    }

    if (SStot < 1e-9) { return 0; }

    return 1 - (SSres/SStot);
}

double AdjustedR2Metric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    const double rows = y_pred.rows();
    const double cols = y_pred.cols();

    R2Metric r2;
    double R2_val = r2.compute(y_true, y_pred);
    int n = y_true.rows();

    return 1 - (1 - R2_val) * ((n - 1)/(n - k - 1));
}

// MSE RMSE and MAE have same calculations as loss but duplicate code just to keep it seperate for OOP
double MSEMetric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = y_pred(i, j) - y_true(i, j);
            result += diff * diff;
        }
    }
    return result / n;
}

double RMSEMetric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = y_pred(i, j) - y_true(i, j);
            result += diff * diff;
        }
    }

    return sqrt(result / n);
}

double MAEMetric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            result += abs(y_pred(i, j) - y_true(i, j));
        }
    }
    return result / n;
}