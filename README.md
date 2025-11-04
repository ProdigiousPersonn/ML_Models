# ML Models

A C++ machine learning library built from the ground up. Implementing various ML algorithms and models using fundamental linear algebra and optimization techniques.

## Features

- Linear regression with gradient descent optimization
- Matrix operations library (addition, multiplication, transpose, inverse)
- L1, L2, MSE, MAE, RMSE cost functions

## Prerequisites

- C++14 or higher
- CMake 3.16+
- A C++ compiler (GCC, Clang, or MSVC)

## Building

### Linux/macOS

```bash
# Clone the repository
git clone <your-repo-url>
cd LinearModel

# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Run the executable
./Build
```

### Windows

```bash
# Clone the repository
git clone <your-repo-url>
cd LinearModel

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build . --config Release

# Run the executable
.\Release\Build.exe
```

## Project Structure

```
LinearModel/
├── source/
│   ├── main.cpp           # Entry point
│   └── Helpers/           # Core implementations
├── external/              # Dependencies (fmt, spdlog)
└── CMakeLists.txt        # Build configuration
```

## Roadmap

### Regression
- [x] **Linear Regression**
- [x] **Evaluation Metrics (Regression):** MSE, MAE, RMSE, R-squared
- [ ] **Regularization:** L1 (Lasso) & L2 (Ridge)

### Classification
- [ ] **Logistic Regression**
- [ ] **Evaluation Metrics (Classification):**
    - [ ] Confusion Matri
    - [ ] Accuracy, Precision, Recall, F1-Score
    - [ ] ROC Curve and AUC
- [ ] **K-Nearest Neighbors (KNN)**
- [ ] **Support Vector Machines (SVMs)**

### Tree-Based Models
- [ ] **Decision Trees**
- [ ] **Random Forests**

### Unsupervised Learning
- [ ] **K-Means Clustering**

### Deep Learning Foundations
- [ ] **Neural Networks (Feedforward)**
- [ ] **Backpropagation**
- [ ] **Optimizers:**
    - [ ] **Mini-Batch Gradient Descent**
    - [ ] **Adam Optimizer**
- [ ] **Model Serialization**

### DL Architectures
- [ ] **Convolutional Neural Networks (CNNs)** (For images)
- [ ] **Recurrent Neural Networks (RNNs)** (For sequences)

### Modern NLP (Language)
- [ ] **Embeddings (Word2Vec, GloVe)**
- [ ] **Attention Mechanisms**
- [ ] **Transformers**
- [ ] **Language Models (Basic LLM architecture)**

## License

This project is available for educational purposes.
