#include <vector>

class GradientDescent {
    private:
        double learningRate;
    public: 
        GradientDescent(double _learningRate);
        std::vector<double> step(const std::vector<double>& currentParams, const std::vector<double>& gradient) const;
};