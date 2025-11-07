#include "utils.h"

GradientDescent::GradientDescent(double _learningRate) 
    : learningRate(_learningRate) {
}

std::vector<double> GradientDescent::step(const std::vector<double>& current_params, const std::vector<double>& gradient) const {
    
    if (current_params.size() != gradient.size()) {
        //
    }

    std::vector<double> new_params = current_params;

    for (size_t i = 0; i < new_params.size(); ++i) {
        new_params[i] -= learningRate * gradient[i];
    }

    return new_params;
}