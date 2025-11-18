//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//// g++ -std=c++17 -fopenmp -g -o dmc main.cpp dmc.cpp       ////
//// ./dmc [to run with parallelization]                      ////
//// OMP_NUM_THREADS=1 ./dmc [to run without parallelization] ////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

#include <vector>
#include <cmath>
#include <random>
#include <array>
#include <deque>
#include <omp.h>
#include <complex>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "utils.h"

namespace Constants {
    const int MAX_N_WALKERS = 100000;
    const int N_WALKERS_TARGET = 20000;
    // const int MAX_BRANCH_FACTOR = 4;
    const int DEFAULT_N_PARTICLE = 2;
    const int DEFAULT_N_DIM = 2;

    const double MIN_POPULATION_RATIO = 1e-4;
    const double MIN_DISTANCE = 1e-8;
    const double FINITE_DIFFERENCE_STEP = 1e-4;
    const double FINITE_DIFFERENCE_STEP_2 = FINITE_DIFFERENCE_STEP * FINITE_DIFFERENCE_STEP;
}

namespace Moire {
    inline const double RYDBERG_FOR_HARTREE = 2.0;
    inline const double RYDBERG = 13605.7; // em meV
    inline const double HARTREE = RYDBERG * RYDBERG_FOR_HARTREE; // em meV
    inline const double a0 = 0.5292;

    inline const double Vh1 = (-107.1 / 1.0) / HARTREE;
    inline const double Vh2 = (-(107.1 - 16.9) / 1.0) / HARTREE; 
    inline const double Ve1 = (-17.3 / 1.0) / HARTREE;
    inline const double Ve2 = (-(17.3 - 3.5) / 1.0) / HARTREE;

    inline const double E_field = -50.0  * a0 / HARTREE; 
    inline const double DIL_C0 = 6.387;
    inline const double DIL_C1 = 0.544;
    inline const double DIL_C2 = 0.042;

    inline double a10 = 3.282;
    inline double a20 = 3.160;

    inline double theta = 0.5 * PI / 180.0;
    inline double delta = std::abs(a10 - a20) / a10;

    inline const double MOIRE_LENGTH = a10 / std::sqrt(theta*theta + delta*delta) / a0;

    inline const double K_mag = (4.0 * PI) / (3.0 * MOIRE_LENGTH);

    inline double alpha = 1.5;
    inline double thickness = 6.15 / a0;
    inline double eps = 14.0;
    inline double eps1 = 4.5;
    inline double eps2 = 4.5;
    inline double r0 = alpha * thickness * eps / (eps1 + eps2);

    inline const double K1x = K_mag * 1.0;
    inline const double K1y = K_mag * 0.0;
    inline const double K2x = K_mag * (-0.5);
    inline const double K2y = K_mag * (std::sqrt(3.0) / 2.0);
    inline const double K3x = K_mag * (-0.5);
    inline const double K3y = K_mag * (-std::sqrt(3.0) / 2.0);

    inline const double theta_s_div_2 = 2.0 * PI / 3.0;
    inline const double theta_s = 4.0 * PI / 3.0;
};

struct BlockResult {
    double energy;
    double variance;
    double stdError;
};

class DMC {
    private:
        int nWalkers, nParticles, dim, stride;
        double deltaTau, referenceEnergy, instEnergy, meanEnergy;

        std::vector<std::mt19937> gens;
        // std::normal_distribution<double> dist;
        // std::uniform_real_distribution<double> uniform;

        std::vector<double> positions;
        std::vector<double> drifts;
        std::vector<double> localEnergy;

        void initializeWalkers();

        std::vector<double> getDrift(const double* position) const;

        double getLocalEnergy(const double* position) const;

        void updateReferenceEnergy(double blockEnergy, double blockTime);

        double potentialEnergy(const double* position) const;

        double driftGreenFunction(const double* newPosition, const double* oldPosition, const double* oldDrift) const;

        double branchGreenFunction(double newLocalEnergy, double oldLocalEnergy) const;
        
        double trialWaveFunction(const double* position) const;

        void timeStep();

        BlockResult blockStep(int nSteps);

    public: 
        DMC(double deltaTau, 
            int nWalkers = Constants::N_WALKERS_TARGET, 
            int nParticles = Constants::DEFAULT_N_PARTICLE, 
            int dim = Constants::DEFAULT_N_DIM);

        void run();
        
};