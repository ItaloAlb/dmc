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