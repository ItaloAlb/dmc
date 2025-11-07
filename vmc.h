#include <vector>
#include <random>

class VMC {
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

        double getLocalEnergy(const double* position);

        void updateReferenceEnergy(double blockEnergy);

        double potentialEnergy(const double* position) const;

        double driftGreenFunction(const double* newPosition, const double* oldPosition, const double* oldDrift) const;

        double branchGreenFunction(double newLocalEnergy, double oldLocalEnergy) const;
        
        double trialWaveFunction(const double* position) const;

        void timeStep();

        void blockStep(int nSteps);

    public: 
        DMC(double deltaTau, 
            int nWalkers = Constants::N_WALKERS_TARGET, 
            int nParticles = Constants::DEFAULT_N_PARTICLE, 
            int dim = Constants::DEFAULT_N_DIM);

        void run();
        
};