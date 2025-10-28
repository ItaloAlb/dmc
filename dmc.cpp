#include "dmc.h"
using namespace Constants;

DMC::DMC(double deltaTau_, int nWalkers_, int nParticles_, int dim_)
    : deltaTau(deltaTau_),
      nWalkers(nWalkers_) ,
      nParticles(nParticles_),
      dim(dim_),
      referenceEnergy(REFERENCE_ENERGY),
      instEnergy(0.0),
      meanEnergy(0.0)
{

    positions.resize(nWalkers_ * nParticles_ * dim_);
    drifts.resize(nWalkers_ * nParticles_* dim_);
    localEnergy.resize(nWalkers_);

    stride = nParticles_* dim_;

    int nThreads = omp_get_max_threads();
    gens.resize(nThreads);

    std::random_device rd;
    for (int i = 0; i < nThreads; i++)
    {
        gens[i].seed(rd() + i);
    }
    
    initializeWalkers();
}

void DMC::timeStep(){
    // Create a temporary array to store the new generation of walkers
    std::vector<double> newPositions(MAX_N_WALKERS * stride);
    std::vector<double> newDrifts(MAX_N_WALKERS * stride);
    std::vector<double> newLocalEnergies(MAX_N_WALKERS);
    // Counter for the number of walkers in the new generation
    int newNWalkers = 0;
    // Accumulator for the total local energy of the new ensemble
    double ensembleEnergy = 0.0;
    #pragma omp parallel reduction(+:ensembleEnergy)
    {
        int threadId = omp_get_thread_num();
        auto& gen = gens[threadId];
        std::normal_distribution<double> dist(0.0, std::sqrt(deltaTau));
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Iterate over each walker in the current ensemble
        #pragma omp for
        for(int i = 0; i < nWalkers; i++) {
            std::vector<double> newPosition(stride);
            // Propose a new position for the walker using a random walk step and drift term
            for(int j = 0; j < stride; j++){
                double chi = dist(gen); // Random number from a normal distribution (diffusion term)
                // Update the position component: newPosition = oldPosition + diffusion_term + drift_term * time_step
                newPosition[j] = positions[i * stride + j] + chi + deltaTau * drifts[i * stride + j];
            }

            // Calculate the trial wave function at the old and new positions
            double oldPsi = trialWaveFunction(&positions[i * stride]);
            double newPsi = trialWaveFunction(&newPosition[0]);

            // Calculate the local energy at the old and new positions
            double oldLocalEnergy = getLocalEnergy(&positions[i * stride]); 
            double newLocalEnergy = getLocalEnergy(&newPosition[0]);
            
            // Check if the proposed move crosses a nodal surface (where Psi changes sign)
            // Moves that cross nodal surfaces are typically rejected in fixed-node DMC
            bool crossedNodalSurface = (oldPsi > 0 && newPsi < 0) || (oldPsi < 0 && newPsi > 0);

            // If the nodal surface is not crossed, proceed with the Metropolis-Hastings acceptance step
            std::vector<double> newDrift(stride); // Declare newDrift here
            // If the nodal surface is not crossed, proceed with the Metropolis-Hastings acceptance step
            if (!crossedNodalSurface) {
                // Calculate the drift at the new proposed position
                newDrift = getDrift(&newPosition[0]);
                // Calculate the forward Green's function for the drift term
                double forwardDriftGreenFunction = driftGreenFunction(&newPosition[0], &positions[i * stride], &drifts[i * stride]);
                // Calculate the backward Green's function for the drift term
                double backwardDriftGreenFunction = driftGreenFunction(&positions[i * stride], &newPosition[0], &newDrift[0]);

                // Calculate the acceptance probability for the Metropolis-Hastings step
                // This ensures that the walkers sample the distribution proportional to Psi^2
                double acceptanceProbability = 
                std::min(1.0, (backwardDriftGreenFunction * newPsi * newPsi) / (forwardDriftGreenFunction * oldPsi * oldPsi));
                
                // Accept or reject the proposed move based on the acceptance probability
                if (uniform(gen) < acceptanceProbability) {
                    // If accepted, update the walker's position, drift, and local energy
                    for (int j = 0; j < stride; j++) {
                        positions[i * stride + j] = newPosition[j];
                        drifts[i * stride + j] = newDrift[j];
                    }
                    localEnergy[i] = newLocalEnergy;
                }
                // If rejected, the walker remains at its old position with its old drift and local energy
            }
            
            // Determine the branching factor (number of copies of the walker)
            // This is based on the local energy and the reference energy (implicitly via branchGreenFunction)
            double eta = uniform(gen); // Random number for stochastic branching
            // The branch factor determines how many copies of the walker are made
            // It's typically an integer, calculated from the Green's function and a random number
            int branchFactor = static_cast<int>(eta + branchGreenFunction(newLocalEnergy, oldLocalEnergy));
            branchFactor = std::min(branchFactor, MAX_BRANCH_FACTOR);
            // If the branch factor is positive, create copies of the walker
            if (branchFactor > 0) {
                #pragma omp critical
                for (int n = 0; n < branchFactor; n++) {
                    if (newNWalkers >= MAX_N_WALKERS) break;

                    ensembleEnergy += localEnergy[i];

                    std::copy(newPosition.begin(), newPosition.end(), newPositions.begin() + newNWalkers * stride);
                    std::copy(newDrift.begin(), newDrift.end(), newDrifts.begin() + newNWalkers * stride);
                    newLocalEnergies[newNWalkers] = newLocalEnergy;

                    newNWalkers++;
                }
            }
        }
    }

    // Update the instantaneous energy of the ensemble
    instEnergy = newNWalkers > 0 ? ensembleEnergy / newNWalkers: 0.0;
    // Replace the old generation of walkers with the new generation
    positions.assign(newPositions.begin(), newPositions.begin() + newNWalkers * stride);
    drifts.assign(newDrifts.begin(), newDrifts.begin() + newNWalkers * stride);
    localEnergy.assign(newLocalEnergies.begin(), newLocalEnergies.begin() + newNWalkers);
    // Update the total number of walkers
    nWalkers = newNWalkers;
}

void DMC::blockStep(int nSteps) {
    
}

void DMC::updateReferenceEnergy(double blockEnergy) {
    double ratio = static_cast<double>(nWalkers) / static_cast<double>(N_WALKERS_TARGET);
    if (ratio < MIN_POPULATION_RATIO) ratio = MIN_POPULATION_RATIO;
    referenceEnergy = blockEnergy - ALPHA * std::log(ratio);
}

double DMC::driftGreenFunction(const double* newPosition, 
                               const double* oldPosition, 
                               const double* oldDrift) const {
    // Δ = (R - R' - τ v_D(R'))
    double norm2 = 0.0;
    for (int j = 0; j < stride; j++) {
        double diff = newPosition[j] - oldPosition[j] - deltaTau * oldDrift[j];
        norm2 += diff * diff;
    }
    // 1 / (2πτ)^(N/2)
    double factor = 1.0 / std::pow(2.0 * M_PI * deltaTau, 0.5 * stride);

    double exponent = - norm2 / (2.0 * deltaTau);
    // 1 / (2πτ)^(N/2) * exp(-Δ / (2 * τ))
    return factor * std::exp(exponent);
}

double DMC::branchGreenFunction(double newLocalEnergy,
                                double oldLocalEnergy) const {
    // exp(- τ/2 [E_L(R) + E_L(R') - 2E_T])
    return std::exp(- 0.5 * deltaTau * (newLocalEnergy + oldLocalEnergy - 2.0 * referenceEnergy));
}

std::vector<double> DMC::getDrift(const double* position) const {
    std::vector<double> drift(stride, 0.0);
    for (int i = 0; i < stride; i++) {
        std::vector<double> Rp(position, position + stride);
        std::vector<double> Rm(position, position + stride);
        Rp[i] += FINITE_DIFFERENCE_STEP;
        Rm[i] -= FINITE_DIFFERENCE_STEP;
        double forwardPsi = std::log(std::abs(trialWaveFunction(&Rp[0])));
        double backwardPsi = std::log(std::abs(trialWaveFunction(&Rm[0])));

        double lnDiff = forwardPsi - backwardPsi;
        drift[i] = lnDiff / (2.0 * FINITE_DIFFERENCE_STEP);
    }
    return drift;
}

double DMC::getLocalEnergy(const double* position) {
    double lap = - 2 * stride * std::log(std::abs(trialWaveFunction(position)));
    double grad = 0.0;
    std::vector<double> _position(position, position + stride);
    for (int i = 0; i < stride; i++) {
        _position[i] = position [i] + FINITE_DIFFERENCE_STEP;
        double forwardPsi = std::log(std::abs(trialWaveFunction(&_position[0])));

        _position[i] = position [i] - FINITE_DIFFERENCE_STEP;
        double backwardPsi = std::log(std::abs(trialWaveFunction(&_position[0])));

        _position[i] = position [i];

        double diff = std::abs((forwardPsi - backwardPsi) / (2.0 * FINITE_DIFFERENCE_STEP));
        grad += diff * diff;
        lap += forwardPsi + backwardPsi;
    }
    lap = lap / (FINITE_DIFFERENCE_STEP_2);
    return - 0.5 * (lap + grad) + potentialEnergy(&position[0]);
}

double DMC::potentialEnergy(const double* position) const {
    double dx = position[0] - position[2];
    double dy = position[1] - position[3];
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double r = std::sqrt(dx2 + dy2);
    if (r < MIN_DISTANCE) r = MIN_DISTANCE;
    return -1.0 / r;
}

double DMC::trialWaveFunction(const double* position) const {
    double dx = position[0] - position[2];
    double dy = position[1] - position[3];
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double r = std::sqrt(dx2 + dy2);
    if (r < MIN_DISTANCE) r = MIN_DISTANCE;
    double r2 = r * r;
    double c1 = -1.0;
    double c2 = 1.0;
    double c3 = 1.0;
    return std::exp((c1 * r + c2 * r2) / (1 + c3 * r));
    // return c1 * r2 * std::log(r) * std::exp(- c2 * r2) - c3 * r * (1 - std::exp(- c2 * r2));
}

void DMC::initializeWalkers() {
    // For simplicity and to directly address sampling from |Psi|^2,
    // we'll use a basic Metropolis-Hastings-like approach for initialization.

    std::mt19937 gen = gens[0];
    std::normal_distribution<double> dist_(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    const int nEquilibrationSteps = 100; // Number of Metropolis steps for equilibration
    const double stepSize = 0.5; // Step size for Metropolis proposal
    const double L = 10.0;

    
    for (int i = 0; i < nWalkers; ++i) {
        // Start with random positions for all walkers
        for (int j = 0; j < nParticles * dim; ++j) {
            positions[i * stride + j] = 2 * uniform(gen) * L - L;
        }

        std::vector<double> currentPosition(positions.begin() + i * stride, positions.begin() + (i + 1) * stride);
        double currentPsiSquared = trialWaveFunction(&currentPosition[0]);
        currentPsiSquared *= currentPsiSquared;

        // Now, equilibrate these walkers to sample from |Psi|^2 using Metropolis-Hastings
        for (int step = 0; step < nEquilibrationSteps; ++step) {
            // Propose a new position
            std::vector<double> proposedPosition(positions.begin() + i * stride, positions.begin() + (i + 1) * stride);
            for (int j = 0; j < nParticles * dim; ++j) {
                proposedPosition[j] += dist_(gen) * stepSize; // Random walk step
            }

            double proposedPsiSquared = trialWaveFunction(&proposedPosition[0]);
            proposedPsiSquared *= proposedPsiSquared;

            // Acceptance probability
            double acceptanceRatio = proposedPsiSquared / currentPsiSquared;

            if (uniform(gen) < std::min(1.0, acceptanceRatio)) {
                for (int j = 0; j < nParticles * dim; ++j) {
                    currentPosition[j] = proposedPosition[j];
                }
                // currentPosition = proposedPosition;
                currentPsiSquared = proposedPsiSquared;
            }
            // If rejected, walker stays at currentPosition
        }

        std::copy(currentPosition.begin(), currentPosition.end(), positions.begin() + i * stride);
        // After equilibration, initialize drift and localEnergy for each walker
        std::vector<double> drift = getDrift(&positions[i * stride]);
        std::copy(drift.begin(), drift.end(), drifts.begin() + i * stride);
        localEnergy[i] = getLocalEnergy(&positions[i * stride]);
        instEnergy += localEnergy[i];
    }
    instEnergy = instEnergy / nWalkers;
    std::cout << "inst Energy: " << instEnergy << std::endl;
}

void DMC::run() {
    int nBlockSteps = 15000;
    int nStepsPerBlock = 10;

    std::ofstream fout("dmc.dat");

    std::deque<double> energyQueue;

    for(int j = 0; j < nBlockSteps; j++) {
        double blockEnergy = 0.0;
        for(int i = 0; i < nStepsPerBlock; i++) {
            timeStep();
            blockEnergy += instEnergy;
        }
        blockEnergy = blockEnergy / nStepsPerBlock;

        energyQueue.push_back(blockEnergy);
        if (energyQueue.size() > 100) {
            energyQueue.pop_front();
        }

        meanEnergy = 0.0;
        for (double e : energyQueue) {
            meanEnergy += e;
        }
        meanEnergy /= energyQueue.size();

        double ratio = static_cast<double>(nWalkers) / static_cast<double>(N_WALKERS_TARGET);
        updateReferenceEnergy(blockEnergy);

        fout << j << " "
             << blockEnergy << " "
             << referenceEnergy << " "
             << meanEnergy << " "
             << nWalkers << "\n";
        
        
        std::cout << "Block " << j
                  << " | Energy = " << blockEnergy
                  << " | Ref Energy = " << referenceEnergy
                  << " | Mean Energy = " << meanEnergy
                  << " | Population = " << nWalkers
                  << std::endl;
    }

    fout.close();
}