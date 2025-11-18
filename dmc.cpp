#include "dmc.h"

using namespace Constants;
using namespace std::complex_literals;

DMC::DMC(double deltaTau_, int nWalkers_, int nParticles_, int dim_)
    : deltaTau(deltaTau_),
      nWalkers(nWalkers_) ,
      nParticles(nParticles_),
      dim(dim_),
      referenceEnergy(0.0),
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
            // bool crossedNodalSurface = (oldPsi > 0 && newPsi < 0) || (oldPsi < 0 && newPsi > 0);
            bool crossedNodalSurface = false;

            // If the nodal surface is not crossed, proceed with the Metropolis-Hastings acceptance step
            std::vector<double> newDrift(stride); // Declare newDrift here
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
            // branchFactor = std::min(branchFactor, MAX_BRANCH_FACTOR);
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

BlockResult DMC::blockStep(int nSteps) {
    double mean = 0.0;
    double mean2 = 0.0;

    // Welford Algorithm
    double delta, delta2;
    for (int n = 1; n < nSteps + 1; ++n) {
        timeStep();
        
        delta = instEnergy - mean;
        mean += delta / n;
        delta2 = instEnergy - mean;
        mean2 += delta * delta2;
    }

    BlockResult result;
    result.energy = mean;

    if (nSteps > 1) {
        result.variance = mean2 / (nSteps - 1); 
        result.stdError = std::sqrt(result.variance / nSteps);
    } else {
        result.variance = 0.0;
        result.stdError = 0.0;
    }
    return result;
}

void DMC::updateReferenceEnergy(double blockEnergy, double blockTime) {
    double ratio = static_cast<double>(nWalkers) / static_cast<double>(N_WALKERS_TARGET);
    if (ratio < MIN_POPULATION_RATIO) ratio = MIN_POPULATION_RATIO;
    referenceEnergy = blockEnergy - 1 / blockTime * std::log(ratio);
}

double DMC::driftGreenFunction(const double* newPosition, 
                               const double* oldPosition, 
                               const double* oldDrift) const {
    double norm2 = 0.0;
    for (int j = 0; j < stride; j++) {
        double diff = newPosition[j] - oldPosition[j] - deltaTau * oldDrift[j];
        norm2 += diff * diff;
    }
    double factor = 1.0 / std::pow(2.0 * PI * deltaTau, 0.5 * stride);
    double argexp = - norm2 / (2.0 * deltaTau);
    return factor * std::exp(argexp);
}

double DMC::branchGreenFunction(double newLocalEnergy,
                                double oldLocalEnergy) const {
    // exp(- τ/2 [E_L(R) + E_L(R') - 2E_T])
    return std::exp(- 0.5 * deltaTau * (newLocalEnergy + oldLocalEnergy - 2.0 * referenceEnergy));
}

std::vector<double> DMC::getDrift(const double* position) const {
    std::vector<double> drift(stride, 0.0);

    std::vector<double> Rp(position, position + stride);
    std::vector<double> Rm(position, position + stride);
    for (int i = 0; i < stride; i++) {
        Rp[i] += FINITE_DIFFERENCE_STEP;
        Rm[i] -= FINITE_DIFFERENCE_STEP;

        double forwardPsi = std::log(std::abs(trialWaveFunction(&Rp[0])));
        double backwardPsi = std::log(std::abs(trialWaveFunction(&Rm[0])));

        Rp[i] -= FINITE_DIFFERENCE_STEP;
        Rm[i] += FINITE_DIFFERENCE_STEP;

        double lnDiff = forwardPsi - backwardPsi;
        drift[i] = lnDiff / (2.0 * FINITE_DIFFERENCE_STEP);
    }
    return drift;
}

// double DMC::getLocalEnergy(const double* position) {
//     double lap = - 2 * stride * std::log(std::abs(trialWaveFunction(position)));
//     double grad = 0.0;
//     for (int i = 0; i < stride; i++) {
//         std::vector<double> Rp(position, position + stride);
//         std::vector<double> Rm(position, position + stride);
//         Rp[i] += FINITE_DIFFERENCE_STEP;
//         Rm[i] -= FINITE_DIFFERENCE_STEP;
//         double forwardPsi = std::log(std::abs(trialWaveFunction(&Rp[0])));
//         double backwardPsi = std::log(std::abs(trialWaveFunction(&Rm[0])));
//         double diff = (forwardPsi - backwardPsi) / (2.0 * FINITE_DIFFERENCE_STEP);
//         grad += diff * diff;
//         lap += forwardPsi + backwardPsi;
//     }
//     lap = lap / (FINITE_DIFFERENCE_STEP_2);
//     return - 0.5 * (lap + grad) + potentialEnergy(&position[0]);
// }

double DMC::getLocalEnergy(const double* position) const {
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

// Hydrogen atom 2D
// double DMC::potentialEnergy(const double* position) const {
//     double dx = position[0] - position[2];
//     double dy = position[1] - position[3];
//     double dx2 = dx * dx;
//     double dy2 = dy * dy;
//     double r = std::sqrt(dx2 + dy2);
//     if (r < MIN_DISTANCE) r = MIN_DISTANCE;
//     return -1.0 / r;
// }



//Helium atom 3D
// double DMC::potentialEnergy(const double* position) const {
//     double x1 = position[0];
//     double y1 = position[1];
//     double z1 = position[2];
    
//     double x2 = position[3];
//     double y2 = position[4];
//     double z2 = position[5];

//     double r1 = std::sqrt(x1*x1 + y1*y1 + z1*z1);
    
//     double r2 = std::sqrt(x2*x2 + y2*y2 + z2*z2);

//     double dx12 = x1 - x2;
//     double dy12 = y1 - y2;
//     double dz12 = z1 - z2;
//     double r12 = std::sqrt(dx12*dx12 + dy12*dy12 + dz12*dz12);

//     const double Z = 2.0;
//     return -Z / r1 - Z / r2 + 1.0 / r12;
// }

// rytova keldysh
// double DMC::potentialEnergy(const double* position) const {
//     double dx = position[0] - position[2];
//     double dy = position[1] - position[3];
//     double dx2 = dx * dx;
//     double dy2 = dy * dy;
//     double r = std::sqrt(dx2 + dy2);

//     double alpha = 1.0;
//     double thickness = 1.0;
//     double eps = 1.0;
//     double eps1 = 1.0;
//     double eps2 = 1.0;

//     double r0 = alpha * thickness * eps / (eps1 + eps2);

//     double h0 = gsl_sf_struve_H0(r / r0);
//     double y0 = gsl_sf_bessel_Y0(r / r0);

//     double Vrk = - M_PI / ((eps1 + eps2) * r0) * (h0 - y0);

//     return Vrk;
// }

// rytova-keldysh + moire
double DMC::potentialEnergy(const double* position) const {
    const double RYDBERG_FOR_HARTREE = 2.0;
    const double RYDBERG = 13605.7; // em meV
    const double HARTREE = RYDBERG * RYDBERG_FOR_HARTREE; // em meV
    const double a0 = 0.5292; // em Angstrom
    const double PI = 3.14159265358979323846;
    
    const double Vh1 = (-107.1 / 1.0) / HARTREE;
    const double Vh2 = (-(107.1 - 16.9) / 1.0) / HARTREE; 
    const double Ve1 = (-17.3 / 1.0) / HARTREE;
    const double Ve2 = (-(17.3 - 3.5) / 1.0) / HARTREE;

    const double E_field = -300.0  * a0 / HARTREE; 
    const double DIL_C0 = 6.387;
    const double DIL_C1 = 0.544;
    const double DIL_C2 = 0.042;

    double a10 = 3.282;
    double a20 = 3.160;

    double theta = 0.5 * PI / 180;
    double delta = std::abs(a10 - a20) / a10;

    const double MOIRE_LENGTH = a10 / std::sqrt(theta*theta + delta*delta) / a0;
    
    const double K_mag = (4.0 * PI) / (3.0 * MOIRE_LENGTH);

    double xe = position[0];
    double ye = position[1];
    double xh = position[2];
    double yh = position[3];

    double dx_eh = xe - xh;
    double dy_eh = ye - yh;

    double alpha = 1.5;
    double thickness = 6.15 / a0; // em a0
    double eps = 14.0;
    double eps1 = 4.5;
    double eps2 = 4.5;

    double r_eh = std::sqrt(dx_eh * dx_eh + dy_eh * dy_eh + thickness * thickness);

    double r0 = alpha * thickness * eps / (eps1 + eps2);

    double h0 = stvh0(r_eh / r0);
    double y0 = jy0b(r_eh / r0);

    double Vrk = - PI / ((eps1 + eps2) * r0) * (h0 - y0);
    
    const double K1x = K_mag * 1.0;
    const double K1y = K_mag * 0.0;
    const double K2x = K_mag * (-0.5);
    const double K2y = K_mag * (std::sqrt(3.0) / 2.0);
    const double K3x = K_mag * (-0.5);
    const double K3y = K_mag * (-std::sqrt(3.0) / 2.0);

    const double theta_s_div_2 = 2.0 * PI / 3.0;
    const double theta_s = 4.0 * PI / 3.0;

    double K1_dot_re = K1x * xe + K1y * ye;
    double K2_dot_re = K2x * xe + K2y * ye;
    double K3_dot_re = K3x * xe + K3y * ye;

    std::complex<double> f1_e = (std::exp(-1i * K1_dot_re) + 
                                   std::exp(-1i * K2_dot_re) + 
                                   std::exp(-1i * K3_dot_re)) / 3.0;

    std::complex<double> f2_e = (std::exp(-1i * K1_dot_re) + 
                                   std::exp(-1i * (K2_dot_re + theta_s_div_2)) + 
                                   std::exp(-1i * (K3_dot_re + theta_s))) / 3.0;

    double f1_sq_e = std::norm(f1_e);
    double f2_sq_e = std::norm(f2_e);

    double C_e = DIL_C0 + DIL_C1 * f1_sq_e + DIL_C2 * f2_sq_e;
    double V_E_e = E_field * C_e * 0.5;

    double Ve = Ve1 * f1_sq_e + Ve2 * f2_sq_e + V_E_e;
    
    double K1_dot_rh = K1x * xh + K1y * yh;
    double K2_dot_rh = K2x * xh + K2y * yh;
    double K3_dot_rh = K3x * xh + K3y * yh;

    std::complex<double> f1_h = (std::exp(-1i * K1_dot_rh) + 
                                   std::exp(-1i * K2_dot_rh) + 
                                   std::exp(-1i * K3_dot_rh)) / 3.0;

    std::complex<double> f2_h = (std::exp(-1i * K1_dot_rh) + 
                                   std::exp(-1i * (K2_dot_rh + theta_s_div_2)) + 
                                   std::exp(-1i * (K3_dot_rh + theta_s))) / 3.0;

    double f1_sq_h = std::norm(f1_h); // |f1_h|^2
    double f2_sq_h = std::norm(f2_h); // |f2_h|^2

    double C_h = DIL_C0 + DIL_C1 * f1_sq_h + DIL_C2 * f2_sq_h;
    double V_E_h = E_field * C_h * 0.5;

    double Vh = Vh1 * f1_sq_h + Vh2 * f2_sq_h + V_E_h;
    
    return Ve + Vh + Vrk;
}

// double DMC::trialWaveFunction(const double* position) const {
//     double dx = position[0] - position[2];
//     double dy = position[1] - position[3];
//     double dx2 = dx * dx;
//     double dy2 = dy * dy;
//     double r = std::sqrt(dx2 + dy2);
//     if (r < MIN_DISTANCE) r = MIN_DISTANCE;
//     double r2 = r * r;
//     double c1 = -1.0;
//     double c2 = 0.8;
//     double c3 = 0.0;
//     // return std::exp((c1 * r + c2 * r2) / (1 + c3 * r));
//     return std::exp(-c2 * r);
//     // return c1 * r2 * std::log(r) * std::exp(- c2 * r2) - c3 * r * (1 - std::exp(- c2 * r2));
// }


// double DMC::trialWaveFunction(const double* position) const {
//     double dx1 = position[0] - 0;
//     double dy1 = position[1] - 0;
//     double r1 = std::sqrt(dx1 * dx1 + dy1 * dy1);

//     double dx2 = position[2] - 0;
//     double dy2 = position[3] - 0;
//     double r2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

//     double dx12 = position[0] - position[2];
//     double dy12 = position[1] - position[3];
//     double r12 = std::sqrt(dx12 * dx12 + dy12 * dy12);

//     if (r1 < MIN_DISTANCE) r1 = MIN_DISTANCE;
//     if (r2 < MIN_DISTANCE) r2 = MIN_DISTANCE;
//     if (r12 < MIN_DISTANCE) r12 = MIN_DISTANCE;

//     double c1 = 1.86;
//     double c2 = 0.7;

//     return std::exp(-c1 * (r1 + r2)) * std::exp(r12 / (2.0 * (1.0 + c2 * r12)));
// }


//Helium atom 3D
// double DMC::trialWaveFunction(const double* position) const {
//     double x1 = position[0];
//     double y1 = position[1];
//     double z1 = position[2];
    
//     double x2 = position[3];
//     double y2 = position[4];
//     double z2 = position[5];

//     double r1 = std::sqrt(x1*x1 + y1*y1 + z1*z1);
    
//     double r2 = std::sqrt(x2*x2 + y2*y2 + z2*z2);

//     double dx12 = x1 - x2;
//     double dy12 = y1 - y2;
//     double dz12 = z1 - z2;
//     double r12 = std::sqrt(dx12*dx12 + dy12*dy12 + dz12*dz12);

//     double c1 = 1.86;
//     double c2 = 0.7;

//     return std::exp(-c1 * (r1 + r2)) * std::exp(r12 / (2.0 * (1.0 + c2 * r12)));
// }

// rytova keldysh + moire
// double DMC::trialWaveFunction(const double* position) const {
//     double alpha = 0.9;
//     double beta = 0.15;

//     double xe = position[0];
//     double ye = position[1];
//     double xh = position[2];
//     double yh = position[3];

//     const double a0 = 0.5292;

//     double dx_eh = xe - xh;
//     double dy_eh = ye - yh;
//     double thickness = 6.15 / a0;
//     double r_eh = std::sqrt(dx_eh * dx_eh + dy_eh * dy_eh + thickness * thickness);

//     double argexp = - beta * r_eh;

//     double a10 = 3.282;
//     double a20 = 3.160;

//     double theta = 0.5 * PI / 180;
//     double delta = std::abs(a10 - a20) / a10;

//     const double MOIRE_LENGTH = a10 / std::sqrt(theta*theta + delta*delta) / a0;
    
//     const double K_mag = (4.0 * PI) / (3.0 * MOIRE_LENGTH);

//     const double K1x = K_mag * 1.0;
//     const double K1y = K_mag * 0.0;
//     const double K2x = K_mag * (-0.5);
//     const double K2y = K_mag * (std::sqrt(3.0) / 2.0);
//     const double K3x = K_mag * (-0.5);
//     const double K3y = K_mag * (-std::sqrt(3.0) / 2.0);

//     const double theta_s_div_2 = 2.0 * PI / 3.0;
//     const double theta_s = 4.0 * PI / 3.0;

//     double K1_dot_re = K1x * xe + K1y * ye;
//     double K2_dot_re = K2x * xe + K2y * ye;
//     double K3_dot_re = K3x * xe + K3y * ye;

//     std::complex<double> f1_e = (std::exp(-1i * K1_dot_re) + 
//                                  std::exp(-1i * K2_dot_re) + 
//                                  std::exp(-1i * K3_dot_re)) / 3.0;

//     std::complex<double> f2_e = (std::exp(-1i * K1_dot_re) + 
//                                  std::exp(-1i * (K2_dot_re + theta_s_div_2)) + 
//                                  std::exp(-1i * (K3_dot_re + theta_s))) / 3.0;

//     double f1_sq_e = std::norm(f1_e);
//     double f2_sq_e = std::norm(f2_e);

//     double Ve = f1_sq_e + f2_sq_e;
    
//     double K1_dot_rh = K1x * xh + K1y * yh;
//     double K2_dot_rh = K2x * xh + K2y * yh;
//     double K3_dot_rh = K3x * xh + K3y * yh;

//     std::complex<double> f1_h = (std::exp(-1i * K1_dot_rh) + 
//                                  std::exp(-1i * K2_dot_rh) + 
//                                  std::exp(-1i * K3_dot_rh)) / 3.0;

//     std::complex<double> f2_h = (std::exp(-1i * K1_dot_rh) + 
//                                  std::exp(-1i * (K2_dot_rh + theta_s_div_2)) + 
//                                  std::exp(-1i * (K3_dot_rh + theta_s))) / 3.0;

//     double f1_sq_h = std::norm(f1_h);
//     double f2_sq_h = std::norm(f2_h);

//     double Vh = f1_sq_h + f2_sq_h;

//     double exp = std::exp(argexp);
    
//     return exp * (1 - alpha * (Ve + Vh));
// }

double DMC::trialWaveFunction(const double* position) const {
    double alpha = 4.3;
    double beta = 4.35;

    const double RYDBERG_FOR_HARTREE = 2.0;
    const double RYDBERG = 13605.7; // em meV
    const double HARTREE = RYDBERG * RYDBERG_FOR_HARTREE; // em meV

    double xe = position[0];
    double ye = position[1];
    double xh = position[2];
    double yh = position[3];

    const double a0 = 0.5292;

    double dx_eh = xe - xh;
    double dy_eh = ye - yh;
    double thickness = 6.15 / a0;
    double r_eh = std::sqrt(dx_eh * dx_eh + dy_eh * dy_eh + thickness * thickness);

    double argexp = - alpha * r_eh;

    double a10 = 3.282;
    double a20 = 3.160;

    const double E_field = -300.0  * a0 / HARTREE; 
    const double DIL_C0 = 6.387;
    const double DIL_C1 = 0.544;
    const double DIL_C2 = 0.042;

    double theta = 0.5 * PI / 180;
    double delta = std::abs(a10 - a20) / a10;

    const double MOIRE_LENGTH = a10 / std::sqrt(theta*theta + delta*delta) / a0;
    
    const double K_mag = (4.0 * PI) / (3.0 * MOIRE_LENGTH);

    const double K1x = K_mag * 1.0;
    const double K1y = K_mag * 0.0;
    const double K2x = K_mag * (-0.5);
    const double K2y = K_mag * (std::sqrt(3.0) / 2.0);
    const double K3x = K_mag * (-0.5);
    const double K3y = K_mag * (-std::sqrt(3.0) / 2.0);

    const double theta_s_div_2 = 2.0 * PI / 3.0;
    const double theta_s = 4.0 * PI / 3.0;

    double K1_dot_re = K1x * xe + K1y * ye;
    double K2_dot_re = K2x * xe + K2y * ye;
    double K3_dot_re = K3x * xe + K3y * ye;

    std::complex<double> f1_e = (std::exp(-1i * K1_dot_re) + 
                                 std::exp(-1i * K2_dot_re) + 
                                 std::exp(-1i * K3_dot_re)) / 3.0;

    std::complex<double> f2_e = (std::exp(-1i * K1_dot_re) + 
                                 std::exp(-1i * (K2_dot_re + theta_s_div_2)) + 
                                 std::exp(-1i * (K3_dot_re + theta_s))) / 3.0;

    double f1_sq_e = std::norm(f1_e);
    double f2_sq_e = std::norm(f2_e);

    double C_e = DIL_C0 + DIL_C1 * f1_sq_e + DIL_C2 * f2_sq_e;
    double V_E_e = (E_field) * C_e * 0.5;

    double Ve = f1_sq_e + f2_sq_e + V_E_e;
    
    double K1_dot_rh = K1x * xh + K1y * yh;
    double K2_dot_rh = K2x * xh + K2y * yh;
    double K3_dot_rh = K3x * xh + K3y * yh;

    std::complex<double> f1_h = (std::exp(-1i * K1_dot_rh) + 
                                 std::exp(-1i * K2_dot_rh) + 
                                 std::exp(-1i * K3_dot_rh)) / 3.0;

    std::complex<double> f2_h = (std::exp(-1i * K1_dot_rh) + 
                                 std::exp(-1i * (K2_dot_rh + theta_s_div_2)) + 
                                 std::exp(-1i * (K3_dot_rh + theta_s))) / 3.0;

    double f1_sq_h = std::norm(f1_h);
    double f2_sq_h = std::norm(f2_h);

    double C_h = DIL_C0 + DIL_C1 * f1_sq_h + DIL_C2 * f2_sq_h;
    double V_E_h = (E_field) * C_h * 0.5;

    double Vh = f1_sq_h + f2_sq_h + V_E_h;

    double exp = std::exp(argexp);
    
    return exp * std::exp((- beta * (Ve + Vh)));
}

// void DMC::initializeWalkers() {
//     // For simplicity and to directly address sampling from |Psi|^2,
//     // we'll use a basic Metropolis-Hastings-like approach for initialization.

//     std::mt19937 gen = gens[0];
//     std::normal_distribution<double> dist_(0.0, 1.0);
//     std::uniform_real_distribution<double> uniform(0.0, 1.0);

//     const int nEquilibrationSteps = 1e5;
//     const double stepSize = 0.5;
//     const double L = 1.0;

    
//     for (int i = 0; i < nWalkers; ++i) {
//         // Start with random positions for all walkers
//         for (int j = 0; j < nParticles * dim; ++j) {
//             positions[i * stride + j] = 2 * uniform(gen) * L - L;
//         }

//         std::vector<double> currentPosition(positions.begin() + i * stride, positions.begin() + (i + 1) * stride);
//         double currentPsiSquared = trialWaveFunction(&currentPosition[0]);
//         currentPsiSquared *= currentPsiSquared;

//         // Now, equilibrate these walkers to sample from |Psi|^2 using Metropolis-Hastings
//         for (int step = 0; step < nEquilibrationSteps; ++step) {
//             // Propose a new position
//             std::vector<double> proposedPosition(positions.begin() + i * stride, positions.begin() + (i + 1) * stride);
//             for (int j = 0; j < nParticles * dim; ++j) {
//                 proposedPosition[j] += dist_(gen) * stepSize; // Random walk step
//             }

//             double proposedPsiSquared = trialWaveFunction(&proposedPosition[0]);
//             proposedPsiSquared *= proposedPsiSquared;

//             // Acceptance probability
//             double acceptanceRatio = proposedPsiSquared / currentPsiSquared;

//             if (uniform(gen) < std::min(1.0, acceptanceRatio)) {
//                 for (int j = 0; j < nParticles * dim; ++j) {
//                     currentPosition[j] = proposedPosition[j];
//                 }
//                 // currentPosition = proposedPosition;
//                 currentPsiSquared = proposedPsiSquared;
//             }
//             // If rejected, walker stays at currentPosition
//         }

//         std::copy(currentPosition.begin(), currentPosition.end(), positions.begin() + i * stride);
//         // After equilibration, initialize drift and localEnergy for each walker
//         std::vector<double> drift = getDrift(&positions[i * stride]);
//         std::copy(drift.begin(), drift.end(), drifts.begin() + i * stride);
//         localEnergy[i] = getLocalEnergy(&positions[i * stride]);
//         instEnergy += localEnergy[i];
//     }
//     instEnergy = instEnergy / nWalkers;
//     referenceEnergy = instEnergy;
// }

void DMC::initializeWalkers() {
    const int nEquilibrationSteps = 1e4;
    const double stepSize = 5.0;
    const double L = 1.0;

    instEnergy = 0.0; 

    #pragma omp parallel reduction(+:instEnergy)
    {
        int tid = omp_get_thread_num();
        std::mt19937& gen = gens[tid]; // Pega o gerador da thread
        std::normal_distribution<double> dist_(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < nWalkers; ++i) {
            // Start with random positions for all walkers
            for (int j = 0; j < nParticles * dim; ++j) {
                positions[i * stride + j] = 2 * uniform(gen) * L - L;
            }

            // Copia a posição inicial para um vetor temporário
            std::vector<double> currentPosition(positions.begin() + i * stride, positions.begin() + (i + 1) * stride);
            double currentPsiSquared = trialWaveFunction(&currentPosition[0]);
            currentPsiSquared *= currentPsiSquared;

            // Equilíbrio do Metropolis-Hastings para este walker
            for (int step = 0; step < nEquilibrationSteps; ++step) {
                // Propose a new position
                // NOTA: Reutilizar 'proposedPosition' pode ser mais eficiente
                std::vector<double> proposedPosition(currentPosition.begin(), currentPosition.end());
                for (int j = 0; j < nParticles * dim; ++j) {
                    proposedPosition[j] += dist_(gen) * stepSize; // Random walk step
                }

                double proposedPsiSquared = trialWaveFunction(&proposedPosition[0]);
                proposedPsiSquared *= proposedPsiSquared;

                // Acceptance probability
                double acceptanceRatio = proposedPsiSquared / currentPsiSquared;

                if (uniform(gen) < std::min(1.0, acceptanceRatio)) {
                    // Aceita a nova posição
                    currentPosition = proposedPosition; // Cópia de vetores
                    currentPsiSquared = proposedPsiSquared;
                }
                // Se rejeitado, o walker permanece em currentPosition
            }

            // Copia a posição equilibrada final de volta para o array principal
            std::copy(currentPosition.begin(), currentPosition.end(), positions.begin() + i * stride);

            // Inicializa drift e energia local para o walker
            std::vector<double> drift = getDrift(&positions[i * stride]);
            std::copy(drift.begin(), drift.end(), drifts.begin() + i * stride);
            localEnergy[i] = getLocalEnergy(&positions[i * stride]);
            
            // 5. Acumula na variável 'instEnergy' privada da thread
            instEnergy += localEnergy[i];
        }
    } // Fim da região paralela. O OpenMP soma todos os 'instEnergy' privados.

    instEnergy = instEnergy / nWalkers;
    referenceEnergy = instEnergy;
}

// void DMC::run() {
//     int nBlockSteps = 3000;
//     int nStepsPerBlock = 100;

//     std::ofstream fout("dmc.dat");

//     std::deque<double> energyQueue;

//     for(int j = 0; j < nBlockSteps; j++) {
//         double blockEnergy = 0.0;
//         for(int i = 0; i < nStepsPerBlock; i++) {
//             timeStep();
//             blockEnergy += instEnergy;
//         }
//         blockEnergy = blockEnergy / nStepsPerBlock;

//         energyQueue.push_back(blockEnergy);
//         if (energyQueue.size() > 100) {
//             energyQueue.pop_front();
//         }

//         meanEnergy = 0.0;
//         for (double e : energyQueue) {
//             meanEnergy += e;
//         }
//         meanEnergy /= energyQueue.size();

//         double ratio = static_cast<double>(nWalkers) / static_cast<double>(N_WALKERS_TARGET);
//         updateReferenceEnergy(blockEnergy);

//         fout << j << " "
//              << blockEnergy << " "
//              << referenceEnergy << " "
//              << meanEnergy << " "
//              << nWalkers << "\n";
        
        
//         std::cout << "Block " << j
//                   << " | Energy = " << blockEnergy
//                   << " | Ref Energy = " << referenceEnergy
//                   << " | Mean Energy = " << meanEnergy
//                   << " | Population = " << nWalkers
//                   << std::endl;
//     }

//     fout.close();
// }

void DMC::run() {
    int nBlockSteps = 10000;
    int nStepsPerBlock = 100;

    double blockTime = deltaTau * nStepsPerBlock;
    
    const int runningAverageWindow = 500; 

    std::ofstream fout("dmc.dat");
    std::deque<double> energyQueue;
    
    std::vector<double> blockMeanEnergies;
    blockMeanEnergies.reserve(nBlockSteps);

    for (int j = 0; j < nBlockSteps; j++) {
        BlockResult blockResult = blockStep(nStepsPerBlock);

        energyQueue.push_back(blockResult.energy);
        if (energyQueue.size() > runningAverageWindow) {
            energyQueue.pop_front();
        }

        meanEnergy = std::accumulate(energyQueue.begin(), energyQueue.end(), 0.0) / energyQueue.size();

        updateReferenceEnergy(blockResult.energy, blockTime); 

        fout << j << " "
             << blockResult.energy << " "
             << referenceEnergy << " "
             << meanEnergy << " "
             << nWalkers << " "
             << blockResult.variance << " "
             << blockResult.stdError << "\n";

        std::cout << "Block " << std::setw(4) << j
                    << " | Block Energy = " << std::fixed << std::setprecision(8) << blockResult.energy
                    << " | Reference Energy = " << std::fixed << std::setprecision(8) << referenceEnergy
                    << " | Mean Energy = " << std::fixed << std::setprecision(8) << meanEnergy
                    << " | Population = " << std::fixed << nWalkers
                    << " | Variance = " << std::fixed << std::setprecision(8) << blockResult.variance
                    << " | Std Error = " << std::fixed << std::setprecision(8) << blockResult.stdError
                    << std::endl;
    }

    fout.close();

    double variance = 0.0;
    for (double energy : energyQueue) {
        variance += (energy - meanEnergy) * (energy - meanEnergy);
    }
    
    double stdError = 0.0;
    if (energyQueue.size() > 1) {
        variance /= (energyQueue.size() - 1); 
        
        stdError = std::sqrt(variance / energyQueue.size());
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Mean Energy: " << meanEnergy << std::endl;
    std::cout << "Variance: " << variance << std::endl;
    std::cout << "Standard Error: " << stdError << std::endl;
}