#include <cstddef>
#include <iostream>
#include <stdlib.h>

#include <vector>
#include <viewer.h>
#include <fstream>
#include <algorithm>

#include "glm/common.hpp"
#include "glm/ext/quaternion_geometric.hpp"
#include "glm/ext/quaternion_trigonometric.hpp"
#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "glm/trigonometric.hpp"
#include "particlesystem.h"
#include "scene.h"

using namespace glm;
using namespace std;



vec2 posUnDamped(float t, const vec2& x0, const vec2& v0, const vec2& acc)
{
    // analytic 2D formula
    // x(t) = x0 + v0 * t + 0.5 * a * t^2
    return x0 + v0 * t + 0.5f * acc * t * t;
}

float posDampedStokes(float t, float x0, float v0, float acc, float kappa_m)
{
    // analytic 1D formula
    // x(t) = x0 + (1/kappa_m) * ( a*t + (v0 - a/kappa_m) * (1 - exp(-kappa_m * t)) )
    return x0 + (1.0f / kappa_m) * (acc * t + (v0 - acc / kappa_m) * (1.0f - glm::exp(-kappa_m * t)));
}

glm::vec2 computeFlockingForce (Point& p_i, Point& p_j, const ExternalForces& eF)
{
    // Pairwise accelerations (p_i to p_j) for all i,j:
    // a_a_ij = - k_a / d_ij * nrm(x_ij)
    // a_v_ij = k_v * (v_j - v_i)
    // a_c_ij = k_c * x_ij
    // With: x_ij = x_j - x_i
    // The k's are provided in externalForces.

    // Weighted sum for all accelerations a_# (a_a, a_v, a_c):
    // a_#_i = sum[w_t(t_ij) * w_d(d_ij) * a_#_ij]

    // The weights are calculated the following way:
    //          { 1                   if d < r_1,
    // w_d(d) = { (r_2-d)/(r_2-r_1)   if r_1 <= d <= r_2,
    //          { 0                   if d > r_2.
    // d(istance)
    // r(adius)

    //              { 1                                 if -th_1/2 < abs(th) < th_1/2
    // w_th(th) =   { (th_2/2-abs(th))/((th_2-th_1)/2)  if w_1/2 <= abs(th) <= w_2/2
    //              { 0                                 if abs(th) > th_2/2
    // th(eta) = angle

    glm::vec2 x_ij = p_j.position - p_i.position;
    
    // Distance and angle of vectors.
    float d_ij = glm::length(x_ij);
    if (d_ij == 0.0f) return glm::vec2(0.0f); //<- Avoid division by zero

    // Check to avoid NaN values.
    float th_ij = 0.0f;
    float v_i_len = glm::length(p_i.velocity);
    if (v_i_len > 0.0001f) {
        float cos_t = glm::clamp(glm::dot(p_i.velocity / v_i_len, x_ij / d_ij), -1.0f, 1.0f);
        th_ij = glm::acos(cos_t);
    }

    // If one of the weights is zero, the acceleration between the points is zero as well.
    if (d_ij > eF.flockR2) return glm::vec2(0.0f);
    if (abs(th_ij) > eF.flockTh2 * 0.5f) return glm::vec2(0.0f);

    // Compute accerleration.
    glm::vec2 a_a_ij = -eF.flockAvoidance/d_ij * normalize(x_ij);
    glm::vec2 a_v_ij = eF.flockMatching*(p_j.velocity-p_i.velocity);
    glm::vec2 a_c_ij = eF.flockCentering * x_ij;

    // Compute weights.
    float w_d_ij = (d_ij < eF.flockR1) ? 1.0f : (eF.flockR2-d_ij)/(eF.flockR2-eF.flockR1);
    float w_th_ij = (-eF.flockTh1*0.5f < abs(th_ij) && abs(th_ij) < eF.flockTh1 * 0.5f) ? 
                        1.0f : (eF.flockTh2*0.5f-abs(th_ij))/((eF.flockTh2-eF.flockTh1)*0.5f);

    // Sum up partial accererations.
    glm::vec2 A = glm::vec2(0.0f);
    A += w_th_ij * w_d_ij * a_a_ij;
    A += w_th_ij * w_d_ij * a_v_ij;
    A += w_th_ij * w_d_ij * a_c_ij; 

    // F = m * A
    return p_i.mass * A;
}

void updateFlockingBruteForce(vector<Point>& points, const ExternalForces& eF, const shared_ptr<OcTreeStd<size_t, glm::vec2, 2>>& tree)
{
    for (int i = 0; i < points.size(); i++) {
        if (points[i].fixed) continue;
        for (int j = 0; j < points.size(); j++) {
            if (i==j) continue;
            if (points[j].fixed) continue;
            points[i].force += computeFlockingForce(points[i], points[j], eF);
        }
    }
}

void updateFlockingOctree(vector<Point>& points, const ExternalForces& eF, const shared_ptr<OcTreeStd<size_t, glm::vec2, 2>>& tree)
{
    for (size_t i = 0; i < points.size() ; i++) {
        Point p_i = points[i];
        if (p_i.fixed) continue;
        multimap<double , size_t> res ; //<−prepare result container
        tree->getEuclideanRangeFine (p_i.position, eF.flockR2, res) ; //<−query
        
        glm::vec2 a_a_i = glm::vec2(0.0f);
        glm::vec2 a_v_i = glm::vec2(0.0f);
        glm::vec2 a_c_i = glm::vec2(0.0f);

        for (const auto& r : res) { //<−iterate close neighbors
            const size_t j = r.second;
            if (i == j) continue;
            Point p_j = points[j];
            if (p_j.fixed) continue;

            p_i.force += computeFlockingForce(p_i, p_j, eF);
        }
    }
}


void updateExtForces(vector<Point>& points, const ExternalForces& eF, const shared_ptr<OcTreeStd<size_t, glm::vec2, 2>>& tree )
{   
    // constants
    const float g        = 9.81f;          // gravitational acceleration
    const float groundY  = -1.5f;          // ground plane height
    const float kpen     = 10000.0f;       // penalty stiffness for ground collision
    const glm::vec2 gravitationalCenter   = glm::vec2(0.0f, 0.0f); // gravitational center

    for (Point& p : points)
    {
        if (p.fixed) continue; // skip fixed particles

        glm::vec2 F(0.0f);

        // gravity
        if (eF.enableGravity)
            F += p.mass * glm::vec2(0.0f, -g);

        // wind
        F += eF.wind;

        // ground collision (penalty force)
        if (eF.enableGround)
        {
            float penetration = groundY - p.position.y; // >0 when below ground
            if (penetration > 0.0f)
                F += kpen * penetration * glm::vec2(0.0f, 1.0f);
        }

        // gravitational center (using Newton's law)
        if (eF.centerGravity > 0.0f)
        {
            glm::vec2 d  = gravitationalCenter - p.position;
            float r2 = glm::dot(d, d);
            F += (eF.centerGravity * p.mass / r2) * glm::normalize(d);
        }

        p.force = F;
    }
    updateFlockingOctree(points, eF, tree);
}

glm::vec2 getDampedAcceleration(const Point& point) {
    // a(t) = A - (k/m)v(t)
    // A = F_ext / m
    glm::vec2 a = (point.force / point.mass) - (point.damping * point.velocity) / point.mass;
    return a;
}

void computeTimeStep(float dt, 
    ParticleSystem::eMethod method, 
    vector<Point> &points, 
    const ExternalForces& extForces, 
    const shared_ptr<OcTreeStd<size_t, glm::vec2, 2>>& tree)
{
    // cache variables for the next time step by local static variables
    static uint64_t count = 0;
    static vector<Point> oldPoints;


    // we have 2nd order ODE: x''(t) = a(t, x(t), x'(t))
    // we can rewrite to a first order system: 
    //   x'(t) = v(t)                ->  fx(t,v(t)) = v(t)
    //   v'(t) = a(t, x(t), v(t))    ->  fv(t,x(t),v(t)) = a(t, x(t), v(t))


    switch (method)
    {
        case ParticleSystem::eMethod::EX_EULER:
        {
            // -- HERE: update points call updateExtForces() to update forces
            updateExtForces(points, extForces, tree);

            // explicit (forward) Euler: 
            //   x(t+1) = x(t) + dt * fx(t, v(t)) = x(t) + dt * v(t)
            //                                    = x_n + dt * v_n
            //   v(t+1) = v(t) + dt * fv(t, x(t), v(t)) = v(t) + dt * a(t, x(t), v(t))
            //                                          = v_n + dt * a(x_n, v_n)

            for (Point& point: points) {

                if (point.fixed) continue;

                // Update position based on *current* velocity
                point.position += dt * point.velocity;

                // Update velocity based on  a(t) = A - (k/m)v(t)
                //                           A = F_ext / m
                point.velocity += dt * getDampedAcceleration(point);
            }
            break;
        }

        case ParticleSystem::eMethod::EX_SYMPLECTIC:
        {
            updateExtForces(points, extForces, tree);

            // symplectic (semi-implicit) Euler: 
            //   v_{n+1} = v_n + dt * a(x_n, v_n)
            //   x_{n+1} = x_n + dt * v_{n+1}

            for (Point& point: points) {

                if (point.fixed) continue;

                // Update velocity first based on a = F / m
                point.velocity += dt * getDampedAcceleration(point);

                // Update position based on the *new* velocity
                point.position += dt * point.velocity;
            }
            break;
        }

        case ParticleSystem::eMethod::EX_VERLET:
        {   

            // Verlet method: 
            //    x_{n+1} = 2 x_n - x_{n-1} + a_n * dt^2
            //    v_{n+1} ≈ (x_{n+1} - x_{n-1}) / (2 dt)   (central difference)

            // initialize oldPoints for first time step
            if (oldPoints.size() != points.size())
            {
                oldPoints = points; // copy all info to oldPoints (mass, damping, ... )
                for (int i = 0; i < points.size(); ++i)
                {
                    // euler step applied backwards: x_{n-1} = x_n - v_n * dt
                    oldPoints[i].position = points[i].position - points[i].velocity * dt;
                }
            }

            updateExtForces(points, extForces, tree);

            for (int i = 0; i < points.size(); ++i)
            {

                Point& point = points[i];
                Point& oldPoint = oldPoints[i];

                if (point.fixed) continue;

                glm::vec2 a_n = getDampedAcceleration(point);
                glm::vec2& x_n   = point.position;
                glm::vec2& x_n_1 = oldPoint.position;

                // position-verlet update: x_{n+1} = 2 x_n - x_{n-1} + a_n * dt^2
                glm::vec2 x_n_plus_1 = 2.0f * x_n - x_n_1 + a_n * (dt * dt);
                // velocity-update (central difference): v_{n+1} ≈ (x_{n+1} - x_{n-1}) / (2 dt)
                glm::vec2 v_n_plus_1 = (x_n_plus_1 - x_n_1) / (2.0f * dt);

                // update old and new positions for next timestep
                oldPoint.position = x_n;
                point.position = x_n_plus_1;
                point.velocity = v_n_plus_1;
            }

            break;
        }
        
        case ParticleSystem::eMethod::EX_RUNGE4:
        {
            updateExtForces(points, extForces, tree);
            for (Point& point: points) {
                std::vector<Point> tmp_vec = {point};

                vector<glm::vec2> k(4);
                vector<glm::vec2> l(4);

                glm::vec2 vel_org = point.velocity;
                glm::vec2 pos_org = point.position;

                // k_1 & l_1
                k[0] = vel_org;
                l[0] = getDampedAcceleration(point);

                for (int i = 1; i < 4; i++) {
                    // k_n; then compute l_n based on new position and velocity
                    float step_size = (i==3) ? dt : dt*0.5f;
                    k[i] = vel_org + step_size * l[i-1];
                    point.velocity = k[i];
                    point.position = pos_org + step_size * k[i-1];
                    tmp_vec[0] = point;
                    updateExtForces(tmp_vec, extForces, tree);
                    point.force = tmp_vec[0].force;
                    l[i] = getDampedAcceleration(point);
                }

                // final update of position and velocity
                point.velocity = vel_org + dt/6.0f * (l[0] + 2.0f*l[1] + 2.0f*l[2] + l[3]);
                point.position = pos_org + dt/6.0f * (k[0] + 2.0f*k[1] + 2.0f*k[2] + k[3]);
            }
            break;
        }

        count++;

    }
}


float computeKineticEnergy(const ParticleSystem& sim)
{      
    double kineticEnergy = 0.0f;
    for (const Point& point: sim.points) {
        kineticEnergy += 0.5f * point.mass * glm::dot(point.velocity, point.velocity);
    }

    return kineticEnergy;
}

float computePotentialEnergy(const ParticleSystem& sim)
{
    if (!sim.extForces.enableGravity) return 0.0f;

    const float g = 9.81f;
    const float groundY = -1.5f;

    float potentialEnergy = 0.0f;
    for (const Point& point : sim.points)
    {
        if (point.fixed) continue;
        float h = point.position.y - groundY;   // height over ground
        potentialEnergy += g * point.mass * h;
    }
    return potentialEnergy;
}

vec2 angleToVec2(float ang)
{
    return vec2(cos(ang), sin(ang));
}

vec2 computeAnalytic(const vec2& x0, const vec2& v0, float mass, const vec2& wind, float damping, float t)
{
    vec2 acc = vec2(0.0, -9.81) + wind / mass;
    float kappa = damping / mass;

    if (abs(damping) < 1e-4)
        return posUnDamped(t, x0, v0, acc);
    else
    {
        float xtx = posDampedStokes(t, x0.x, v0.x, acc.x, kappa);
        float xty = posDampedStokes(t, x0.y, v0.y, acc.y, kappa);
        return vec2(xtx, xty);
    }
}

std::pair<float, double> errorAtImpact(ParticleSystem& sim, float dt, ParticleSystem::eMethod method)
{
    // store current configuration (we'll restore it at the end)
    const auto method0 = sim.method;
    const float dt0 = sim.timeStep;

    // reset scene
    setup_scene(sim, eScene::PARABOLA);

    // configure integrator and time step
    sim.method = method;
    sim.timeStep = dt;

    if (sim.emitters.size() == 0) return { 0.0f, 0.0 };

    const Emitter& e = sim.emitters[0];

    vec2 v0 = e.velocity * angleToVec2(e.direction);

    // create one new particle to be tracked for measurement
    sim.createPoint(e.position).velocity = v0;
    Point& p = sim.points[sim.points.size() - 1];

    // store actual life time and set to forever
    for (auto& ps : sim.points)
        ps.lifeTime = 0.0;

    // timing start (seconds) -> milliseconds
    const double tStart = glfwGetTime();

    // prepare measure state
    float local_t = 0.0;

    auto PAnalytic = vec2(0.0);
    auto lastP = p.position;
    float finalError = 0.0f;

    // simulation loop for one shot and output error for one particle
    size_t safetyCount = 0;
    while ((safetyCount < 2 || p.position.y > -1.5f) &&
           safetyCount < 100000)
    {
        lastP = p.position;
        sim.update();

        // Interpolate between the overshooting point and the last
        // to retrieve the error at the impact position ( y = -1.5 )
        if (safetyCount > 2 && p.position.y < -1.5f)
        {
            float d1 = lastP.y + 1.5f;
            float d2 = -1.5f - p.position.y;
            float w1 = (d1 / (d1 + d2));
            float w2 = 1.0f - w1;

            vec2 interpolatedP = lastP * w2 + p.position * w1;
            float interpolatedT = local_t * w2 + (local_t + sim.timeStep) * w1;

            PAnalytic = computeAnalytic(e.position, v0, sim.mass, sim.extForces.wind, sim.damping, interpolatedT);

            // NEW: compute the final error (between analytic and interpolated)
            finalError = glm::distance(interpolatedP, PAnalytic);
            break;
        }

        local_t += sim.timeStep;
        safetyCount++;
    }

    const double tEnd = glfwGetTime();
    const double elapsedMs = (tEnd - tStart) * 1000.0;

    // restore previous configuration
    setup_scene(sim, eScene::PARABOLA);
    sim.method = method0;
    sim.timeStep = dt0;

    return { finalError, elapsedMs };
}


void exportStepSizeSeries(ParticleSystem& sim, const std::vector<float>& dts)
{
    const std::string filename = "OptiNum_A1_StepSizeSeries.csv";
    std::ofstream out(filename, std::ios_base::trunc);

    out << "dt"
        << ", ExEuler_error, ExEuler_ms"
        << ", ExSymplectic_error, ExSymplectic_ms"
        << ", ExVerlet_error, ExVerlet_ms"
        << ", ExRK4_error, ExRK4_ms"
        << "\n";

    for (float dt : dts)
    {
        out << dt;

        auto [err_euler, ms_euler] = errorAtImpact(sim, dt, ParticleSystem::eMethod::EX_EULER);
        out << ", " << err_euler << ", " << ms_euler;

        auto [err_symp, ms_symp] = errorAtImpact(sim, dt, ParticleSystem::eMethod::EX_SYMPLECTIC);
        out << ", " << err_symp << ", " << ms_symp;

        auto [err_verlet, ms_verlet] = errorAtImpact(sim, dt, ParticleSystem::eMethod::EX_VERLET);
        out << ", " << err_verlet << ", " << ms_verlet;

        auto [err_rk4, ms_rk4] = errorAtImpact(sim, dt, ParticleSystem::eMethod::EX_RUNGE4);
        out << ", " << err_rk4 << ", " << ms_rk4;

        out << "\n";
    }

    out.close();
}


void exportErrorOverTime(ParticleSystem& sim, string post_fix)
{
    if (sim.emitters.size() == 0) return;

    const Emitter& e = sim.emitters[0];

    vec2 v0 = e.velocity * angleToVec2(e.direction);

    // create one new particle to be tracked for measurement
    sim.createPoint(e.position).velocity = v0;
    Point& p = sim.points[sim.points.size() - 1];

    // store actual life time and set to forever
    float lifeTimeStore = p.lifeTime;
    for (auto& ps : sim.points)
        ps.lifeTime = 0.0;

    // prepare measure state
    float local_t = 0.0;

    // prepare out file
    ofstream out_file;
    out_file.open("OptiNum_A1Errors_" + post_fix + ".csv", std::ios_base::trunc);
    out_file << "time, x, y, aX, aY, DX, DY, Error" << endl;

    auto PAnalytic = vec2(0.0);
    auto lastP = vec2(0.0);

    // simulation loop for one shot and output error for one particle
    size_t safetyCount = 0;
    while ((safetyCount < 2 || p.position.y > -1.5f) &&
        safetyCount < 10000)
    {
        PAnalytic = computeAnalytic(e.position, v0, sim.mass, sim.extForces.wind, sim.damping, local_t);

        out_file << local_t << ", " << p.position.x << ", " << p.position.y << ", "
            << PAnalytic.x << ", " << PAnalytic.y << ", " << p.position.x - PAnalytic.x << ", "
            << p.position.y - PAnalytic.y << ", " << glm::distance(p.position, PAnalytic) << endl;

        lastP = p.position;
        sim.update();

        // Interpolate between the overshooting point and the last
        // to retrieve the error at the impact position ( y = -1.5 )
        if (safetyCount > 2 && p.position.y < -1.5f)
        {
            float d1 = lastP.y + 1.5f;
            float d2 = -1.5f - p.position.y;
            float w1 = (d1 / (d1 + d2));
            float w2 = 1.0f - w1;

            vec2 interpolatedP = lastP * w2 + p.position * w1;
            float interpolatedT = local_t * w2 + (local_t + sim.timeStep) * w1;

            PAnalytic = computeAnalytic(e.position, v0, sim.mass, sim.extForces.wind, sim.damping, interpolatedT);

            out_file << interpolatedT << ", " << interpolatedP.x << ", " << interpolatedP.y << ", "
                << PAnalytic.x << ", " << PAnalytic.y << ", " << interpolatedP.x - PAnalytic.x << ", "
                << interpolatedP.y - PAnalytic.y << ", " << glm::distance(interpolatedP, PAnalytic) << endl;

            break;
        }

        local_t += sim.timeStep;
        safetyCount++;
    }

    out_file.close();

    // restore life times
    for (auto& ps : sim.points)
        if (!ps.fixed)
            ps.lifeTime = lifeTimeStore;
}

vector<vec2> computeAnalyticTrajectory(const vec2& x0, const vec2& v0, float mass, const vec2& wind, float damping, float dt)
{
    vector<vec2> analytic;

    auto xt = vec2(0.0);
    auto t = 0.0f;
    int count = 0;

    while (xt.y >= -1.5 && count < 5000)
    {
        xt = computeAnalytic(x0, v0, mass, wind, damping, t);
        analytic.push_back(xt);
        t += dt;
        count++;
    }

    return analytic;
}

vector<pair<vec2, vec2>> lineAsLines(const vector<vec2>& line, int segments)
{
    vector<pair<vec2, vec2>> res;
    size_t inc = line.size() / segments;
    for (size_t i = 0; i < line.size() - inc; i += inc)
        res.push_back({ line[i], line[i + inc] });

    res.push_back({ res[res.size() - 1].first, line[line.size() - 1] } );
    return res;
}

vector<pair<vec2, vec2>> emitterAsLines(const Emitter& e, float radius = 0.2 )
{
    vec2 dir  = { std::cos(e.direction), std::sin(e.direction) };
    vec2 orth = { -std::sin(e.direction), std::cos(e.direction) };

    vec2 B = e.position + radius * dir;
    vec2 C = B - 0.5f * e.deviation * orth;
    vec2 D = B + 0.5f * e.deviation * orth;

    vector<pair<vec2, vec2>> res =
        {
            pair<vec2, vec2>( e.position, B ),
            { e.position, C },
            { e.position, D },
            { C, D }
        };

    return res;
}

void runScene(ParticleSystem& sim, ParticleSystem::eMethod m, float dt, const std::string& postfix)
{
    setup_scene(sim, eScene::PARABOLA);
    sim.method = m;
    sim.timeStep = dt;
    exportErrorOverTime(sim, postfix);
}

void exportErrors()
{
    ParticleSystem test_sim;

    test_sim.extForces.wind = glm::vec2(0.30f, 0.10f);
    test_sim.damping = 0.20f;

    runScene(test_sim, ParticleSystem::eMethod::EX_EULER, 0.05f,   "Euler_dt0_05");
    runScene(test_sim, ParticleSystem::eMethod::EX_EULER, 0.001f,  "Euler_dt0_001");
    runScene(test_sim, ParticleSystem::eMethod::EX_SYMPLECTIC, 0.05f,   "Symplectic_dt0_05");
    runScene(test_sim, ParticleSystem::eMethod::EX_SYMPLECTIC, 0.001f,  "Symplectic_dt0_001");
    runScene(test_sim, ParticleSystem::eMethod::EX_VERLET, 0.05f,   "Verlet_dt0_05");
    runScene(test_sim, ParticleSystem::eMethod::EX_VERLET, 0.001f,  "Verlet_dt0_001");
    runScene(test_sim, ParticleSystem::eMethod::EX_RUNGE4, 0.05f,   "Runge4_dt0_05");
    runScene(test_sim, ParticleSystem::eMethod::EX_RUNGE4, 0.001f,  "Runge4_dt0_001");
}


int main(int argc, char** argv)
{
    exportErrors();

    Viewer viewer;
    viewer.mWindow.title = "01 Particle System";
    viewer.mWindow.width = 1280;
    viewer.mWindow.height = 720;
    viewer.mWindow.vsync = true;
    viewer.mWindow.mHDPI = false;    // <-- adjust for displaysize: 4k(true), HD(false) 

    /* Simulation Handler */
    ParticleSystem sim;
    int updatesPerFrame = 4;

    /* Display Helpers */
    bool showAnalytic = false;
    vector<vec2> analyticTrajectory;
    vector<pair<vec2, vec2>> analyticTrajectoryLines;

    vector<float> Energy;
    vector<float> EnergyKin;
    vector<float> EnergyPot;

    viewer.onInit([&]()
    {
        viewer.mRender.scale = 128.0f;
        viewer.mRender.pointRadius = 0.05f;
        viewer.mRender.lineWidth = 0.01f;

        /* load scene */
        setup_scene(sim, eScene::PARABOLA);
    });

    viewer.onUpdate([&](Window& window, double dt)
    {
            for (int i = 0; i < updatesPerFrame; i++)
            {
                sim.update();

                float Ek = computeKineticEnergy(sim);
                float Ep = computePotentialEnergy(sim);

                if (Energy.size() >= 512 )
                {
                    Energy.erase(Energy.begin(), Energy.begin() + 1);
                    EnergyKin.erase(EnergyKin.begin(), EnergyKin.begin() + 1);
                    EnergyPot.erase(EnergyPot.begin(), EnergyPot.begin() + 1);
                }
                EnergyKin.push_back( Ek );
                EnergyPot.push_back( Ep );
                Energy.push_back( Ek + Ep );
            }
    });

    viewer.onDraw([&](Window& window, double dt)
    {
        /* render analytic */
        if (showAnalytic)
        {
            viewer.drawLines(analyticTrajectoryLines.begin(), analyticTrajectoryLines.end(), [&](const auto& line, glm::vec2& start, glm::vec2& end, glm::vec4& color)
                {
                    start = line.first;
                    end = line.second;
                    color = { 0.5, 0.5, 0.5, 1.0 };
                    return true;
                });
        }

        /* render points */
        viewer.drawPoints(sim.points.begin(), sim.points.end(), [&](const auto& point, glm::vec2& coord, glm::vec4& color)
        {
            coord = point.position;
            color = !point.fixed ? glm::vec4(0.0, 0.0, 1.0, 1.0) : glm::vec4(1.0, 0.0, 0.0, 1.0);
            return true;
        });

        /* render emitters */
        std::vector<std::pair<glm::vec2, glm::vec2>> em_lines;
        for (const Emitter& e : sim.emitters)
        {
            std::vector<std::pair<glm::vec2, glm::vec2>> tmp = emitterAsLines(e, 0.4f);
            em_lines.insert(em_lines.end(), tmp.begin(), tmp.end() );
        }

        viewer.drawLines( em_lines.begin(), em_lines.end(), [&](const auto& line, glm::vec2& start, glm::vec2& end, glm::vec4& color)
            {
                start = line.first;
                end = line.second;
                color = { 0.5, 0.5, 0.7, 1.0 };
                return true;
            });


        /* render ground */
        if (sim.extForces.enableGround)
            viewer.drawBoundary(glm::vec2{0, -1.5f}, {0, 1}, 10.0f, 0.5f);
    });

    viewer.onGui([&](Window& window, double dt)
        {
            ImGui::Begin("Settings", nullptr,
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoCollapse);
            {
                if (ImGui::CollapsingHeader("Simulation"))
                {
                    ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.6f, 1.0f), "FPS: %4.2f, Nr Particles %llu", viewer.fps(), sim.points.size() );

                    ImGui::Separator();

                    if (ImGui::Button("reset scene"))
                        setup_scene(sim, eScene::PARABOLA);

                    ImGui::SliderInt("UpdatesPerFrame", &updatesPerFrame, 1, 10);
                    if (ImGui::SliderFloat("LifeTime", &sim.lifeTime, 1.0f, 100.0f))
                        sim.pointLifeTime(sim.lifeTime);

                }
                ImGui::Separator();
                bool inChanged = false;

                if (ImGui::CollapsingHeader("External Forces"))
                {
                    inChanged |= ImGui::Checkbox("Gravity", &sim.extForces.enableGravity);
                    inChanged |= ImGui::Checkbox("Ground", &sim.extForces.enableGround);

                    inChanged |= ImGui::SliderFloat("Wind.X", &sim.extForces.wind.x, -1.0f, 1.0f);
                    inChanged |= ImGui::SliderFloat("Wind.Y", &sim.extForces.wind.y, -1.0f, 1.0f);

                    if (ImGui::SliderFloat("Damping", &sim.damping, 0.0f, 1.0f))
                    {
                        inChanged |= true;
                        sim.pointDamping(sim.damping);
                    }

                    inChanged |= ImGui::SliderFloat("CenterGrav", &sim.extForces.centerGravity, 0.0f, 200.0f);

                    inChanged |= ImGui::SliderFloat("F_Avoidance", &sim.extForces.flockAvoidance, 0.0f, 10.0f);
                    inChanged |= ImGui::SliderFloat("F_Matching", &sim.extForces.flockMatching, 0.0f, 20.0f);
                    inChanged |= ImGui::SliderFloat("F_Centering", &sim.extForces.flockCentering, 0.0f, 120.0f);
                    inChanged |= ImGui::SliderFloat("F_R1", &sim.extForces.flockR1, 0.0f, 2.0f);
                    inChanged |= ImGui::SliderFloat("F_R2", &sim.extForces.flockR2, 0.0f, 2.0f);
                    inChanged |= ImGui::SliderFloat("F_Th1", &sim.extForces.flockTh1, 0.0f, 2.0f);
                    inChanged |= ImGui::SliderFloat("F_Th21", &sim.extForces.flockTh2, 0.0f, 3.1415f);
                    inChanged |= true;

                }

            ImGui::Separator();
            
            int _method = static_cast<int>(sim.method);

            if (ImGui::CollapsingHeader("Integration"))
            {
                ImGui::RadioButton("ExEuler", &_method, 0); ImGui::SameLine();
                ImGui::RadioButton("ExSymplectic", &_method, 1); ImGui::SameLine();
                ImGui::RadioButton("ExVerlet", &_method, 2); ImGui::SameLine();
                ImGui::RadioButton("ExRk4", &_method, 3);
                sim.method = static_cast<ParticleSystem::eMethod>(_method);

                ImGui::SliderFloat("Timestep", &sim.timeStep, 0.000001f, 0.05f, "%.6f");
            }


            ImGui::Separator();

            if (ImGui::CollapsingHeader("Emitter"))
            {
                if (sim.emitters.size() > 0)
                {
                    Emitter& e = sim.emitters[0];

                    inChanged |= ImGui::SliderFloat("Direction", &e.direction, 0.0f, glm::pi<float>() / 2.0f, "%.6f");
                    inChanged |= ImGui::SliderFloat("Deviation", &e.deviation, 0.0f, 1.0f, "%.6f");
                    inChanged |= ImGui::SliderFloat("Rate", &e.rate, 0.00001f, 60.0f, "%.1f");
                    inChanged |= ImGui::SliderFloat("Velocity", &e.velocity, 0.0f, 20.0f, "%.1f");
                }
            }

            ImGui::Checkbox("Analytic", &showAnalytic);

            if (showAnalytic)
            {
                const Emitter& e = sim.emitters[0];
                const vec2& x0 = e.position;
                const vec2 v0 = e.velocity * angleToVec2(e.direction);

                analyticTrajectory      = computeAnalyticTrajectory(x0, v0, sim.mass, sim.extForces.wind, sim.damping, 0.001f);
                analyticTrajectoryLines = lineAsLines(analyticTrajectory, 256);
            }

            if (showAnalytic)
            {
                static char buf[256] = "";
                ImGui::Text("   "); ImGui::SameLine();
                ImGui::InputText("ExportPostFix", buf, 256);
                ImGui::Text("   "); ImGui::SameLine();
                if (ImGui::Button("ExportError"))
                {
                    exportErrorOverTime(sim, string(buf));
                }

                ImGui::SameLine();
                ImGui::Text("   "); ImGui::SameLine();

                if (ImGui::Button("ExportStepsizeSeries"))
                {
                    std::vector<float> dts = { 0.05f, 0.02f, 0.01f, 0.005f, 0.002f, 0.001f };
                    exportStepSizeSeries(sim, dts);
                    // reset scene
                    setup_scene(sim, eScene::PARABOLA);
                    setup_scene(sim, eScene::PARABOLA);
                }
            }


            if (ImGui::CollapsingHeader("Energy"))
            {
                if (Energy.size() > 0)
                {
                    char buf1[512] = "";
                    sprintf(buf1, "Energy: %.2f + %.2f = %.2f ", *EnergyKin.rbegin(), *EnergyPot.rbegin(), *Energy.rbegin());
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.8f, 1.0f), buf1);

                    ImVec2 wS = ImGui::GetWindowSize();
                    char buf2[512] = "";
                    float maxEnergy = *max_element(Energy.begin(), Energy.end());
                    sprintf(buf2, "(max = %.1f)", maxEnergy);
                    ImGui::PlotLines(buf2, Energy.data(), int(Energy.size()), 0, "", 0.0f, maxEnergy, ImVec2(wS.x * 0.65f, wS.x * 0.2f));
                }
            }
        }
        ImGui::End();
    });

    viewer.onKey([&] (Window& window, Keyboard& keyboard, int key, int mod, bool press)
    {
        if(key == GLFW_KEY_ESCAPE && press) { window.close(true); }

        /* reload scene */
        if(key == GLFW_KEY_R && press)
        {
            setup_scene(sim, eScene::PARABOLA);
        }

        /* zoom in */
        if(key == GLFW_KEY_W && press) { viewer.mRender.scale += 25.0f; }

        /* zoom out */
        if(key == GLFW_KEY_S && press) { viewer.mRender.scale -= 25.0f; }

    });

    viewer.run();

    return EXIT_SUCCESS;
}
