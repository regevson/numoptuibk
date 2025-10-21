/*******************************************************************
 *
 * author: Marcel Ritter, Nikolaus Rauch
 * date: 6.10.2021
 *
 * Particle System Simulation Wrapper
 *  - uses old legacy opengl
 *  - stores particles and emitters
 *  - provides an octree for spatial range queries
 */

#pragma once

#include "../common/octree.h"

#include <vector>
#include <array>
#include <random>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>


struct Point
{
    Point(const glm::vec2& pos = glm::vec2(), float mass = 0.0f, float damping = 0.0f, float life_time = 0.0f, float timeStamp = 0.0f);

    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 force;

    float mass;
    float damping;
    float lifeTime;
    float timeStamp;

    bool fixed;
};


struct Emitter
{
    Emitter(const glm::vec2& pos = glm::vec2(), 
        float dir = 0.0, float dev = 0.0, float rat = 1.0, float forc = 1.0 );

    glm::vec2 position;
    float direction;
    float deviation;
    float rate;
    float velocity;

    float acc_dt;
};

struct ExternalForces
{
    glm::vec2 wind         { glm::vec2(0.0f) };
    bool enableGround      { true };
    bool enableGravity     { true };
    float gravity          { 9.81f };
    float centerGravity    { 0.0f };
    float flockAvoidance   { 0.0f };
    float flockMatching    { 0.0f };
    float flockCentering   { 0.0f };
    float flockR1          { 0.3f };
    float flockR2          { 0.6f };
    float flockTh1         { glm::pi<float>() / 4.0f };
    float flockTh2         { 3.0f * glm::pi<float>() / 4.0f };
};

struct ParticleSystem
{
    enum class eMethod : int
    {
        EX_EULER = 0,
        EX_SYMPLECTIC,
        EX_VERLET,
        EX_RUNGE4
    };

    const std::array<eMethod, 4> AllMethods = {
        ParticleSystem::eMethod::EX_EULER,
        ParticleSystem::eMethod::EX_SYMPLECTIC,
        ParticleSystem::eMethod::EX_VERLET,
        ParticleSystem::eMethod::EX_RUNGE4
        };


    ParticleSystem();

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator = (const ParticleSystem&) = delete;

    Point& createPoint(const glm::vec2& pos = glm::vec2() );

    Emitter& createEmitter(const glm::vec2& pos = glm::vec2(), 
        float dir = 0.0, float dev = 0.0, float rate = 1.0, float force = 0.0);
    
    void spawnParticles(float dt);


    /* Careful! this invalidates all references to points  */
    void clear();

    void update(float dt);
    void update();

    void updateSpatial();

    /* set global values for new points -> overwrites values from individual points! */
    double pointMass() const;
    void pointMass(float mass);

    double pointDamping() const;
    void pointDamping(float d);

    float pointLifeTime() const;
    void pointLifeTime(float lt);

public:
    float timeStep;
    float timeAcc;

    /* identical mass, damping, and life time for all points */
    float mass;
    float damping;
    float stiffness;
    float lifeTime;

    glm::vec2 pointsCenter;
    glm::vec2 pointsMin;
    glm::vec2 pointsMax;

    std::shared_ptr<OcTreeStd<size_t, glm::vec2, 2>> tree;
    
    eMethod method;
    ExternalForces extForces;

    std::vector<Point> points;
    std::vector<Emitter> emitters;

    std::random_device rdev;
    std::mt19937 gen;
    std::uniform_real_distribution<float> rand;
};

