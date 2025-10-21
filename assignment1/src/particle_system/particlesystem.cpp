#include "particlesystem.h"
#include <cassert>

#include <glm/glm.hpp>

#define VECTOR_MAX (1024*8)

using namespace glm;
using namespace std;

/* External function for implementing the different numerical solvers */
extern void computeTimeStep(
    float dt, 
    ParticleSystem::eMethod method, 
    std::vector<Point> &points, 
    const ExternalForces& extForces, 
    const shared_ptr<OcTreeStd<size_t, glm::vec2, 2>>& tree );

Point::Point(const vec2 &pos, float mass, float damping, float life_t, float stamp)
    : position(pos)
    , velocity(0.0)
    , force(0.0)
    , lifeTime(life_t)
    , timeStamp(stamp)
    , mass(mass)
    , damping(damping)
    , fixed(false)
{}

Emitter::Emitter(const vec2& pos, float dir, float dev, float rat, float forc)
    : position(pos)
    , direction(dir)
    , deviation(dev)
    , rate(rat)
    , velocity(forc)
    , acc_dt( 0.0 )
{}

ParticleSystem::ParticleSystem()
    : timeStep(0.005)
    , timeAcc( 0.0 )
    , mass(0.15)
    , damping(0.08)
    , stiffness(60.0)
    , lifeTime( 3.0 )
    , pointsCenter( vec2( 0.0) )
    , pointsMin(vec2(0.0))
    , pointsMax(vec2(0.0))
    , method(eMethod::EX_SYMPLECTIC)
{
    points.reserve(VECTOR_MAX);
}

Point &ParticleSystem::createPoint(const vec2 &pos )
{
    assert(points.size() < VECTOR_MAX);

    points.emplace_back(pos, mass, damping, lifeTime, timeAcc );
    return points.back();
}

Emitter& ParticleSystem::createEmitter(const vec2& pos, float dir, float dev, float rate, float force)
{
    assert(emitters.size() < VECTOR_MAX);

    emitters.emplace_back(pos, dir, dev, rate, force);
    return emitters.back();
}

void ParticleSystem::spawnParticles(float dt)
{
    // remove aged particles 
    if (points.size() > 1)
    {
        auto it = points.begin();
        for (; it != points.end(); it++)
            if (it->lifeTime > 0.0 &&
                timeAcc - it->timeStamp > it->lifeTime)
                it = points.erase(it);
    }

    // emit new particles 
    for (Emitter& e : emitters)
    {
        e.acc_dt += dt;

        if (e.rate > 0.0 && 
            e.acc_dt > 1.0 / e.rate)
        {
            float r = rand(gen) * 2.0 - 1.0;
            e.acc_dt = 1.0 / e.rate - e.acc_dt;
            createPoint(e.position).velocity = 
                e.velocity * 
                normalize( (glm::vec2( std::cos(e.direction), std::sin(e.direction) ) + 
                e.deviation * r * glm::vec2( -std::sin(e.direction), std::cos(e.direction) )) );
        }
    }
}


void ParticleSystem::clear()
{
    points.clear();
    emitters.clear();
}

void ParticleSystem::update()
{
    update(timeStep);
}

void ParticleSystem::updateSpatial()
{
    // -- compute spatial stats
    pointsCenter = vec2(0.0f);
    pointsMin = vec2(numeric_limits<float>::max());
    pointsMax = vec2(numeric_limits<float>::min());

    for (const auto& po : points)
    {
        const vec2& p = po.position;

        pointsCenter += p;

        if (p.x < pointsMin.x) pointsMin.x = p.x;
        if (p.y < pointsMin.y) pointsMin.y = p.y;
        if (p.x > pointsMax.x) pointsMax.x = p.x;
        if (p.y > pointsMax.y) pointsMax.y = p.y;
    }

    pointsCenter /= float(points.size());

    // -- update search tree
    tree = make_shared<OcTreeStd<size_t, vec2, 2>>( pointsMin, pointsMax, 32, "TheTree" );
    for (size_t i = 0; i < points.size(); ++i)
        tree->saveInsert(points[i].position, i);
}  

void ParticleSystem::update(float dt)
{
    if(dt <= 0.0) return;

    timeAcc += dt;

    spawnParticles( dt );

    updateSpatial();

    computeTimeStep( dt, method, points, extForces, tree );
}

double ParticleSystem::pointMass() const
{
    return mass;
}

void ParticleSystem::pointMass(float m)
{
    mass = m;
    for(auto& p : points)
    {
        p.mass = mass;
    }
}

double ParticleSystem::pointDamping() const
{
    return damping;
}

void ParticleSystem::pointDamping(float d)
{
    damping = d;
    for(auto& p : points)
    {
        p.damping = damping;
    }
}


float ParticleSystem::pointLifeTime() const
{
    return lifeTime;
}

void ParticleSystem::pointLifeTime(float lt)
{
    lifeTime = lt;
    for (auto& p : points)
    {
        if( !p.fixed ) 
            p.lifeTime = lifeTime;
    }
}
