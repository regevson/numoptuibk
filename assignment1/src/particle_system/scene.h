/*******************************************************************
 *
 * author: Nikolaus Rauch
 * date: 12.02.2021
 *
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "particlesystem.h"

enum class eScene : int
{
    PARABOLA = 0,
};

void setup_scene(ParticleSystem& sim, eScene scene)
{
    sim.clear();

    if(scene == eScene::PARABOLA)
    {
        sim.createEmitter(glm::vec2{ -4.0, -1.5 }, 1.0, 0.0, 1.0, 9.0 );
//        sim.createEmitter(glm::vec2{ 3.0, -1.5 }, 4.0 * pi / 6.0, 0.1, 1.0, 10.0);

        sim.createPoint(glm::vec2{-4.0, -1.5});
//        sim.createPoint(glm::vec2{ 3.0, -1.5}, 0.0);

        sim.points[0].lifeTime = 0.0;
        sim.points[0].fixed = true;
//        sim.points[1].fixed = true;
     }
}

