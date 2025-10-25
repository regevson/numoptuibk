#include <stdlib.h>

#include <viewer.h>
#include <fstream>
#include <algorithm>

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



void updateExtForces(vector<Point>& points, const ExternalForces& eF, const shared_ptr<OcTreeStd<size_t, glm::vec2, 2>>& tree )
{
    const double stiffnessPenaltyK = 10000.0;
    const double yGround = -1.5;

    const glm::vec2 zeroPoint = glm::vec2(0.0,0.0);

    for (Point& point: points) {
        // Skip fixed points
        if (point.fixed) continue;
        
        glm::vec2 force = glm::vec2(0.0, 0.0);

        // Universal Gravity
        force += ((eF.centerGravity * point.mass) / glm::distance2(zeroPoint, point.position)) * glm::normalize(zeroPoint-point.position);
        // Uniform Gravity 
        force += glm::vec2(0.0, -eF.gravity * point.mass);
        // Wind
        force += eF.wind;
        // Collision Force (Ground)
        if (point.position[1] < yGround) {
            force += glm::vec2(0.0, stiffnessPenaltyK * (yGround-point.position[1]));
        }

        // Update point's force
        point.force = force;
    }
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

    switch (method)
    {
        case ParticleSystem::eMethod::EX_EULER:
        {
            // -- HERE: update points call updateExtForces() to update forces
            updateExtForces(points, extForces, tree);
            for (Point& point: points) {
                // Update position based on *current* velocity
                point.position += dt * point.velocity;

                // Update velocity based on a = F / m
                point.velocity += dt * (point.force / point.mass);
            }
        }

        case ParticleSystem::eMethod::EX_SYMPLECTIC:
        {
            // -- HERE: update points call updateExtForces() to update forces
            break;
        }

        case ParticleSystem::eMethod::EX_VERLET:
        {
            // -- HERE: update points call updateExtForces() to update forces
            break;
        }
        
        case ParticleSystem::eMethod::EX_RUNGE4:
        {
            // -- HERE: update points call updateExtForces() to update forces
        }
        
        count++;

    }
}


float computeKineticEnergy(const ParticleSystem& sim)
{
    // -- HERE: compute kinetic energy
    return 0.0f;
}

float computePotentialEnergy(const ParticleSystem& sim)
{
    // -- HERE: compute kinetic energy

    return 0.0f;
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


// -- HERE add function ... errorAtImpact( ... )
//{
//}


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

int main(int argc, char** argv)
{
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
                    // -- HERE: create csv file with final errors dependent on dt and integration method 
                    //          utilizing the errorAtImpact() function

                    // reset scene
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
