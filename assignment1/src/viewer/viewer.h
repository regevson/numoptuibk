/*******************************************************************
 *
 * author: Nikolaus Rauch (main), Marcel Ritter
 * date: 15.10.2021
 *
 * Simple Viewer (OpenGL2 Context, ImGui, ImPlot)
 */
#pragma once

#include "window.h"

#include <string>
#include <functional>

#include <imgui/imgui.h>
#include <imgui/implot.h>

namespace detail { struct WindowImGui; }

class Viewer
{
    typedef std::function<void (Window& window, Keyboard& keyboard, int key, int mod, bool press)> KeyboardCallback;
    typedef std::function<void (Window& window, Mouse& mouse, int button, int mod, bool press)> MouseCallback;
    typedef std::function<void (Window& window, double dt)> UpdateCallback;
    typedef std::function<void (Window& window, double dt)> DrawCallback;
    typedef std::function<void (Window& window, double dt)> GuiCallback;
    typedef std::function<void (void)> SetupCallback;

public:
    Viewer();
    ~Viewer();

    void run();

    double fps() const;


    /* install callbacks */
    void onInit(const SetupCallback& initCB);
    void onShutdown(const SetupCallback& shutdownCB);

    void onKey(const KeyboardCallback& keyCB);
    void onMouseButton(const MouseCallback& mouseCB);
    void onUpdate(const UpdateCallback& updateCB);
    void onDraw(const DrawCallback& drawCB);
    void onGui(const GuiCallback& guiCB);


    /* render functions */
    template<class InputIt, class UnaryFunction>
    void drawPoints(InputIt first, InputIt last, UnaryFunction f);

    template<class InputIt, class UnaryFunction>
    void drawLines(InputIt first, InputIt last, UnaryFunction f);

    template<class InputIt, class UnaryFunction>
    void drawTriangles(InputIt first, InputIt last, UnaryFunction f);

    template<class InputIt, class UnaryFunction>
    void drawOutline(InputIt first, InputIt last, UnaryFunction f);

    void drawBoundary(const glm::vec2& pos, const glm::vec2& normal, float size, float pseudoShadingSize = -1.0f);

private:
    float renderScale() const;

    //---------------------------------------------//
public:
    /* window settings */
    struct
    {
        std::string title;
        int width;
        int height;
        bool vsync;
        bool mHDPI;
    } mWindow;

    struct
    {
        float pointRadius;
        float lineWidth;
        float scale;

        bool mWireframe;

        glm::vec3 bgColor;
    } mRender;

private:
    struct
    {
        double accumTime;
        double fps;
        double lag;
        unsigned int frames;
    } mFPSCounter;

private:
    MouseCallback mOnMouseButton;
    KeyboardCallback mOnKey;
    UpdateCallback mOnUpdate;
    DrawCallback mOnDraw;
    GuiCallback mOnGui;
    SetupCallback mOnInit;
    SetupCallback mOnShutdown;

    friend struct detail::WindowImGui;
};

#include "viewer.inl"

