/*******************************************************************
 *
 * author: Nikolaus Rauch
 * date: 12.02.2021
 *
 * Simple Window (GLFW) Wrapper Class
 *
 */
#pragma once

#include "base.h"
#include "mouse.h"
#include "keyboard.h"

#include <memory>
#include <functional>
#include <string>

enum class eClientAPI : int
{
    OPENGL          = GLFW_OPENGL_API,
    OPENGL_ES       = GLFW_OPENGL_ES_API
};

enum class eOpenGLProfile : int
{
    ANY             = GLFW_OPENGL_ANY_PROFILE,
    COMPATIBILITY   = GLFW_OPENGL_COMPAT_PROFILE,
    CORE            = GLFW_OPENGL_CORE_PROFILE
};

enum class eContextRobustness : int
{
    NO_ROBUSTNESS           = GLFW_NO_ROBUSTNESS,
    NO_RESET_NOTIFICATION   = GLFW_NO_RESET_NOTIFICATION,
    LOSE_CONTEXT_ON_RESET   = GLFW_LOSE_CONTEXT_ON_RESET
};

enum class eContextReleaseBehaviour : int
{
    ANY     = GLFW_ANY_RELEASE_BEHAVIOR,
    FLUSH   = GLFW_RELEASE_BEHAVIOR_FLUSH,
    NONE    = GLFW_RELEASE_BEHAVIOR_NONE
};


class Window
{
public:
    struct ContextAttributes
    {
        ContextAttributes();
        void apply() const;

        eClientAPI mAPI;
        int mVersionMajor;
        int mVersionMinor;

        bool mOpenGLForwardCompatible;
        bool mOpenGLDebugContext;

        eOpenGLProfile mProfile;
        eContextRobustness mRobustness;
        eContextReleaseBehaviour mReleaseBehaviour;
    };

    struct FrameBufferAttributes
    {
        FrameBufferAttributes();
        void apply() const;

        int mRedBits;
        int mGreenBits;
        int mBlueBits;
        int mAlphaBits;
        int mDepthBits;
        int mStencilBits;
        int mSamples;

        bool mDoubleBuffer;
        int mRefreshRate;
    };

    struct StyleAttributes
    {
        StyleAttributes();
        void apply() const;

        bool mResizable;
        bool mVisible;
        bool mDecorated;
        bool mFocused;
        bool mAutoIconify;
        bool mFloating;
    };


    Window(const std::string& title, int width, int height, const ContextAttributes& context = ContextAttributes(), const StyleAttributes& style = StyleAttributes(), const FrameBufferAttributes& framebuffer = FrameBufferAttributes());
    Window(const Window&) = delete;
    Window(Window&& w);
    Window& operator = (const Window&) = delete;
    Window& operator = (Window&& w);
    virtual ~Window();

    GLFWwindow* handle() const;
    Keyboard& keyboard() const;
    Mouse& mouse() const;

    void close(bool value);
    bool closed() const;
    void title(const std::string& title);
    void position(int x, int y);
    void size(int width, int height);

    void iconify();
    void restore();
    void show();
    void hide();

    void makeCurrent();
    void swapBuffers();

    glm::ivec2 position() const;
    glm::ivec2 size() const;
    glm::ivec2 framebufferSize() const;
    float aspectRatio() const;

    bool focused() const;
    bool iconified() const;
    bool visible() const;

    void vsync(bool enable);

    virtual void onKey(int key, int mod, bool press);
    virtual void onChar(unsigned int c);
    virtual void onMouseButton(int x, int y, int button, int mod, bool press);
    virtual void onMouseScroll(double delta);
    virtual void onResize(int width, int height);

    /* GLFW */
    static void initializedGLFW();
    static void shutdownGLFW();
    static void pollEventsGLFW();

private:
    GLFWwindow* mHandle;

    std::string mTitle;

    std::unique_ptr<Keyboard> mKeyboard;
    std::unique_ptr<Mouse> mMouse;

    static bool sGLFWisInitialized;
};

