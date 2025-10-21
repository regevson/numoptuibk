/*******************************************************************
 *
 * author: Nikolaus Rauch
 * date: 12.02.2021
 *
 */
#pragma once

#include "base.h"

#include <memory>
#include <array>

class Window;

class Mouse
{
public:
    enum eButton: int
    {
        LEFT = GLFW_MOUSE_BUTTON_1,
        RIGHT,
        MIDDLE,
        OTHER_1,
        OTHER_2,
        OTHER_3,
        OTHER_4,
        OTHER_5
    };

    enum class eCursorState : int
    {
        VISIBLE     = GLFW_CURSOR_NORMAL,
        HIDDEN      = GLFW_CURSOR_HIDDEN,
        DISABLED    = GLFW_CURSOR_DISABLED
    };

    Mouse(const Mouse&) = delete;
    Mouse& operator = (const Mouse&) = delete;

    void position(double x, double y);
    const glm::dvec2& position();

    void cursorState(eCursorState state);
    eCursorState cursorState() const;

    void scroll(float scroll);
    float scroll() const;

    bool operator[] (std::size_t button) const;
    bool& operator[] (std::size_t button);

    Window& window() const;

private:
    glm::dvec2 mPosition;
    std::array<bool, GLFW_MOUSE_BUTTON_LAST>  mButtonState;
    float mScroll;

    Window& mSourceWindow;
    std::unique_ptr<GLFWcursor, void(*)(GLFWcursor*)> mMouseCursor;


    Mouse(Window &sourceWindow);
    friend class Window;
};
