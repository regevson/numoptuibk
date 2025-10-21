/*******************************************************************
 *
 * author: Nikolaus Rauch
 * date: 12.02.2021
 *
 */
#pragma once

#include "base.h"

#include <array>

class Window;

class Keyboard
{
public:
    Keyboard(const Keyboard&) = delete;
    Keyboard& operator = (const Keyboard&) = delete;

    void stickyKeys(bool enabled);
    bool stickyKeys() const;

    Window &sourceWindow() const;
    bool operator[](std::size_t button) const;
    bool& operator[](std::size_t button);

private:
    Window& mSourceWindow;
    std::array<bool, GLFW_KEY_LAST> mKeyState;

    Keyboard(Window& sourceWindow);
    friend class Window;
};

