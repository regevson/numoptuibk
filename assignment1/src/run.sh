#!/bin/bash
killall particlesystem &> /dev/null
make -C build
./build/bin/particlesystem &
