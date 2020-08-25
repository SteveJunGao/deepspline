#!/bin/bash
clear
#rm -rd build
mkdir build
cd build
cmake ..
if make -j10; then
    echo "Success"
    rm ../compile_commands.json
    cp compile_commands.json ../
    clear
    ./spline
else
    echo "Fail"
fi
