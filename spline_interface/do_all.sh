# conda install libgcc
g++ -Wall -fPIC -O2 -c myLibSpline.cpp -std=c++11 -fpermissive
g++ -Wall -fPIC -O2 -c bspline.cpp
g++ -shared -o myLibSpline.so myLibSpline.o bspline.o
python generate_spline.py
