#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include "opencv2/opencv.hpp"
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv/highgui.h"
#if _DEBUG 
#pragma comment(lib, "opencv_world340d.lib")
#else
#pragma comment(lib, "opencv_world340.lib")

#endif
#include "afx.h"
#include "cmath"
using namespace cv;
using namespace std;
