#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include "opencv2/opencv.hpp"
#include "opencv/highgui.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core246d.lib")
#pragma comment(lib, "opencv_imgproc246d.lib")
#pragma comment(lib, "opencv_highgui246d.lib")
#pragma comment(lib, "opencv_video246d.lib")
#else
#pragma comment(lib, "opencv_core246.lib")
#pragma comment(lib, "opencv_imgproc246.lib")
#pragma comment(lib, "opencv_highgui246.lib")
#pragma comment(lib, "opencv_video246.lib")

#endif

#include "afx.h"
#include "cmath"
using namespace cv;
using namespace std;

