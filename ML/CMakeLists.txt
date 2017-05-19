cmake_minimum_required(VERSION 2.8)
project( ML )
find_package( OpenCV REQUIRED )
add_executable( ML 
	ML.cpp
	trees.cpp
	bayes.cpp
	)
target_link_libraries( ML ${OpenCV_LIBS} )
