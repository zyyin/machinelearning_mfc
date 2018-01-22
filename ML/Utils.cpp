/*
* Copyright (C) 2017 China.ShangHai, zhiyeyin@gmail.com
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
* http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "Utils.h"
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
Utils::Utils(void)
{
}

Utils::~Utils(void)
{
}

void Utils::BrowseFolder(const char* folder, vector<string>& fileList, vector<string>& nameList)
{
  struct dirent *direntp;
  DIR *dirp = opendir("/");
  if (dirp != NULL) { 
	while ((direntp = readdir(dirp)) != NULL) {
	 fileList.push_back(direntp->d_name);
		   }
	   }
}

vector<string> Utils::split(const string &str,const string &pattern)
{
	//const char* convert to char*
	char strc[1024];
	strcpy(strc, str.c_str());
	vector<string> resultVec;
	char* tmpStr = strtok(strc, pattern.c_str());
	while (tmpStr != NULL)
	{
		resultVec.push_back(string(tmpStr));
		tmpStr = strtok(NULL, pattern.c_str());
	}

	return resultVec;
};
