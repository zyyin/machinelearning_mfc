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

#include "StdAfx.h"
#include "Utils.h"

Utils::Utils(void)
{
}

Utils::~Utils(void)
{
}

void Utils::BrowseFolder(const char* folder, vector<string>& fileList, vector<string>& nameList)
{
	CFileFind finder;
	CString strPath;
	CString str = folder;
	str += "\\*.*";
	BOOL bWorking = finder.FindFile(str);
	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		if(!finder.IsDirectory() && !finder.IsDots()) {

			strPath=finder.GetFilePath();
			fileList.push_back(LPCTSTR(strPath));
			nameList.push_back(LPCTSTR(finder.GetFileName()));
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