#pragma once
#include <string>
#include <vector>
using namespace std;

class Utils
{
public:
	Utils(void);
	~Utils(void);
	void BrowseFolder(const char* folder, vector<string>& fileList, vector<string>& nameList);

	vector<string> split(const string &str,const string &pattern);
};
