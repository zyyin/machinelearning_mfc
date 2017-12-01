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


#include "xFile.h"

xFile::xFile()
{
    mFilename = NULL;
    pFile = NULL;
    pString = NULL;
}
xFile::~xFile()
{
    Close();
}


xFile::xFile(const char* szFilename, const char* mode)
{
    mFilename = NULL;
    pFile == NULL;
    pString = new char[LINE_LEN + 1];
    mFilename = new char[strlen(szFilename) + 1];
    strcpy(mFilename, szFilename);
    pFile = fopen(mFilename, mode);
}

bool xFile::Open(const char* szFilename, const char* mode)
{
    Close();
    pString = new char[LINE_LEN + 1];
    mFilename = new char[strlen(szFilename) + 1];
    strcpy(mFilename, szFilename);
    pFile = fopen(mFilename, mode);
    return pFile == NULL ? 0 : 1;
}
void xFile::Close()
{

    if(mFilename)
    {
	delete mFilename;
	mFilename = NULL;
    }
    if(pFile)
    {
        fclose(pFile);
        pFile = NULL;
    }
    if(pString)
    {
	delete pString;
	pString = NULL;
    }
}

ULONG xFile::Read(void* pBuffer, int size)
{
    if(pBuffer == NULL || size == 0)
        return 0;
    if(pFile == NULL)
        return 0;
    return fread(pBuffer, 1, size, pFile);
}
bool xFile::ReadString(char* szBuffer, int size)
{
    if(szBuffer == NULL || size == 0 || pFile == NULL)
        return false;
    int len = size > LINE_LEN ? LINE_LEN : size;
    if(!fgets(szBuffer, len, pFile))
	return false;
    if('\n' == szBuffer[strlen(szBuffer)-1])
	{
	    szBuffer[strlen(szBuffer)-1] = '\0';
	}
    return true;
}
void xFile::Write(const void* pBuffer, int nCount)
{
    if(pBuffer == NULL || nCount == 0 || pFile == NULL)
        return;
    fwrite(pBuffer, 1, nCount, pFile);
}

ULONG xFile::SeekToEnd()
{
    if(pFile == NULL)
        return 0;
    ULONG curpos = ftell(pFile);
    fseek(pFile, 0L, SEEK_END);
    return ftell(pFile) - curpos;
}

void xFile::SeekToBegin()
{
    if(pFile == NULL)
        return;
    fseek(pFile, 0L, SEEK_SET);
}
ULONG xFile::Seek(ULONG offset, int nFrom)
{
    if(pFile == NULL)
        return 0;
    return fseek(pFile, offset, nFrom);
}
ULONG xFile::GetFileSize()
{
    SeekToBegin();
    return SeekToEnd();
}

ULONG xFile::GetPosition() const 
{
    if(pFile == NULL)
        return 0;
    return ftell(pFile);
}
char* xFile::GetFileName() const
{
    return mFilename;
}

char* xFile::GetFileFmt() const
{
    if(mFilename == NULL)
        return NULL;
    char* ptr = strrchr(mFilename, '.');
    return ptr + 1;
}
void xFile::Flush()
{
    if(pFile == NULL)
        return;
    fflush(pFile);
}
char* xFile::ReadString()
{
	if(!ReadString(pString, LINE_LEN))
		return NULL;
	return pString;	
}








/********************************** END **********************************************/

