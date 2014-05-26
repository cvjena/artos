#include "strutils.h"
#include <cctype>
using namespace std;


string trim(string str)
{
    size_t pos = str.find_first_not_of(" \r\n\t");
    if (pos > 0 && pos != string::npos)
        str = str.substr(pos);
    pos = str.find_last_not_of(" \r\n\t");
    if (pos < str.length() - 1 && pos != string::npos)
        str = str.substr(0, pos + 1);
    return str;
}

string strtolower(const string & str)
{
    string lstr(str.length(), '\0');
    string::const_iterator itIn;
    string::iterator itOut;
    for (itIn = str.begin(), itOut = lstr.begin(); itIn != str.end(); itIn++, itOut++)
        *itOut = tolower(*itIn);
    return lstr;
}

string strtoupper(const string & str)
{
    string lstr(str.length(), '\0');
    string::const_iterator itIn;
    string::iterator itOut;
    for (itIn = str.begin(), itOut = lstr.begin(); itIn != str.end(); itIn++, itOut++)
        *itOut = toupper(*itIn);
    return lstr;
}

int splitString(const string & str, const char * delimiters, vector<string> & tokens)
{
    int numTokens = 0;
    size_t pos = 0, startPos = 0;
    do
    {
        startPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, startPos);
        if (startPos < pos)
        {
            tokens.push_back(str.substr(startPos, pos - startPos));
            numTokens++;
        }
    }
    while (pos != string::npos);
    return numTokens;
}