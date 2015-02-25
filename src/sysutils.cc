#include "sysutils.h"
#include <stdlib.h>
#include <cstdarg>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <dirent.h>
#include <sys/stat.h>
#endif

using namespace std;

string get_cwd()
{
#ifdef _WIN32
    DWORD bufsize = GetCurrentDirectory(0, NULL);
    char * wd = static_cast<char*>(malloc(sizeof(char) * bufsize));
    GetCurrentDirectory(bufsize, wd);
#else
    char * wd = getcwd(NULL, 0); // Try to let getcwd() allocate the buffer
    if (wd == NULL)
    {
        // Allocate a large-sized buffer manually
        size_t bufsize = sizeof(char) * (PATH_MAX + 1);
        size_t max_bufsize = bufsize * 4;
        while (wd == NULL && bufsize <= max_bufsize)
        {
            wd = static_cast<char*>(malloc(bufsize));
            if (getcwd(wd, bufsize) == NULL)
            {
                free(wd);
                wd = NULL;
                bufsize *= 2;
            }
        }
    }
#endif
    string wd_str(wd);
    free(wd);
    return wd_str;
}

bool change_cwd(const string & wd)
{
#ifdef _WIN32
    return (SetCurrentDirectory(wd.c_str()) != 0);
#else
    return (chdir(wd.c_str()) == 0);
#endif
}

string real_path(const string & path)
{
#ifdef _WIN32
    DWORD bufsize = GetFullPathName(path.c_str(), 0, NULL, NULL);
    char rp[bufsize];
    GetFullPathName(path.c_str(), bufsize, rp, NULL);
    string rp_str(rp);
    return rp_str;
#else
    char rp[PATH_MAX + 1];
    char * result = realpath(path.c_str(), rp);
    string rp_str((result != NULL) ? rp : path);
    return rp_str;
#endif
}

string extract_dirname(const string & path)
{
#ifdef _WIN32
    size_t last_slash = path.find_last_of("/\\");
    return (last_slash == string::npos) ? "" : path.substr(0, last_slash);
#else
    char cpath[path.length() + 1];
    path.copy(cpath, string::npos);
    cpath[path.length()] = '\0';
    char * cdirname = dirname(cpath);
    string dirname_str(cdirname);
    return dirname_str;
#endif
}

string extract_basename(const string & path)
{
#ifdef _WIN32
    size_t last_slash = path.find_last_of("/\\");
    return (last_slash == string::npos) ? path : path.substr(last_slash + 1);
#else
    char cpath[path.length() + 1];
    path.copy(cpath, string::npos);
    cpath[path.length()] = '\0';
    char * cbasename = basename(cpath);
    string basename_str(cbasename);
    return basename_str;
#endif
}

string replace_file_extension(const string & path, const string & newExtension)
{
    size_t slash_pos = path.find_last_of("/\\");
    size_t dot_pos = path.rfind('.');
    if (dot_pos == string::npos || (slash_pos != string::npos && dot_pos < slash_pos))
        return path + newExtension;
    else
        return path.substr(0, dot_pos) + newExtension;
}

string strip_file_extension(const string & path)
{
    return replace_file_extension(path, "");
}

string join_path(int numComponents, ...)
{
#ifdef _WIN32
    char delim = '\\';
#else
    char delim = '/';
#endif
    string result = "";
    va_list args;
    va_start(args, numComponents);
    for (int i = 0; i < numComponents; i++)
    {
        string comp = va_arg(args, char*);
        // strip front slashes
        if (i > 0 && (comp[0] == '/' || comp[0] == '\\'))
            comp = comp.substr(1);
        // strip trailing slashes
        if (i < numComponents - 1 && (i > 0 || comp != "/") && (comp[comp.length()-1] == '/' || comp[comp.length()-1] == '\\'))
            comp = comp.substr(0, comp.length() - 1);
        // insert delimiter and component
        if (i > 0)
            result += delim;
        result += comp;
    }
    va_end(args);
    return result;
}

bool is_file(const string & path)
{
#ifdef _WIN32
    DWORD ftyp = GetFileAttributesA(path.c_str());
    return (ftyp != INVALID_FILE_ATTRIBUTES && !(ftyp & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st_buf;
    int status = stat(path.c_str(), &st_buf);
    return (status == 0 && S_ISREG(st_buf.st_mode));
#endif
}

bool is_dir(const string & path)
{
#ifdef _WIN32
    DWORD ftyp = GetFileAttributesA(path.c_str());
    return (ftyp != INVALID_FILE_ATTRIBUTES && (ftyp & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st_buf;
    int status = stat(path.c_str(), &st_buf);
    return (status == 0 && S_ISDIR(st_buf.st_mode));
#endif
}

void scandir(const string & dir, vector<string> & files, const FileType ft, const string & extensionFilter)
{
#ifdef _WIN32
    WIN32_FIND_DATA findData;
    HANDLE hFind;
    string searchPattern = join_path(2, dir.c_str(), "*");
    if (!extensionFilter.empty())
        searchPattern += "." + extensionFilter;
    hFind = FindFirstFile(searchPattern.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            string name(findData.cFileName);
            if (((ft & ftFile) && !(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                    || ((ft & ftDirectory) && (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && name != "." && name != ".."))
                files.push_back(findData.cFileName);
        }
        while (FindNextFile(hFind, &findData));
        FindClose(hFind);
    }
#else
    DIR * dirp = opendir(dir.c_str());
    if (dirp != NULL)
    {
        struct dirent * entry;
        size_t extLen = extensionFilter.length() + 1;
        while ((entry = readdir(dirp)) != NULL)
        {
            string name(entry->d_name);
#ifdef _DIRENT_HAVE_D_TYPE
            if (ft == ftAny || ((ft & ftFile) && entry->d_type == DT_REG) || ((ft & ftDirectory) && entry->d_type == DT_DIR))
#else
            string path = join_path(2, dir.c_str(), name.c_str());
            if (ft == ftAny || ((ft & ftFile) && is_file(path)) || ((ft & ftDirectory) && is_dir(path)))
#endif
                if (name != "." && name != ".."
                        && (extLen <= 1 || (name.length() > extLen && name.substr(name.length() - extLen) == "." + extensionFilter)))
                    files.push_back(name);
        }
        closedir(dirp);
    }
#endif
}
