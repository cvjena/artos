/**
* @file
* Cross-platform (file-)system utilities.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/

#ifndef SYSUTILS_H
#define SYSUTILS_H

#include <string>

/**
* @return The current working directory.
*/
std::string get_cwd();

/**
* Changes the current working directory.
*
* @param[in] wd Path of the new working directory.
*
* @return True if the working directory has been changed successfully, otherwise false.
*/
bool change_cwd(const std::string & wd);


/**
* Determines the real absolute path of a file or directory by resolving any links and '.' or '..' references.
*
* @param[in] path Relative path.
*
* @return The full real path pointing to the same location as the input argument.
*/
std::string real_path(const std::string & path);

/**
* Extracts the directory part of a path, e. g. '/home/foo' from '/home/foo/bar.txt'.
* 
* @param[in] path A path on the filesystem.
*
* @return The part of `path` before the last '/' or '\' (exclusively).
*/
std::string extract_dirname(const std::string & path);

/**
* Extracts the filename part of a path, e. g. 'bar.txt' from '/home/foo/bar.txt'.
*
* @param[in] path A path on the filesystem.
*
* @return The part of `path` after the last '/' or '\' (exclusively).
*/
std::string extract_basename(const std::string & path);

/**
* Replaces the file extension in a given path to a file.
* If there is no file extension, it will be appended.
*
* @param[in] path A path on the file system.
*
* @param[in] newExtension The new file extension (inclusively the ".").
*
* @return `path` with the file extension replaced by `newExtension`.
*/
std::string replace_file_extension(const std::string & path, const std::string & newExtension);

/**
* Removes the file extension from a given path.
*
* @param[in] path A path on the filesystem.
*
* @return The part of `path` before the last dot in the filename part (exclusively).
*/
std::string strip_file_extension(const std::string & path);

/**
* Concatenates multiple path components to a single path by inserting an appropriate
* file system delimiter (either '/' or '\') between them.
*
* The component arguments are given as null-terminated C-strings.
*
* Example: `join_path(4, '/', 'home', 'foo', 'bar.txt')` will return '/home/foo/bar.txt'.
*
* @param[in] numComponents Number of successive arguments.
*
* @return The concatenated path.
*/
std::string join_path(int numComponents, ...);


/**
* Checks if a file exists and is a regular file.
*
* @param[in] path A path on the filesystem.
*
* @return True if `path` points to an existing regular file, otherwise false.
*/
bool is_file(const std::string & path);

/**
* Checks if a directory exists.
*
* @param[in] path A path on the filesystem.
*
* @return True if `path` points to an existing directory, otherwise false.
*/
bool is_dir(const std::string & path);

#endif

