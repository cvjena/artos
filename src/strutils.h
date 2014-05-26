/**
* @file
* Common string operations.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/

#ifndef STRUTILS_H
#define STRUTILS_H

#include <string>
#include <vector>

/**
* Removes white-space and newline characters from the beginning and the end of a string.
*
* @param[in] str The string to be trimmed.
*
* @return The trimmed string.
*/
std::string trim(std::string str);

/**
* Turns a string into lower-case.
*
* @param[in] str The input string.
*
* @return Lower-case variant of `str`.
*/
std::string strtolower(const std::string & str);

/**
* Turns a string into upper-case.
*
* @param[in] str The input string.
*
* @return Upper-case variant of `str`.
*/
std::string strtoupper(const std::string & str);

/**
* Splits a string up into tokens by given delimiters.
*
* @param[in] str The string to be split.
*
* @param[in] delimiters Each character in this string is a delimiter. Delimiters won't be contained in the tokens.
*
* @param[out] tokens Each token is appended to this vector.
*
* @return Number of tokens.
*/
int splitString(const std::string & str, const char * delimiters, std::vector<std::string> & tokens);

#endif