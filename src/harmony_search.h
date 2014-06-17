#include <vector>

namespace ARTOS
{

typedef float (*hs_objective_function)(const std::vector<float> &, const std::vector<int> &, void *);

/**
* Performs the discrete *Harmony Search* algorithm to approximate the optimum of a multivariate objective function.
*
* @see Geem, Kim, Loganathan. A New Heuristic Optimization Algorithm: Harmony Search.
* In SIMULATION February 2001 76, pp. 60-68.
*
* @see http://en.wikipedia.org/wiki/Harmony_Search
*
* @param[in] ofunc A callback to the function to be optimized. It will be called with a vector of parameters as first
* argument and has to return a floating point value. A vector with the indices of the current parameters in the
* respective `params` vector is passed as second argument.
*
* @param[in] params Vector of vectors with possible values for each parameter of the objective function.  
* For example, if the function to be optimized had two parameters, this would be a vector of size 2; the first element
* would be a vector with all possible values of the first parameter of the function and the second element would be a
* vector with all possible values of the second parameter.
*
* @param[in] ofuncData Additional data to pass to the objective function `ofunc` as third argument.
*
* @param[in] maximize If this is set to true, the objective function will be maximized, otherwise minimized.
*
* @param[out] bestFitness If set to a non-NULL value, the float variable pointed to will receive the value
* of the returned parameter combination according to the objective function.
*
* @param[in] hms *Harmony Memory Size*: The number of initially generated and stored parameter vectors.
*
* @param[in] iterations Harmony Search will terminate after exactly this number of rounds.
*
* @param[in] hmcr *Harmony Memory Considering Rate*: Probability to choose a new parameter from the *Harmony Memory*
* instead of selecting a random one.
*
* @param[in] par *Pitch Adjusting Rate*: Probability to shift a parameter selected from *Harmony Memory* to one of it's
* neighbours.
*/
std::vector<float> harmony_search(hs_objective_function ofunc, const std::vector< std::vector<float> > & params, void * ofuncData = 0,
                                  const bool maximize = false, float * bestFitness = 0,
                                  const unsigned int hms = 30, const unsigned int iterations = 100000,
                                  const double hmcr = 0.9, const double par = 0.3);

}