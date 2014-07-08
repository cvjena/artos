#include "harmony_search.h"
#include <cassert>
#include "Random.h"
using namespace std;

vector<float> ARTOS::harmony_search(hs_objective_function ofunc, const vector< vector<float> > & params, void * ofuncData,
                                    const bool maximize, float * bestFitness,
                                    const unsigned int hms, const unsigned int iterations,
                                    const double hmcr, const double par)
{
    assert(params.size() > 0);
    assert(hmcr > 0 && hmcr < 1);
    assert(par > 0 && par < 1);

    vector< vector<int> > hm;
    hm.reserve(hms);
    vector<float> fitness;
    fitness.reserve(hms);
    vector<float> ofuncParams(params.size());
    float flip = (maximize) ? -1.0f : 1.0f;
    double halfPar = par / 2.0;
    unsigned int i, j, round, iBest = 0, iWorst = 0;
    
    // Initialize hm (Harmony Memory) at random
    Random::seedOnce();
    for (i = 0; i < hms; i++)
    {
        hm.push_back(vector<int>());
        hm.back().reserve(params.size());
        for (j = 0; j < params.size(); j++)
        {
            hm.back().push_back(Random::getInt(params[j].size() - 1));
            ofuncParams[j] = params[j][hm.back().back()];
        }
        fitness.push_back(ofunc(ofuncParams, hm.back(), ofuncData));
        if (flip * fitness[i] > flip * fitness[iWorst])
            iWorst = i;
        else if (flip * fitness[i] < flip * fitness[iBest])
            iBest = i;
    }
    
    // Iteratively generate new possible solutions
    vector<int> newHarmony(params.size());
    float newFitness;
    double parChoice;
    for (round = 0; round < iterations; round++)
    {
        for (i = 0; i < params.size(); i++)
        {
            if (Random::getBool(hmcr))
            {
                // Pick an existing value from Harmony Memory
                newHarmony[i] = Random::choose(hm)[i];
                parChoice = Random::getDouble();
                if (parChoice < par)
                {
                    // Modify picked value slightly
                    if (parChoice < halfPar)
                    {
                        newHarmony[i]++;
                        if (newHarmony[i] >= params[i].size())
                            newHarmony[i] = params[i].size() - 1;
                    }
                    else
                    {
                        newHarmony[i]--;
                        if (newHarmony[i] < 0)
                            newHarmony[i] = 0;
                    }
                }
            }
            else
            {
                // Pick random value
                newHarmony[i] = Random::getInt(params[i].size() - 1);
            }
            ofuncParams[i] = params[i][newHarmony[i]];
        }
        newFitness = ofunc(ofuncParams, newHarmony, ofuncData);
        if (flip * newFitness < flip * fitness[iWorst])
        {
            if (flip * newFitness < flip * fitness[iBest])
                iBest = iWorst;
            hm[iWorst] = newHarmony;
            fitness[iWorst] = newFitness;
            for (i = 0; i < fitness.size(); i++)
                if (flip * fitness[i] > flip * fitness[iWorst])
                    iWorst = i;
        }
    }
    
    // Return best solution in Harmony Memory
    for (i = 0; i < params.size(); i++)
        ofuncParams[i] = params[i][hm[iBest][i]];
    if (bestFitness != 0)
        *bestFitness = fitness[iBest];
    return ofuncParams;
}


vector<float> ARTOS::repeated_harmony_search(hs_objective_function ofunc, const vector< vector<float> > & params, void * ofuncData,
                                    const bool maximize, float * bestFitness,
                                    const unsigned int hms, const unsigned int iterations,
                                    const double hmcr, const double par)
{
#ifdef _OPENMP
    Random::seedOnce();
    vector<float> bestSolution;
    float bestValue;
    #pragma omp parallel num_threads(16)
    {
        float fitness;
        vector<float> solution = harmony_search(ofunc, params, ofuncData, maximize, &fitness, hms, iterations, hmcr, par);
        #pragma omp critical
        {
            if (bestSolution.empty() || (!maximize && fitness < bestValue) || (maximize && fitness > bestValue))
            {
                bestSolution = solution;
                bestValue = fitness;
            }
        }
    }
    if (bestFitness != 0)
        *bestFitness = bestValue;
    return bestSolution;
#else
    return harmony_search(ofunc, params, ofuncData, maximize, bestFitness, hms, iterations, hmcr, par);
#endif
}
