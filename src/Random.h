#include <cstdlib>
#include <cmath>
#include <vector>

namespace ARTOS
{

/**
* A simple helper class for generating random numbers of different kind.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/
class Random
{

private:

    static bool seeded; /**< Indicates if the random number generator of C has already been seeded. */


public:

    /**
    * Seeds the random number generator with the current time if it hasn't been seeded before.
    * Subsequent calls to this function will do nothing.
    */
    static void seedOnce();
    
    /**
    * @return Uniformly distributed random integral number between 0 and RAND_MAX (inclusively).
    */
    static int getInt()
    {
        return std::rand();
    };
    
    /**
    * @return Uniformly distributed random integral number between 0 and @p max (inclusively).
    */
    static int getInt(const int max)
    {
        return Random::getInt() % (max + 1);
    };
    
    /**
    * @return Uniformly distributed random integral number between @p min and @p max (inclusively).
    */
    static int getInt(const int min, const int max)
    {
        return Random::getInt(max - min) + min;
    };
    
    /**
    * @return Uniformly distributed random floating point number between 0 and 1 (inclusively).
    */
    static float getFloat()
    {
        return static_cast<float>(std::rand()) / RAND_MAX;
    };
    
    /**
    * @return Uniformly distributed random floating point number between @p min and @p max (inclusively).
    */
    static float getFloat(const float & min, const float & max)
    {
        return Random::getFloat() * (max - min) + min;
    };
    
    /**
    * Draws a floating point number out of a specific range according to a uniform distribution.
    * @param[in] min Minimum number to be returned.
    * @param[in] max Maximum number to be returned.
    * @param[in] precision Distance between each two consecutive numbers in the set.
    * @return Random floating point number out of `{min, min + precision, min + 2*precision, ..., max - precision, max}`.
    */
    static float getFloat(const float & min, const float & max, const float & precision)
    {
        return Random::getInt(static_cast<int>((max - min) / precision)) * precision + min;
    };
    
    /**
    * @return Uniformly distributed random double-precision floating point number between 0 and 1 (inclusively).
    */
    static double getDouble()
    {
        return static_cast<double>(std::rand()) / RAND_MAX;
    };
    
    /**
    * @return Uniformly distributed random double-precision floating point number between @p min and @p max (inclusively).
    */
    static double getDouble(const double & min, const double & max)
    {
        return Random::getDouble() * (max - min) + min;
    };
    
    /**
    * Draws a double-precision floating point number out of a specific range according to a uniform distribution.
    * @param[in] min Minimum number to be returned.
    * @param[in] max Maximum number to be returned.
    * @param[in] precision Distance between each two consecutive numbers in the set.
    * @return Random floating point number out of `{min, min + precision, min + 2*precision, ..., max - precision, max}`.
    */
    static double getDouble(const double & min, const double & max, const double & precision)
    {
        return Random::getInt(static_cast<int>((max - min) / precision)) * precision + min;
    };
    
    /**
    * @return Random boolean value which will be true with a probability of @p chance.
    */
    static double getBool(const double & chance = 0.5)
    {
        return (Random::getDouble() < chance);
    };
    
    /**
    * Generates a normally distributed random number with given mean and standard deviation using Marsaglia's polar method.
    * @param[in] mean The mean (expectation value) of the random variable to be generated.
    * @param[in] sigma The standard deviation of the random variable to be generated.
    * @return Returns a value distributed according \f$\mathcal{N}(\mbox{mean},\,\mbox{sigma}^2)\f$.
    * @note This function is not thread-safe. If you're calling it from multiple concurrent threads, use norm_ts() instead.
    */
    static double norm(double mean = 0.0, double sigma = 1.0)
    {
        static bool hasSpare = false;
        static double spare;
        if (hasSpare)
        {
            hasSpare = false;
            return sigma * spare + mean;
        }
    
        double u, v, q;
        do
        {
            u = Random::getDouble(-1.0, 1.0);
            v = Random::getDouble(-1.0, 1.0);
            q = u * u + v * v;
        }
        while (q == 0.0 || q >= 1);
        q = sqrt(-2.0 * log(q) / q);
        spare = v * q;
        hasSpare = true;
        return sigma * u * q + mean;
    };

    /**
    * Generates a normally distributed random number with given mean and standard deviation using Marsaglia's polar method.
    * This is the thread-safe, but slower version of norm().
    * @param[in] mean The mean (expectation value) of the random variable to be generated.
    * @param[in] sigma The standard deviation of the random variable to be generated.
    * @return Returns a value distributed according \f$\mathcal{N}(\mbox{mean},\,\mbox{sigma}^2)\f$.
    */
    static double norm_ts(double mean = 0.0, double sigma = 1.0)
    {
        double u, q;
        do
        {
            u = Random::getDouble(-1.0, 1.0);
            q = u * u;
            u = Random::getDouble(-1.0, 1.0);
            q += u * u;
        }
        while (q == 0.0 || q >= 1);
        q = sqrt(-2.0 * log(q) / q);
        return sigma * u * q + mean;
    };
    
    /**
    * Selects a random value out of a vector according to a uniform distribution.
    * @param[in] v The vector to choose a value out of.
    * @return Reference to a random value of @p v.
    */
    template<class T>
    static T & choose(std::vector<T> & v)
    {
        return v[Random::getInt(v.size() - 1)];
    };

};

}
