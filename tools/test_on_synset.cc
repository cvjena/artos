/**
* @file
* This is an example on how to use the `libartos` `C`-API for evaluating the
* performance of a specific model on data from a synset of the Image Repository.  
* This could also be done using the `C++`-API, of course.
*
* See `test_on_dir.cc` for an example on how to achieve the equivalent with data
* from an arbitrary directory using the `C++`-API.
*
* @author Bjoern Barz <bjoern.barz@uni-jena.de>
*/


#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include "libartos.h"
using namespace std;

void printHelp(const char *);
int printError(int, unsigned int = 0);
bool displayProgress(unsigned int, unsigned int);


struct option cmdlineOptions[] = {
    {"help", no_argument, NULL, 'h'},
    {"negatives-from-other-synsets", no_argument, NULL, 'n'},
    {"dump-file", required_argument, NULL, 'f'},
    {0, 0, 0, 0}
};


int main(int argc, char * argv[])
{
    // Parse command line options
    bool optHelp = false;
    unsigned int optNegatives = 0;
    const char * optDumpFile = NULL;
    int option;
    while ((option = getopt_long(argc, argv, "hn:f:", cmdlineOptions, NULL)) != -1)
        switch (option)
        {
            case 'h':
                optHelp = true;
                break;
            case 'n':
                optNegatives = strtoul(optarg, NULL, 10);
                break;
            case 'f':
                optDumpFile = optarg;
                break;
        }
    
    if (optHelp || argc - optind < 3)
        printHelp(argv[0]);
    else
    {
    
        string modelFile = argv[optind];
        string repoDir = argv[optind + 1];
        string synsetId = argv[optind + 2];
        int res;
        
        // Create detector for the model
        unsigned int detector = create_detector();
        if (detector == 0)
        {
            cerr << "Could not create detector instance." << endl;
            return 1;
        }
        res = add_model(detector, "Test", modelFile.c_str(), 0.0);
        if (res != ARTOS_RES_OK)
            return printError(res, detector);
        
        // Extract samples
        res = evaluator_add_samples_from_synset(detector, repoDir.c_str(), synsetId.c_str(), optNegatives);
        if (res != ARTOS_RES_OK)
            return printError(res, detector);
        
        // Evaluate
        res = evaluator_run(detector, 100, &displayProgress);
        if (res != ARTOS_RES_OK)
            return printError(res, detector);
        
        float ap, fm0, maxFm, maxFmTh;
        evaluator_get_ap(detector, &ap);
        evaluator_get_max_fmeasure(detector, &maxFm, &maxFmTh);
        evaluator_get_fmeasure_at(detector, 0.0f, &fm0);
        
        cout << endl << "Average Precision: " << ap << endl << endl;
        cout << "Maximum F-Measure: " << maxFm << " (at threshold " << maxFmTh << ")" << endl;
        cout << "F-Measure at threshold 0: " << fm0 << endl;
        
        // Advanced evaluation: Compute precision and recall at threshold 0 from raw results
        unsigned int result_buf_size = 0;
        evaluator_get_raw_results(detector, NULL, &result_buf_size);
        if (result_buf_size > 0)
        {
            RawTestResult * results = new RawTestResult[result_buf_size];
            evaluator_get_raw_results(detector, results, &result_buf_size);
            
            pair<float, float> maxRecall(0.0, 0.0);
            float recall, zeroRecall = -1.0, zeroPrecision = -1.0;
            for (RawTestResult * testResult = results; testResult < results + result_buf_size; testResult++)
            {
                recall = static_cast<float>(testResult->tp) / testResult->np;
                if (recall > maxRecall.first)
                {
                    maxRecall.first = recall;
                    maxRecall.second = static_cast<float>(testResult->tp) / (testResult->tp + testResult->fp);
                }
                if (zeroRecall < 0.0 && testResult->threshold >= 0)
                {
                    zeroRecall = recall;
                    zeroPrecision = static_cast<float>(testResult->tp) / (testResult->tp + testResult->fp);
                }
            }
            cout << endl;
            cout << "Precision at threshold 0: " << zeroPrecision << endl;
            cout << "Recall at threshold 0: " << zeroRecall << endl;
            cout << "Maximum recall: " << maxRecall.first << " (with a precision of " << maxRecall.second << ")" << endl;
            
            delete[] results;
        }
        
        // Dump test results
        if (optDumpFile != NULL)
        {
            res = evaluator_dump_results(detector, optDumpFile);
            if (res != ARTOS_RES_OK)
                printError(res);
        }
        
        // Clean up
        destroy_detector(detector);
    
    }
    
    return EXIT_SUCCESS;
}


void printHelp(const char * progName)
{
    cout << "Runs the detector with a given model against all samples in a given synset" << endl
         << "to determine the Average Precision of that model on the synset as well as the" << endl
         << "F1-Score at threshold 0." << endl << endl
         << "Usage: " << progName << " [options] model-filename image-repository synset" << endl << endl
         << "ARGUMENTS" << endl << endl
         << "    model-filename         Name of the model file." << endl
         << endl
         << "    image-repository       Path to the image repository." << endl
         << endl
         << "    synset                 ID of the synset to test the model against." << endl
         << endl
         << "OPTIONS" << endl << endl
         << "    --help, -h" << endl
         << endl
         << "            Print this message." << endl
         << endl
         << "    --negatives-from-other-synsets=<num>, -n <num>" << endl
         << endl
         << "            If set, the detector will additionally be run against <num> images" << endl
         << "            from all other synsets, while every detection on one of those" << endl
         << "            images will be considered a false positive." << endl
         << endl
         << "    --dump-file=<filename>, -f <filename>" << endl
         << endl
         << "            If given, Precision, Recall and F-Measure for every" << endl
         << "            possible thresholds will be written to that file." << endl;
}


int printError(int errorCode, unsigned int detector)
{
    if (detector != 0)
        destroy_detector(detector);
    
    if (errorCode != ARTOS_RES_OK)
        switch(errorCode)
        {
            case ARTOS_DETECT_RES_INVALID_MODEL_FILE:
                cerr << "The specified model file is invalid." << endl;
                break;
            case ARTOS_SETTINGS_RES_UNKNOWN_FEATURE_EXTRACTOR:
                cerr << "The feature extractor specified in the model file is not known." << endl;
                break;
            case ARTOS_SETTINGS_RES_UNKNOWN_PARAMETER:
                cerr << "A parameter specified in the model file is not known by the specified feature extractor." << endl;
                break;
            case ARTOS_SETTINGS_RES_INVALID_PARAMETER_VALUE:
                cerr << "A parameter value specified in the model file for the feature extractor is invalid." << endl;
                break;
            case ARTOS_IMGREPO_RES_INVALID_REPOSITORY:
                cerr << "The specified image repository is invalid." << endl;
                break;
            case ARTOS_IMGREPO_RES_SYNSET_NOT_FOUND:
                cerr << "Synset not found." << endl;
                break;
            case ARTOS_IMGREPO_RES_EXTRACTION_FAILED:
                cerr << "Could not extract any sample from the specified synset." << endl;
                break;
            case ARTOS_DETECT_RES_NO_MODELS:
                cerr << "No models have been added to the detector yet." << endl;
                break;
            case ARTOS_DETECT_RES_TOO_MANY_MODELS:
                cerr << "Too many models have been added to the detector for evaluation." << endl;
                break;
            case ARTOS_DETECT_RES_NO_IMAGES:
                cerr << "No samples have been added yet." << endl;
                break;
            default:
                cerr << "Unknown error." << endl;
                break;
        }
    
    return errorCode;
}


bool displayProgress(unsigned int current, unsigned int total)
{
    static int lastProgress = -1;
    if (lastProgress < 0)
    {
        cout << "....................";
        lastProgress = 0;
    }
    else
    {
        int progress = (current * 20) / total;
        if (progress > 20)
            progress = 20;
        if (progress > lastProgress)
        {
            int i;
            for (i = 0; i < 20; i++)
                cout << static_cast<char>(8);
            for (i = 0; i < progress; i++)
                cout << "|";
            for (i = progress; i < 20; i++)
                cout << ".";
            cout << flush;
            if (current >= total)
                cout << endl;
            lastProgress = progress;
        }
    }
    return true;
}
