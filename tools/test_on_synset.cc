#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>
#include <getopt.h>
#include "ModelEvaluator.h"
#include "ImageRepository.h"
#include "strutils.h"
using namespace ARTOS;
using namespace std;

void printHelp(const char *);
bool displayProgress(unsigned int, unsigned int, void*);


struct option cmdlineOptions[] = {
    {"help", no_argument, NULL, 'h'},
    {"negatives-from-other-synsets", no_argument, NULL, 'n'},
    {"dump-file", required_argument, NULL, 'f'},
    {0, 0, 0, 0}
};


int main(int argc, char * argv[])
{
    // Parse command line options
    bool optHelp = false, optNegatives = false;
    const char * optDumpFile = NULL;
    int option;
    while ((option = getopt_long(argc, argv, "hnf:", cmdlineOptions, NULL)) != -1)
        switch (option)
        {
            case 'h':
                optHelp = true;
                break;
            case 'n':
                optNegatives = true;
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
        
        ModelEvaluator eval(modelFile);
        if (eval.getNumModels() == 0)
        {
            cerr << "Invalid model file." << endl;
            return 1;
        }
        
        // Extract positive samples
        ImageRepository repo(repoDir);
        Synset synset = repo.getSynset(synsetId);
        if (synset.id.empty())
        {
            cerr << "Synset or image repository not found." << endl;
            return 2;
        }
        vector<Sample*> samples;
        unsigned int numObjects = 0;
        for (SynsetImageIterator imgIt = synset.getImageIterator(false); imgIt.ready(); ++imgIt)
        {
            SynsetImage simg = *imgIt;
            JPEGImage & img = simg.getImage();
            if (!img.empty())
            {
                Sample * s = new Sample();
                s->m_simg = simg;
                if (simg.loadBoundingBoxes())
                    s->m_bboxes = simg.bboxes;
                else
                    s->m_bboxes.assign(1, ARTOS::Rectangle(0, 0, img.width(), img.height()));
                s->modelAssoc.assign(s->bboxes().size(), 0);
                s->data = NULL;
                samples.push_back(s);
                numObjects += s->bboxes().size();
            }
        }
        
        // Extract negative samples
        vector<JPEGImage> negativeSamples;
        if (optNegatives)
            for (SynsetIterator synsetIt = repo.getSynsetIterator(); synsetIt.ready(); ++synsetIt)
            {
                Synset negSynset = *synsetIt;
                if (negSynset.id != synset.id)
                    for (SynsetImageIterator imgIt = negSynset.getImageIterator(); imgIt.ready(); ++imgIt)
                    {
                        SynsetImage simg = *imgIt;
                        JPEGImage & img = simg.getImage();
                        if (!img.empty())
                            negativeSamples.push_back(img);
                    }
            }
        
        // Evaluate
        cout << "Testing model against " << (samples.size() + negativeSamples.size()) << " images with " << numObjects << " objects." << endl;
        int lastProgress = -1;
        eval.testModels(samples, 0, &negativeSamples, 1000, &displayProgress, &lastProgress);
        cout << endl << "Average Precision: " << eval.computeAveragePrecision() << endl << endl;
        vector< pair<float, float> > fMeasures = eval.calculateFMeasures();
        const pair<float, float> * maxFMeasure = NULL, * zeroFMeasure = NULL;
        for (vector< pair<float, float> >::const_iterator fit = fMeasures.begin(); fit != fMeasures.end(); fit++)
        {
            if (zeroFMeasure == NULL && fit->first >= 0)
                zeroFMeasure = &(*fit);
            if (maxFMeasure == NULL || fit->second > maxFMeasure->second)
                maxFMeasure = &(*fit);
        }
        cout << "Maximum F-Measure: " << maxFMeasure->second << " (at threshold " << maxFMeasure->first << ")" << endl;
        if (zeroFMeasure != NULL)
            cout << "F-Measure at threshold 0: " << zeroFMeasure->second << endl;
        pair<float, float> maxRecall(0.0, 0.0);
        float recall, zeroRecall = -1.0, zeroPrecision = -1.0;
        for (vector<ModelEvaluator::TestResult>::const_iterator rit = eval.getResults().begin(); rit != eval.getResults().end(); rit++)
        {
            recall = static_cast<float>(rit->tp) / rit->np;
            if (recall > maxRecall.first)
            {
                maxRecall.first = recall;
                maxRecall.second = static_cast<float>(rit->tp) / (rit->tp + rit->fp);
            }
            if (zeroRecall < 0.0 && rit->threshold >= 0)
            {
                zeroRecall = recall;
                zeroPrecision = static_cast<float>(rit->tp) / (rit->tp + rit->fp);
            }
        }
        cout << "Precision at threshold 0: " << zeroPrecision << endl;
        cout << "Recall at threshold 0: " << zeroRecall << endl;
        cout << "Maximum recall: " << maxRecall.first << " (with a precision of " << maxRecall.second << ")" << endl;
        
        // Dump test results
        if (optDumpFile != NULL)
            if (!eval.dumpTestResults(optDumpFile, -1, true, ModelEvaluator::PRECISION | ModelEvaluator::RECALL | ModelEvaluator::FMEASURE))
                cerr << "Could not open file for writing: " << optDumpFile << endl;
        
        // Clean up
        for (vector<Sample*>::iterator sample = samples.begin(); sample != samples.end(); sample++)
            delete *sample;
    
    }
    
    return EXIT_SUCCESS;
}


void printHelp(const char * progName)
{
    cout << "Runs the detector with a given HOG model against all samples in a given synset" << endl
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
         << "    --negatives-from-other-synsets, -n" << endl
         << endl
         << "            If set, the detector will be run against all images from all other" << endl
         << "            synsets, while every detection on one of those images will be" << endl
         << "            considered a false positive." << endl
         << endl
         << "    --dump-file=<filename>, -f <filename>" << endl
         << endl
         << "            If given, Precision, Recall and F-Measure for every" << endl
         << "            possible thresholds will be written to that file." << endl;
}


bool displayProgress(unsigned int current, unsigned int total, void * data)
{
    int * lastProgress = reinterpret_cast<int*>(data);
    if (*lastProgress < 0)
    {
        cout << "....................";
        *lastProgress = 0;
    }
    else
    {
        int progress = (current * 20) / total;
        if (progress > 20)
            progress = 20;
        if (progress > *lastProgress)
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
            *lastProgress = progress;
        }
    }
    return true;
}
