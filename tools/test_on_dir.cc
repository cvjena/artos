#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <limits>
#include "ModelEvaluator.h"
#include "strutils.h"
#include "sysutils.h"
#include "Scene.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

using namespace ARTOS;
using namespace std;


void listJPEGFiles(const char*, vector<string>&);
bool displayProgress(unsigned int, unsigned int, void*);


int main(int argc, char * argv[])
{
    if (argc < 3)
    {
        cout << "Runs the detector with a given HOG model against all samples in a given directory" << endl
             << "to determine the Average Precision of that model on that data as well as the" << endl
             << "F1-Score at threshold 0." << endl
             << "The directory has to contain an XML annotations file for each image with the same name." << endl << endl
             << "Usage: " << argv[0] << " <model-filename> <data-directory> <dump-filename>?" << endl << endl
             << "ARGUMENTS" << endl << endl
             << "    model-filename         Name of the model file." << endl
             << endl
             << "    data-directory         Path to the directory with images and annotation files." << endl
             << endl
             << "    dump-filename          If given, Precision, Recall and F-Measure for every" << endl
             << "                           possible thresholds will be written to that file." << endl;
        return 0;
    }
    
    if (!is_dir(argv[2]))
    {
        cerr << "Directory not found: " << argv[2] << endl;
        return 2;
    }
    
    ModelEvaluator eval(argv[1]);
    if (eval.getNumModels() == 0)
    {
        cerr << "Invalid model file." << endl;
        return 1;
    }
    
    // Extract samples
    vector<Sample*> samples;
    unsigned int numObjects = 0;
    {
        vector<string> files;
        listJPEGFiles(argv[2], files);
        string imgName;
        double scale;
        ARTOS::Rectangle bbox;
        for (vector<string>::const_iterator filename = files.begin(); filename != files.end(); filename++)
        {
            imgName = strip_file_extension(*filename);
            if (is_file(join_path(2, argv[2], (imgName + ".xml").c_str())))
            {
                JPEGImage img(join_path(2, argv[2], filename->c_str()));
                if (!img.empty())
                {
                    Scene scene(join_path(2, argv[2], (imgName + ".xml").c_str()));
                    if (scene.objects().size() > 0)
                    {
                        Sample * sample = new Sample();
                        sample->m_img = img;
                        scale = static_cast<double>(scene.width()) / img.width();
                        for (vector<Object>::const_iterator object = scene.objects().begin(); object != scene.objects().end(); object++)
                            if (!object->difficult())
                            {
                                bbox = object->bndbox();
                                bbox.setX(round(bbox.x() * scale));
                                bbox.setY(round(bbox.y() * scale));
                                bbox.setWidth(round(bbox.width() * scale));
                                bbox.setHeight(round(bbox.height() * scale));
                                if (bbox.width() > 0 && bbox.height() > 0)
                                {
                                    sample->m_bboxes.push_back(bbox);
                                    sample->modelAssoc.push_back(0);
                                    numObjects++;
                                }
                            }
                        samples.push_back(sample);
                    }
                    else
                        cerr << "Could not parse annotations for " << imgName << endl;
                }
                else
                    cerr << "Could not open " << *filename << endl;
            }
            else
                cerr << "No XML file found for " << imgName << endl;
        }
    }
    if (samples.size() == 0)
    {
        cerr << "No images found." << endl;
        return 3;
    }
    cout << "Testing model against " << samples.size() << " images with " << numObjects << " objects." << endl;
    
    // Evaluate
    int lastProgress = -1;
    eval.testModels(samples, 0, NULL, 1000, &displayProgress, &lastProgress);
    cout << "Average Precision: " << eval.computeAveragePrecision() << endl;
    vector< pair<float, float> > fMeasures = eval.calculateFMeasures();
    cout << "F-Measure: ";
    for (vector< pair<float, float> >::const_iterator fit = fMeasures.begin(); fit != fMeasures.end(); fit++)
        if (fit->first >= 0)
        {
            cout << fit->second;
            break;
        }
    cout << endl;
    if (argc > 3)
        eval.dumpTestResults(argv[3], -1, true, ModelEvaluator::PRECISION | ModelEvaluator::RECALL | ModelEvaluator::FMEASURE);
    
    // Cleanup
    for (vector<Sample*>::iterator sample = samples.begin(); sample != samples.end(); sample++)
        delete *sample;
    
    return 0;
}


void listJPEGFiles(const char * dir, vector<string> & files)
{
#ifdef _WIN32
    WIN32_FIND_DATA findData;
    HANDLE hFind;
    hFind = FindFirstFile(join_path(2, dir, "*.jpg").c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            files.push_back(findData.cFileName);
        }
        while (FindNextFile(hFind, &findData));
        FindClose(hFind);
    }
#else
    DIR * dirp = opendir(dir);
    if (dirp != NULL)
    {
        struct dirent * entry;
        while ((entry = readdir(dirp)) != NULL)
        {
            string name(entry->d_name);
            if (name.length() > 4 && name.substr(name.length() - 4) == ".jpg")
                files.push_back(name);
        }
        closedir(dirp);
    }
#endif
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
