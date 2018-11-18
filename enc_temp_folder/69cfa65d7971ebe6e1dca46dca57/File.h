#ifndef FILE_HEADER
#define FILE_HEADER

using namespace std;

class File {
public:
    File(string path);
    ~File();
    void open(fstream::openmode openMode = fstream::in);
    void close();
    TrainingSet *readTrainingSet();
    void saveANN(ANN *);
    ANN *readANN();
    
private:
    void check();
    int countLines();
    int nbLines;
    string path;
    fstream file;
};

#endif