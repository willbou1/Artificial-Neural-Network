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
    void saveANN(ANN *ann);
    ANN *readANN();
	void saveWordClassifier(WordClassifier *wordClassifier);
	WordClassifier *readWordClassifier();
    
private:
    void check();
    int countLines();
    int m_nbLines;
    string m_path;
    fstream m_file;
};

#endif