#ifndef __PATH_GENOME__
#define __PATH_GENOME__

#include <ga/GAGenome.h>

class PathGenome : public GAGenome {
public:
    typedef struct {
        unsigned int x = 0;
        unsigned int y = 0;
        bool checked = false;
    } _2DDot;

    GADefineIdentity("PathGenome", 201);
    // Initializes the genome connecting the checks in the order they appear.
    static void linearInitializer(GAGenome &);
    // Randomly initializes the genome.
    static void randomInitializer(GAGenome &);
    //
    // static int flipMutator(GAGenome &, float);
    // static float Comparator(const GAGenome &, const GAGenome &);
    // static float Evaluator(GAGenome &);
    // static void PathInitializer(GAGenome &);

    // Constructors.
    PathGenome(unsigned int checksNum);
    PathGenome(unsigned int checksNum, _2DDot *checks);
    PathGenome(const PathGenome &orig);

    // Hide superclass' evaluate member function.
    __host__ __device__ float evaluate();

    // Destructors.
    virtual ~PathGenome();
    virtual GAGenome *clone(GAGenome::CloneMethod flag = CONTENTS) const;
    virtual void copy(const GAGenome &c);
    virtual int equal(const GAGenome &g) const;
    _2DDot gene(unsigned int idx = 0) const {
        return this->path[idx];
    }
    short gene(unsigned int idx, unsigned int x, unsigned int y);

    _2DDot *getPath() {
        return this->path;
    }
protected:
    int checksNum;
    _2DDot *checks;
    _2DDot *path;
};

#endif
