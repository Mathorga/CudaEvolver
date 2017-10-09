#ifndef __PATH_GENOME__
#define __PATH_GENOME__

#include <ga/GAGenome.h>

struct 2DDot {
    unsigned int x;
    unsigned int y;
};

class PathGenome : public GAGenome {
public:
    GADefineIdentity("PathGenome", 201);
    // Randomly initializes the genome.
    static void randomInitializer(GAGenome &);
    //
    static int flipMutator(GAGenome &, float);
    static float Comparator(const GAGenome &, const GAGenome &);
    static float Evaluator(GAGenome &);
    static void PathInitializer(GAGenome &);

    // Constructors.
    PathGenome(int checksNum);
    PathGenome(const PathGenome &orig);

    // Hide superclass' evaluate member function.
    __host__ __device__ float evaluate();

    // Destructors.
    virtual ~PathGenome();
    virtual GAGenome *clone(GAGenome::CloneMethod flag = CONTENTS) const;
    virtual void copy(const GAGenome &c);
    virtual int equal(const GAGenome &g) const;
    short gene(unsigned int x = 0) const {
        return this->path[x];
    }
    short gene(unsigned int x, unsigned int value);

    2DDot &path() {
        return this->path;
    }
protected:
    int checksNum;
    2DDot *path;
};

#endif
