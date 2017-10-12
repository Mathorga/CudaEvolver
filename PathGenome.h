#ifndef __PATH_GENOME__
#define __PATH_GENOME__

#define __CUDA__ __host__ __device__

#include <ga/GAGenome.h>

class PathGenome : public GAGenome {
public:
    typedef struct {
        unsigned int x = 0;
        unsigned int y = 0;
        unsigned int id = 0;
    } _2DDot;

    GADefineIdentity("PathGenome", 201);
    // Initializes the genome connecting the checks in the order they appear.
    static void linearInitializer(GAGenome &);
    // Randomly initializes the genome.
    static void randomInitializer(GAGenome &);
    // Swaps couples of checks. The amount of swaps depends on the mutation rate.
    static int swapMutator(GAGenome &, float);
    // Performs crossover taking a random part of one parent and adding to it the missing checks from the other parent
    // in the order they appear in it.
    static int onePointCrossover(const GAGenome &, const GAGenome &, GAGenome *, GAGenome *);

    // static float Comparator(const GAGenome &, const GAGenome &);

    static float cudaEvaluator(GAGenome &);

    // Constructors.
    PathGenome(unsigned int checksNum);
    PathGenome(unsigned int checksNum, _2DDot *checks);
    PathGenome(const PathGenome &orig);

    PathGenome &operator=(const GAGenome &arg) {
        copy(arg);
        return *this;
    }

    // Hide superclass' evaluate member function.
    // float evaluate(GABoolean flag = gaFalse) const ;

    // Destructors.
    virtual ~PathGenome();

    virtual GAGenome *clone(GAGenome::CloneMethod flag = CONTENTS) const;
    virtual void copy(const GAGenome &c);

    #ifdef GALIB_USE_STREAMS
        virtual int read(STD_ISTREAM &is);
        virtual int write(STD_OSTREAM &os) const;
    #endif

    virtual int equal(const GAGenome &g) const;
    __CUDA__ _2DDot gene(unsigned int idx = 0) const {
        return this->path[idx];
    }
    _2DDot gene(unsigned int idx, unsigned int x, unsigned int y) {
        this->path[idx].x = x;
        this->path[idx].y = y;
        return this->path[idx];
    }
    _2DDot gene(unsigned int idx, _2DDot check) {
        this->path[idx].x = check.x;
        this->path[idx].y = check.y;
        this->path[idx].id = check.id;
        return this->path[idx];
    }

    // Setters.
    __CUDA__ float setDistance(int idx, float dist) {
        return (this->distances[idx] = dist);
    }

    // Getters.
    __CUDA__ unsigned int getChecksNum() {
        return this->checksNum;
    }
    __CUDA__ _2DDot *getChecks() {
        return this->checks;
    }
    __CUDA__ _2DDot *getPath() {
        return this->path;
    }
    __CUDA__ float *getDistances() {
        return this->distances;
    }
protected:
    unsigned int checksNum;
    _2DDot *checks;
    _2DDot *path;
    float *distances;
};

#endif
