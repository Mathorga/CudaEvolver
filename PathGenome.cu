#include <ga/garandom.h>
#include "PathGenome.h"

void PathGenome::linearInitializer(GAGenome &genome) {
    PathGenome &child = (PathGenome &) genome;
    for (unsigned int i = 0; i < child.checksNum; i++) {
        child.path[i] = child.checks[i];
    }
}

void PathGenome::randomInitializer(GAGenome &genome) {
    PathGenome &child = (PathGenome &) genome;
    // Create a copy of the checks array.
    _2DDot *checksCopy = (_2DDot *) malloc(child.checksNum * sizeof(_2DDot));
    for (unsigned int i = 0; i < child.checksNum; i++) {
        checksCopy[i] = child.checks[i];
    }

    for (unsigned int i = 0; i < child.checksNum; i++) {
        int index = rand() % (child.checksNum - i);
        child.gene(i, checksCopy[index]);
        for (unsigned int j = index; j < child.checksNum - 1; j++) {
            checksCopy[j] = checksCopy[j + 1];
        }
    }
}

int PathGenome::swapMutator(GAGenome &genome, float mutRate) {
    PathGenome &child = (PathGenome &) genome;
    int nSwaps = 0;

    _2DDot *tmp = (_2DDot *) malloc(child.checksNum * sizeof(_2DDot));

    for (unsigned int i = 0; i < child.checksNum; i++) {
        GARandomSeed();
        if (GARandomFloat(0, 1) <= mutRate / 2) {
            int firstIndex = i;
            int secondIndex = GARandomInt(0, child.checksNum - 1);

            for (unsigned int j = 0; j < child.checksNum; j++) {
                tmp[j] = child.path[j];
            }

            printf("firstIndex:%d\tsecondIndex:%d\n", firstIndex, secondIndex);

            child.path[firstIndex] = child.path[secondIndex];
            child.path[secondIndex] = tmp[firstIndex];
            nSwaps++;

            std::cout << child << "\n";
        }
    }
    return nSwaps;
}

int PathGenome::onePointCrossover(const GAGenome &parent1, const GAGenome &parent2, GAGenome *child1, GAGenome *child2) {
    PathGenome &p1 = (PathGenome &) parent1;
    PathGenome &p2 = (PathGenome &) parent2;

    int childrenNum = 0;

    if (child1 && child2) {
        // Two children crossover.
        PathGenome &c1 = (PathGenome &) *child1;
        PathGenome &c2 = (PathGenome &) *child2;

        // First child.
        unsigned int midPoint = GARandomInt(0, p1.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c1.gene(i, p1.gene(i));
        }
        unsigned int j = 0;
        for (unsigned int i = midPoint + 1; i < p2.checksNum, j < p2.checksNum; i++) {
            for (unsigned int k = 0; k < midPoint; k++) {
                if (p2.gene(j).id != c1.gene(k).id) {
                    c1.gene(i) = p2.gene(j);
                    j++;
                }
            }
            printf("i = %d\tj = %d\np2.checksNum = %d\n", i, j, p2.checksNum);
        }

        printf("\nFirst child done\n");

        // Second child.
        midPoint = GARandomInt(0, p2.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c2.gene(i, p2.gene(i));
        }
        for (unsigned int i = midPoint + 1, j = 0; i < p1.checksNum, j < p1.checksNum; i++) {
            for (unsigned int k = 0; k < midPoint; k++) {
                if (p1.gene(j).id != c2.gene(k).id) {
                    c2.gene(i) = p1.gene(j);
                    j++;
                }
            }
        }

        printf("\nSecond child done\n");

        childrenNum = 2;
    } else if (child1 || child2) {
        // Single child crossover
        PathGenome &c = (child1 ? ((PathGenome &) *child1) : ((PathGenome &) *child2));

        unsigned int midPoint = GARandomInt(0, p1.checksNum - 1);
        printf("\nmidPoint: %d\n", midPoint);

        for (unsigned int i = 0; i <= midPoint; i++) {
            c.gene(i, p1.gene(i));
        }
        for (unsigned int i = midPoint + 1, j = 0; i < p2.checksNum, j < p2.checksNum; i++) {
            for (unsigned int k = 0; k < midPoint; k++) {
                if (p2.gene(j).id != c.gene(k).id) {
                    c.gene(i) = p2.gene(j);
                    j++;
                }
            }
        }

        childrenNum = 1;
    }

    return childrenNum;
}

PathGenome::PathGenome(unsigned int checksNum) {
    this->checksNum = checksNum;
    this->checks = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    this->path = (_2DDot *) malloc(checksNum * sizeof(_2DDot));

    this->init = PathGenome::randomInitializer;
    this->mutr = PathGenome::swapMutator;
}

PathGenome::PathGenome(unsigned int checksNum, _2DDot *checks) {
    this->checksNum = checksNum;
    this->checks = checks;
    this->path = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->path[i].x = 0;
        this->path[i].y = 0;
        this->path[i].id = 0;
    }

    this->init = PathGenome::randomInitializer;
    this->mutr = PathGenome::swapMutator;
}

PathGenome::PathGenome(const PathGenome &orig) {
    this->copy(orig);
}

__host__ __device__ float PathGenome::evaluate() {
    if (_evaluated == gaFalse) {
        // GAGenome *super = (GAGenome *)this;
        if (eval) {
            this->_neval++;
            this->_score = (*eval)(*this);
        }
        this->_evaluated = gaTrue;
    }
    return this->_score;
}

void PathGenome::copy(const GAGenome &orig) {
    if (&orig == this) {
        return;
    }
    PathGenome *original = (PathGenome *) &orig;

    GAGenome::copy(*original);
    checksNum = original->checksNum;
    checks = original->checks;
    path = original->path;
}

PathGenome::~PathGenome() {
    free(this->path);
}

GAGenome *PathGenome::clone(GAGenome::CloneMethod flag) const {
    PathGenome *copy = new PathGenome(this->checksNum);
    if (flag == CONTENTS) {
        copy->copy(*this);
    }
    return copy;
}

#ifdef GALIB_USE_STREAMS
int PathGenome::read(STD_ISTREAM &is) {
    return 0;
}

int PathGenome::write(std::ostream &os) const {
    for (unsigned int i = 0; i < this->checksNum; i++) {
        os << i << "\tx:" << this->gene(i).x << "\ty:" << this->gene(i).y << "\tid:" <<this->gene(i).id << "\n";
    }
    return 0;
}
#endif

int PathGenome::equal(const GAGenome &g) const {
    PathGenome &genome = (PathGenome &)g;
    for (unsigned int i = 0; i < this->checksNum; i++) {
        if (this->gene(i).x != genome.gene(i).x &&
            this->gene(i).y != genome.gene(i).y) {
            return i;
        }
    }
    return 0;
}
