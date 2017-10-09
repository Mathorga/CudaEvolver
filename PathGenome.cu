#include <PathGenome.h>

void PathGenome::randomInitializer(GAGenome &genome) {
    for (int i = 0; i < this->checksNum; i++) {
        this->gene(i, );
    }
}

PathGenome::PathGenome(unsigned int checksNum) {
    this->checksNum = checksNum;
    this->path = (_2DDot *)malloc(checksNum * sizeof(2DDot));
}

PathGenome::PathGenome(const PathGenome &orig) {
    this->copy(orig);
}

__host__ __device__ float PathGenome::valuate() {
    if (_evaluated == gaFalse) {
        GAGenome *super = (GAGenome *)this;
        if (eval) {
            super->_neval++;
            super->_score = (*eval)(*super);
        }
        super->_evaluated = gaTrue;
    }
    return score;
}

void PathGenome::copy(const GAGenome &orig) {
    if (&orig == this) {
        return;
    }
    this->_score = orig._score;
    this->_fitness = orig._fitness;
    this->_evaluated=orig._evaluated;
    this->ga=orig.ga;
    this->ud=orig.ud;
    this->eval=orig.eval;
    this->init=orig.init;
    this->mutr=orig.mutr;
    this->cmp=orig.cmp;
    this->sexcross=orig.sexcross;
    this->asexcross=orig.asexcross;
    this->_neval = 0;

    this->checksNum = orig->checksNum;
    this->path = orig->path;

    if (orig.evd) {
        if (this->evd) {
            this->evd->copy(*orig.evd);
        } else {
            this->evd = orig.evd->clone();
        }
    }
}

PathGenome::~PathGenome() {
    free(this->path);
}

PathGenome::clone(GAGenome::CloneMethod flag) {
    PathGenome *copy = new PathGenome(this->checksNum);
    if (flag == CONTENTS) {
        copy->copy(*this);
    }
    return copy;
}

int PathGenome::equal(const GAGenome &g) const {
    PathGenome &genome = (PathGenome &)g;
    for (int i = 0; i < this->checksNum; i++) {
        if (this->gene(i) != genome->gene(i)) {
            return i;
        }
    }
    return 0;
}
