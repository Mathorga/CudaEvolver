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
    // std::cout << "\nbefore mutation:\n" << child << std::endl;

    _2DDot *tmp = (_2DDot *) malloc(child.checksNum * sizeof(_2DDot));

    for (unsigned int i = 0; i < child.checksNum; i++) {
        GARandomSeed();
        if (GARandomFloat(0, 1) <= mutRate) {
            int firstIndex = i;
            int secondIndex = GARandomInt(0, child.checksNum - 1);

            for (unsigned int j = 0; j < child.checksNum; j++) {
                tmp[j] = child.path[j];
            }

            child.path[firstIndex] = child.path[secondIndex];
            child.path[secondIndex] = tmp[firstIndex];
            nSwaps++;

        }
    }
    // std::cout << "\nafter mutation:\n" << child << std::endl;
    return nSwaps;
}

int PathGenome::onePointCrossover(const GAGenome &parent1, const GAGenome &parent2, GAGenome *child1, GAGenome *child2) {
    PathGenome &p1 = (PathGenome &) parent1;
    PathGenome &p2 = (PathGenome &) parent2;

    // std::cout << "\nparent 1:\n" << p1 << "parent 2:\n" << p2 << std::endl;

    int childrenNum = 0;

    if (child1 && child2) {
        // Two children crossover.
        PathGenome &c1 = (PathGenome &) *child1;
        PathGenome &c2 = (PathGenome &) *child2;

        // First child.
        unsigned int midPoint = GARandomInt(0, p1.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c1.gene(i, p1.gene(i));
            // std::cout << "from 1st parent so far:\n" << c1 << std::endl;
        }
        for (unsigned int i = midPoint + 1; i < p2.checksNum; ) {
            // printf("i = %d\tj = %d\n", i, j);
            for (unsigned int j = 0; j < p2.checksNum; j++) {
                bool insert = true;
                for (unsigned int k = 0; k <= midPoint; k++) {
                    // printf("k:%d\tj:%d\ti:%d\nc1:%d\tp2:%d\n\n", k, j, i, c1.gene(j).id, p2.gene(k).id);
                    // std::cout << "p2:\n" << p2 << std::endl;
                    if (p2.gene(j).id == c1.gene(k).id) {
                        // printf("insert is false\n");
                        insert = false;
                        break;
                    }
                }
                if (insert) {
                    // fflush(stdout);
                    // printf("%d\t%d\t%d\n", p2.gene(j).x, p2.gene(j).y, p2.gene(j).id);
                    c1.gene(i, p2.gene(j));
                    i++;
                    // std::cout << "from second parent so far:\n" << c1 << std::endl;
                }
            }
        }
        // std::cout << "\nchild 1:\n" << c1 << "midpoint:" << midPoint << std::endl;

        // Second child.
        midPoint = GARandomInt(0, p2.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c2.gene(i, p2.gene(i));
        }
        for (unsigned int i = midPoint + 1; i < p1.checksNum; ) {
            for (unsigned int j = 0; j < p1.checksNum; j++) {
                // printf("i = %d\n", i);
                bool insert = true;
                for (unsigned int k = 0; k <= midPoint; k++) {
                    if (p1.gene(j).id == c2.gene(k).id) {
                        insert = false;
                    }
                }
                if (insert) {
                    c2.gene(i, p1.gene(j));
                    i++;
                }
            }
        }

        // std::cout << "child 2:\n" << c2 << std::endl;
        childrenNum = 2;
    } else if (child1 || child2) {
        // Single child crossover
        PathGenome &c = (child1 ? ((PathGenome &) *child1) : ((PathGenome &) *child2));

        unsigned int midPoint = GARandomInt(0, p1.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c.gene(i, p1.gene(i));
        }
        for (unsigned int i = midPoint + 1; i < p2.checksNum; ) {
            // printf("i = %d\tj = %d\n", i, j);
            for (unsigned int j = 0; j < p2.checksNum; j++) {
                bool insert = true;
                for (unsigned int k = 0; k <= midPoint; k++) {
                    // printf("k:%d\tj:%d\ti:%d\nc1:%d\tp2:%d\n\n", k, j, i, c1.gene(j).id, p2.gene(k).id);
                    // std::cout << "p2:\n" << p2 << std::endl;
                    if (p2.gene(j).id == c.gene(k).id) {
                        // printf("insert is false\n");
                        insert = false;
                        break;
                    }
                }
                if (insert) {
                    c.gene(i, p2.gene(j));
                    i++;
                    // std::cout << "from second parent so far:\n" << c1 << std::endl;
                }
            }
        }

        // std::cout << "child:\n" << c << std::endl;
        childrenNum = 1;
    }

    return childrenNum;
}

float PathGenome::orderComparator(const GAGenome &a, const GAGenome &b) {
    return 0.5;
}

__global__ void cudaEval(PathGenome::_2DDot *path, float *distances) {
    float dx = (float) path[(threadIdx.x + 1) % blockDim.x].x - (float) path[threadIdx.x].x;
    float dy = (float) path[(threadIdx.x + 1) % blockDim.x].y - (float) path[threadIdx.x].y;
    distances[threadIdx.x] = sqrtf(powf(dx, 2) + powf(dy, 2));
}

float PathGenome::cudaEvaluator(GAGenome &g) {
    PathGenome *genome = &((PathGenome &) g);
    dim3 blockSize(genome->checksNum);

    // Allocate memory for the genome object's pointers on the device.
    PathGenome::_2DDot *d_path;
    float *d_distances;
    cudaMalloc(&d_path, genome->checksNum * sizeof(PathGenome::_2DDot));
    cudaMalloc(&d_distances, genome->checksNum * sizeof(float));

    // Copy the genome object' pointers on the device.
    cudaMemcpy(d_path, genome->getPath(), genome->checksNum * sizeof(PathGenome::_2DDot), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, genome->getDistances(), genome->checksNum * sizeof(float), cudaMemcpyHostToDevice);

    cudaEval<<<1, blockSize>>>(d_path, d_distances);
    cudaDeviceSynchronize();

    // Copy the object back.
    cudaMemcpy(genome->distances, d_distances, genome->checksNum * sizeof(float), cudaMemcpyDeviceToHost);

    float score = 0.0;
    for (unsigned int i = 0; i < genome->checksNum; i++) {
        // printf("distance %d on host:%f\n\n", i, genome->getDistances()[i]);
        score += genome->getDistances()[i];
    }
    // printf("score:%f\n\n", score);
    // printf("evaluated\n\n");

    cudaFree(d_path);
    cudaFree(d_distances);
    return score;
}



PathGenome::PathGenome(unsigned int checksNum) {
    this->checksNum = checksNum;
    this->checks = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    this->path = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    this->distances = (float *) malloc(checksNum * sizeof(float));
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->checks[i].x = 0;
        this->checks[i].y = 0;
        this->checks[i].id = 0;
    }
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->path[i].x = 0;
        this->path[i].y = 0;
        this->path[i].id = 0;
    }
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->distances[i] = 0.0;
    }

    initializer(PathGenome::randomInitializer);
    mutator(PathGenome::swapMutator);
    crossover(PathGenome::onePointCrossover);
    evaluator(PathGenome::cudaEvaluator);
}

PathGenome::PathGenome(unsigned int checksNum, _2DDot *checks) {
    this->checksNum = checksNum;
    this->checks = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    this->path = (_2DDot *) malloc(checksNum * sizeof(_2DDot));
    this->distances = (float *) malloc(checksNum * sizeof(float));
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->checks[i].x = checks[i].x;
        this->checks[i].y = checks[i].y;
        this->checks[i].id = checks[i].id;
    }
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->path[i].x = 0;
        this->path[i].y = 0;
        this->path[i].id = 0;
    }
    for (unsigned int i = 0; i < this->checksNum; i++) {
        this->distances[i] = 0.0;
    }

    initializer(PathGenome::randomInitializer);
    mutator(PathGenome::swapMutator);
    crossover(PathGenome::onePointCrossover);
    evaluator(PathGenome::cudaEvaluator);
}

PathGenome::PathGenome(const PathGenome &orig) {
    this->copy(orig);
}

void PathGenome::copy(const GAGenome &orig) {
    // std::cout << "\ncopy:\n" << (PathGenome &) orig << "\n";
    if (&orig == this) {
        return;
    }
    PathGenome *original = (PathGenome *) &orig;

    if (original) {
        GAGenome::copy(*original);
        this->checksNum = original->checksNum;
        this->checks = (_2DDot *) malloc(this->checksNum * sizeof(_2DDot));
        for (unsigned int i = 0; i < this->checksNum; i++) {
            // printf("copy loop\n");
            this->checks[i].x = original->checks[i].x;
            this->checks[i].y = original->checks[i].y;
            this->checks[i].id = original->checks[i].id;
        }
        this->path = (_2DDot *) malloc(this->checksNum * sizeof(_2DDot));
        for (unsigned int i = 0; i < this->checksNum; i++) {
            // printf("copy loop\n");
            this->path[i].x = original->path[i].x;
            this->path[i].y = original->path[i].y;
            this->path[i].id = original->path[i].id;
        }
        this->distances = (float *) malloc(this->checksNum * sizeof(float));
        for (unsigned int i = 0; i < this->checksNum; i++) {
            // printf("copy loop\n");
            this->distances[i] = original->distances[i];
        }
    }
}

PathGenome::~PathGenome() {
    free(this->checks);
    free(this->path);
    free(this->distances);
}

GAGenome *PathGenome::clone(GAGenome::CloneMethod flag) const {
    // printf("\nclone\n");
    PathGenome *cpy = new PathGenome(this->checksNum, this->checks);
    if (flag == CONTENTS) {
        cpy->copy(*this);
    }
    else {
        cpy->GAGenome::copy(*this);
        for (unsigned int i = 0; i < this->checksNum; i++) {
            // printf("clone loop\n");
            cpy->checksNum = this->checksNum;
            // cpy->gene(i, this->gene(i));
        }
    }
    return cpy;
}

#ifdef GALIB_USE_STREAMS
int PathGenome::read(STD_ISTREAM &is) {
    return 0;
}

int PathGenome::write(STD_OSTREAM &os) const {
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
