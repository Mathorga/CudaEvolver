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
            // std::cout << "from 1st parent so far:\n" << c1 << std::endl;
        }
        for (unsigned int i = midPoint + 1; i < p2.checksNum; ) {
            // printf("i = %d\tj = %d\n", i, j);
            for (unsigned int j = 0; j < p2.checksNum; j++) {
                bool insert = true;
                for (unsigned int k = 0; k <= midPoint; k++) {
                    if (p2.gene(j).id == c1.gene(k).id) {
                        // printf("insert is false\n");
                        insert = false;
                    }
                }
                if (insert) {
                    c1.gene(i, p2.gene(j));
                    i++;
                    // std::cout << "from second parent so far:\n" << c1 << std::endl;
                }
            }
        }

        // Second child.
        midPoint = GARandomInt(0, p2.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c2.gene(i, p2.gene(i));
        }
        for (unsigned int i = midPoint + 1; i < p1.checksNum; ) {
            for (unsigned int j = 0; j < p1.checksNum; j++) {
                printf("i = %d\n", i);
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

        childrenNum = 2;
    } else if (child1 || child2) {
        // Single child crossover
        PathGenome &c = (child1 ? ((PathGenome &) *child1) : ((PathGenome &) *child2));

        unsigned int midPoint = GARandomInt(0, p1.checksNum - 1);
        for (unsigned int i = 0; i <= midPoint; i++) {
            c.gene(i, p1.gene(i));
        }
        for (unsigned int i = midPoint + 1, j = 0; i < p2.checksNum && j < p2.checksNum; i++) {
            for (unsigned int k = 0; k < midPoint; k++) {
                if (p2.gene(j).id != c.gene(k).id) {
                    c.gene(i, p2.gene(j));
                    j++;
                }
            }
        }

        childrenNum = 1;
    }

    return childrenNum;
}

__global__ void cudaEval(PathGenome *genome) {
    float distance = 0.0;
    float dx = (float) genome->gene((threadIdx.x + 1) % genome->getChecksNum()).x - (float) genome->gene(threadIdx.x).x;
    float dy = (float) genome->gene((threadIdx.x + 1) % genome->getChecksNum()).y - (float) genome->gene(threadIdx.x).y;
    distance = sqrtf(powf(dx, 2) + powf(dy, 2));
    genome->setDistance(threadIdx.x, distance);
}

float PathGenome::cudaEvaluator(GAGenome &g) {
    PathGenome *genome = &((PathGenome &) g);
    dim3 blockSize(genome->checksNum);

    // Allocate memory for the genome object on the device
    PathGenome *d_genome;
    cudaMalloc(&d_genome, sizeof(PathGenome));

    // Copy the genome object to the device.
    cudaMemcpy(d_genome, genome, sizeof(PathGenome), cudaMemcpyHostToDevice);

    // Allocate memory for the genome object's pointers on the device.
    PathGenome::_2DDot *d_path;
    float *d_distances;
    cudaMalloc(&d_path, genome->checksNum * sizeof(PathGenome::_2DDot));
    cudaMalloc(&d_distances, genome->checksNum * sizeof(float));

    // Copy the genome object' pointers on the device.
    cudaMemcpy(d_path, genome->getPath(), genome->checksNum * sizeof(PathGenome::_2DDot), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, genome->getDistances(), genome->checksNum * sizeof(float), cudaMemcpyHostToDevice);

    // Copy pointers values correct locations on the device.
    cudaMemcpy(&(d_genome->path), &d_path, sizeof(PathGenome::_2DDot *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_genome->distances), &d_distances, sizeof(float *), cudaMemcpyHostToDevice);

    cudaEval<<<1, 10>>>(d_genome);
    cudaDeviceSynchronize();

    // Copy the object back.
    cudaMemcpy(genome->distances, d_distances, genome->checksNum * sizeof(float), cudaMemcpyDeviceToHost);

    float score = 0.0;
    for (unsigned int i = 0; i < genome->checksNum; i++) {
        score += genome->getDistances()[i];
    }

    cudaFree(d_path);
    cudaFree(d_distances);
    cudaFree(d_genome);
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
    PathGenome *cpy = new PathGenome(this->checksNum, this->checks);
    if (flag == CONTENTS) {
        cpy->copy(*this);
    }
    else {
        cpy->GAGenome::copy(*this);
        for (unsigned int i = 0; i < this->checksNum; i++) {
            cpy->gene(i, this->gene(i));
        }
    }
    return cpy;
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
