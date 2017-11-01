#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ga/ga.h>

#define POP_SIZE 10
#define GEN_NUMBER 100
#define MUT_PROBABILITY 0.001


// |-----------------------------------------------------------------------------------------|
// Fitness function test.
// |-----------------------------------------------------------------------------------------|
float fitness(GAGenome &g) {
    GA1DBinaryStringGenome &genome = (GA1DBinaryStringGenome &)g;

    float score=0.0;
    for (int i = 0; i < genome.length(); i++) {
        // The more 1s are contained in the string, the higher is the fitness.
        // The score is incremented by the value of the current element of the string (0 or 1).
        score += genome.gene(i);
    }
    return score;
}
// |-----------------------------------------------------------------------------------------|



// |-----------------------------------------------------------------------------------------|
// CUDA fitness function test.
// |-----------------------------------------------------------------------------------------|
float CUDAFitness(GAGenome &g) {
    GA1DBinaryStringGenome &genome = (GA1DBinaryStringGenome &)g;

    float score=0.0;
    for (int i = 0; i < genome.length(); i++) {
        // The more 1s are contained in the string, the higher is the fitness.
        // The score is incremented by the value of the current element of the string (0 or 1).
        score += genome.gene(i);
    }
    return score;
}
// |-----------------------------------------------------------------------------------------|



// |-----------------------------------------------------------------------------------------|
// Initializers.
// |-----------------------------------------------------------------------------------------|
void randomInitializer(GAGenome &g) {
    GA1DBinaryStringGenome &genome=(GA1DBinaryStringGenome &)g;

    for (int i = 0; i < genome.size(); i++) {
        genome.gene(i, GARandomBit());
    }
}

void worstCaseInitializer(GAGenome &g) {
    GA1DBinaryStringGenome &genome=(GA1DBinaryStringGenome &)g;

    for (int i = 0; i < genome.size(); i++) {
        genome.gene(i, 0);
    }
}
// |-----------------------------------------------------------------------------------------|



// |-----------------------------------------------------------------------------------------|
// CUDA useless population evaluator.
// |-----------------------------------------------------------------------------------------|
__global__ void cudaHello() {
    printf("hello, I am thread (%d-%d-%d) of block (%d-%d-%d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

void testEvaluator(GAPopulation &p) {
    // dim3 blockSize(p.size(), 10);
    // cudaHello<<<1, blockSize>>>();
    // cudaDeviceSynchronize();

    for (int i = 0; i < p.size(); i++) {
        p.individual(i).evaluate();
    }
}
// |-----------------------------------------------------------------------------------------|



// |-----------------------------------------------------------------------------------------|
// CUDA population evaluator.
// |-----------------------------------------------------------------------------------------|
__global__ void evaluate(GAPopulation &pop) {
    // pop.individual(threadIdx.x).evaluate();
}

void cudaEvaluator(GAPopulation &p) {
    dim3 blockSize(p.size());

    // TODO Allocate device memory for the population and cudaMemcpy it.
    // cudaMalloc();
    // cudaMemcpy();

    evaluate<<<1, blockSize>>>(p);
}
// |-----------------------------------------------------------------------------------------|



// |-----------------------------------------------------------------------------------------|
// Main.
// |-----------------------------------------------------------------------------------------|
int main(int argc, char const *argv[]) {
    // Create a genome.
    GA1DBinaryStringGenome genome(20, fitness);
    genome.initializer(randomInitializer);

    // Create a population.
    GAPopulation population(genome, POP_SIZE);
    population.evaluator(testEvaluator);

    // Create the genetic algorithm.
    GASimpleGA ga(population);
    ga.nGenerations(GEN_NUMBER);
    ga.pMutation(MUT_PROBABILITY);

    ga.initialize();

    GAPopulation tmpPop = ga.population();
    printf("\nInitial population:\n");
    for (int i = 0; i < tmpPop.size(); i++) {
        printf("Individual %d: ", i);
        GA1DBinaryStringGenome& individual = (GA1DBinaryStringGenome&)tmpPop.individual(i);
        for (int j = 0; j < individual.length(); j++) {
            printf("%d", individual.gene(j));
        }
        printf("\n");
    }
    printf("\nBest: ");
    GA1DBinaryStringGenome &currentBest = (GA1DBinaryStringGenome &)tmpPop.best();
    for (int i = 0; i < currentBest.length(); i++) {
        printf("%d", currentBest.gene(i));
    }
    printf("\n\n");


    for (int i = 0; i < ga.nGenerations(); i++) {
        // getchar();
        printf("\n\n\nGENERATION %d\n", ga.generation() + 1);
        ga.step();
        GAPopulation tmpPop = ga.population();
        // Print the population.
        printf("\nPopulation:\n");
        for (int i = 0; i < tmpPop.size(); i++) {
            printf("Individual %d: ", i);
            GA1DBinaryStringGenome& individual = (GA1DBinaryStringGenome&)tmpPop.individual(i);
            for (int j = 0; j < individual.length(); j++) {
                printf("%d", individual.gene(j));
            }
            printf("\n");
        }
        printf("\nBest: ");
        currentBest = (GA1DBinaryStringGenome &)tmpPop.best();
        for (int i = 0; i < currentBest.length(); i++) {
            printf("%d", currentBest.gene(i));
        }
        printf("\tfitness: %f", tmpPop.max());
        printf("\n\n");

        // Print statistics.
        // std::cout << ga.statistics() << std::endl;
    }
    return 0;
}
// |-----------------------------------------------------------------------------------------|
