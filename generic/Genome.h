#ifndef __GENOME__
#define __GENOME__

class Genome {
public:
    virtual void initialize() = 0;
    virtual void evaluate() = 0;
    virtual void crossover(Genome *partner, Genome **offspring) = 0;
    virtual void mutate(float mutRate) = 0;
    virtual Genome *clone() = 0;
    virtual void scale(float base) = 0;
    virtual void print() = 0;
    // virtual void output(char *string) = 0;

    Genome(unsigned int xDim = 1, unsigned int yDim = 1, unsigned int zDim = 1) {
        xSize = xDim;
        ySize = yDim;
        zSize = zDim;
        score = 0;
        fitness = 0;
    }

    unsigned int getXSize() {
        return xSize;
    }
    unsigned int getYSize() {
        return ySize;
    }
    unsigned int getZSize() {
        return zSize;
    }
    float getScore() {
        return score;
    }
    float getFitness() {
        return fitness;
    }
    void setFitness(float fness) {
        fitness = fness;
    }
    void setScore(float scr) {
        score = scr;
    }

protected:
    unsigned int xSize;
    unsigned int ySize;
    unsigned int zSize;
    float score;
    float fitness;
};

#endif
