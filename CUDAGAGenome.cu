#include <CudaGaGenome.h>

CUDA_MEMBER float CudaGaGenome::evaluate(GABoolean flag) {
   if(_evaluated == gaFalse || flag == gaTrue){
       GAGenome *This = (GAGenome *)this;
       if (eval) {
           This->_neval++;
           This->_score = (*eval)(*This);
       }
       This->_evaluated = gaTrue;
   }
   return _score;
}
