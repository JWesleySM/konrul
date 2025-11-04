module {
  func.func @matmul(%matA: memref<3x3xf64>, %matB: memref<3x3xf64>, %matC: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            %c0 = arith.constant 0.0 : f64
            affine.store %c0, %matC[%i, %j] : memref<3x3xf64>

            affine.for %k = 0 to 3 {
                %0 = affine.load %matC[%i, %j] : memref<3x3xf64>
                %1 = affine.load %matA[%i, %k] : memref<3x3xf64>
                %2 = affine.load %matB[%k, %j] : memref<3x3xf64>
                %3 = arith.mulf %1, %2 : f64
                %4 = arith.addf %0, %3 : f64
                affine.store %4, %matC[%i, %j] : memref<3x3xf64>
            }
        }
    }
    return
  }
}
