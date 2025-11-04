module {
  func.func @matadd(%matA: memref<3x3xf64>, %matB: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
          %0 = affine.load %matA[%i, %j] : memref<3x3xf64>
          %1 = affine.load %matB[%i, %j] : memref<3x3xf64>
          %2 = arith.addf %0, %1 : f64
          affine.store %2, %matA[%i, %j] : memref<3x3xf64>
        }
    }
    return
  }
}
