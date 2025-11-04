module {
  func.func @matmul_sca(%matA: memref<3x3xf64>, %matB: memref<3x3xf64>, %val: f64) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            %v = affine.load %matA[%i, %j] : memref<3x3xf64>
            %res = arith.mulf %v, %val : f64
            affine.store %res, %matB[%i, %j] : memref<3x3xf64>
        }
    }
    return
  }
}
