module {
  func.func @gemv(%A: memref<3x3xf64>, %x: memref<3xf64>, %y: memref<3xf64>) {
    affine.for %i = 0 to 5 {
        %c0 = arith.constant 0.0 : f64
        affine.store %c0, %y[%i] : memref<3xf64>

        affine.for %j = 0 to 5 {
            %0 = affine.load %A[%i, %j] : memref<3x3xf64>
            %1 = affine.load %x[%j] : memref<3xf64>
            %2 = arith.mulf %0, %1 : f64

            %3 = affine.load %y[%i] : memref<3xf64>
            %4 = arith.addf %3, %2 : f64

            affine.store %4, %y[%i] : memref<3xf64>
        }
    }
    return
  }
}
