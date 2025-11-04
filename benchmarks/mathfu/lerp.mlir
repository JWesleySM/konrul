module {
  func.func @lerp(%out: memref<3xf64>, %x: memref<3xf64>, %y: memref<3xf64>, %alpha: f64) {
    %c0 = arith.constant 1.0 : f64
    affine.for %i = 0 to 3 {
        %x_i = affine.load %x[%i] : memref<3xf64>
        %y_i = affine.load %y[%i] : memref<3xf64>
        %out_i = arith.mulf %alpha, %x_i : f64
        %out_i_1 = arith.subf %c0, %alpha : f64
        %out_i_2 = arith.mulf %out_i_1, %y_i : f64
        %out_i_3 = arith.addf %out_i, %out_i_2 : f64
        affine.store %out_i_3, %out[%i] : memref<3xf64>
    }
    return
  }
}
