module {
  func.func @vcopy(%a: memref<3xf64>, %b: memref<3xf64>) {
    affine.for %i = 0 to 3 {
        %0 = affine.load %a[%i] : memref<3xf64>
        affine.store %0, %b[%i] : memref<3xf64>
    }
    return
  }
}
