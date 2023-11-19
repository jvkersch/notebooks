//
//  main.swift
//  metal-accelerate-test
//
//  Created by Joris on 2023/11/15.
//

import Foundation
import Metal
import Accelerate

func populateMatrixBuffer(buffer: MTLBuffer) {
    let count = buffer.length / MemoryLayout<Float>.size
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)

    for i in 0..<count {
        pointer.advanced(by: i).pointee = Float(i + 1)
    }
}

func printBufferContents(buffer: MTLBuffer) {
    let count = buffer.length / MemoryLayout<Float>.size
    let dataPointer = buffer.contents().bindMemory(to: Float.self, capacity: count)

    for i in 0..<count {
        let value = dataPointer.advanced(by: i).pointee
        print("Element \(i): \(value)")
    }
}

// Assuming you're working with a 4x4 matrix of floats
let rows = 4
let columns = 4
let elementSize = MemoryLayout<Float>.size
let bufferSize = rows * columns * elementSize

guard let device = MTLCreateSystemDefaultDevice(),
      let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
    fatalError("Unable to create MTLBuffer")
}

// Assuming you have a method to populate the buffer
populateMatrixBuffer(buffer: buffer)

print("BEFORE")
printBufferContents(buffer: buffer)


// Now, you can use this buffer in your Metal compute or graphics pipeline
let contents = buffer.contents()
// Cast the pointer to the appropriate type
let floatPointer = contents.bindMemory(to: Float.self, capacity: bufferSize)
    
// Now, you can use floatPointer with BLAS routines
// For example, calling a BLAS routine:
let alpha = Float(0.5)
cblas_saxpy(Int32(rows * columns), alpha, floatPointer, 1, floatPointer, 1)

print("AFTER")
printBufferContents(buffer: buffer)
