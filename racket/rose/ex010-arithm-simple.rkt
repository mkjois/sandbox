#lang racket

; This file: Arithmetic expressions and variables.

; This segment is a "compiler" from YL to formula ASTs.
; It compiles by overriding the raket's built-in + function.

(define (+ x y)
  `(+ ,x ,y))

; This is a sample YL program. Evaluation of this program
; has the effect of compiling this program to a formula.

(+ 1 2)

(+ 'x (+ 1 'y))
