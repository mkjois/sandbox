#lang racket

; This file: Arithmetic expressions and variables, now with partial evaluation.

; This segment is a compiler from YL to formula ASTs

; First, rename the racket + so that we can refer to it during partial evaluation

(require (only-in racket (+ racket/+)))

; online partial evaluation creates a residual program that is free
; of all subexpressions that can be evaluated at compilation time

(define (+ x y)
  (if (and (number? x) (number? y)) ; are both args concrete?
      (racket/+ x y)
      `(+ ,x ,y)))

; no change here:

(define-syntax-rule (define-YL-var v ...)
  (begin 
    (define v 'v) ...
    '(declare v) ...
    ))

; This is a sample YL program. 

(define-YL-var x)
(+ 1 2)
(+ x (+ 1 2))
(+ 1 (+ x 2))
