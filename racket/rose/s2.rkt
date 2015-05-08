#lang racket

; rename the racket + so that we can refer to it during partial evaluation

(require (rename-in racket (+ orig+)))

; online partial evaluation creates a residual program that evaluates all 
; concrete subexpressions

(define (+ x y)
  (if (and (number? x) (number? y))
      (orig+ x y)
      `(+ ,x ,y)))

(+ 1 2)

(+ 'x (+ 1 2))

