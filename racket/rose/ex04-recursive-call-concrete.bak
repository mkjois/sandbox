#lang racket

; problems illustrated:
;   sub1 expects number, not 'x
;   fib will unroll forever

(require (only-in racket (+ orig+)(- orig-)))

(define (fib x) 
  (case x 
    [(0 1) 1] 
;    [else (+ (fib (sub1 x)) (fib (- x 2)))]
    [else (+ (fib (- x 1)) (fib (- x 2)))]
  ))

(define (+ x y)
  (if (and (number? x) (number? y))
      (orig+ x y)
      `(+ ,x ,y)))

; create a macro for +, -, ...

(define (- x y)
  (if (and (number? x) (number? y))
      (orig- x y)
      `(- ,x ,y)))

; fully evaluated
(+ 1 (fib 6))

; show that we don't need to be always symbolic
(+ 'x (fib 5))

; this does not terminate
(fib 'x)
