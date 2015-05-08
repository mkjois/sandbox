#lang racket

; This file: Arithmetic expressions and variables.

; This segment is a compiler from YL to formula ASTs

; We now add declafration of variables.

(define-syntax-rule (define-YL-var v ...)
  (begin 
    (define v 'v) ...
    '(declare v) ...
    ))

(define (+ x y)
  `(+ ,x ,y))

; This is a sample YL program

(define-YL-var x y)

(+ x (+ 1 y))
