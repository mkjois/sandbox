#lang racket

(define-syntax (a stx)
  (syntax-case stx ()
    [(_ x ...)
     #`'((+ x 1) ...)]))

(a 1 2 3)

(define-syntax (b stx)
  (syntax-case stx ()
    [(_ x)
     #`'(#,(+ (syntax->datum #'x) 1))]))

(b 1)

;(define-syntax (c stx)
;  (syntax-case stx ()
;    [(_ x ...)
;     #`'(#,(+ #'x 1) ...)]))
;
; (a 1 2 3)
