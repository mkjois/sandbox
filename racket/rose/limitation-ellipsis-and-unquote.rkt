#lang racket

; assume we want to create a list of certain expressions, here (+ x 1) ...

(define-syntax (a stx)
  (syntax-case stx ()
    [(_ x ...) 
     #`'((+ x 1) ...)]))

(a 1 2 3) ; --> '((+ 1 1) (+ 2 1) (+ 3 1))


; now assume I want to partially evaluate this list.
; this is easy when with a one-element list

(define-syntax (b stx)
  (syntax-case stx ()
    [(_ x) 
     #`'(#,(+ (syntax->datum #'x) 1))]))

(b 1)  ; --> '(2)

; Unfortunately, the straingtforward addition of ... into the macro b 
; fails.  The problem: ... and its patter variable cannot be separated 
; by unquote computation (here the + 1).
;
;(define-syntax (c stx)
;  (syntax-case stx ()
;    [(_ x ...) 
;     #`'(#,(+ (syntax->datum #'x) 1) ...)]))
;
; (a 1 2 3)
