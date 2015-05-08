#lang racket

; (define-syntax-rule (my-rename)
;   (require (only-in racket (+ plus))))

; (my-rename)
; 
; this call fails becasue plus is not in scope
;
; (plus 1 2)

(define-syntax-rule (my-rename op)
  (require (only-in racket (+ op))))

(my-rename plus)
 
; this call now succeeds because plus was introduced 
; in our scope, when the macro was invoked

(plus 1 2)
