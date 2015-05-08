#lang racket

; This file: overriding 'if'

(require (only-in racket (if racket/if)))

(define-syntax-rule (if e t f)  ; Quiz: can this macro be a call?
  (racket/if (boolean? e)      ; Quiz: should we add a conjunct that e is true on rhs?
             (racket/if e t f)    ; Quiz: is partial evaluation based on e necessary? Yes because in def + we use own if not the orig if
             `(ite ,e ,t ,f)))    ; Quiz: influence of target formula language F.  How would we would have to translate if F contained no (ite)?


; no change for operators and variable definitions
  
(define-syntax-rule (lift-ops op ...)
  (begin (lift-op op) ... ))
    
(define-syntax-rule (lift-op op)
  (begin 
    (require (only-in racket (op opn)))
    (define (op x y)
      (if (and (number? x) (number? y)) ; are both args concrete?
          (opn x y)
          `(op ,x ,y)))))

(lift-ops + - * / =)

(define-syntax-rule (define-YL-var v ...)
  (begin 
    (define v 'v) ...
    '(declare v) ...
    ))



; This is a YL program

(define-YL-var n x a b c)

(define (polynomial n x coefs)
  (if (= n 0)
      (car coefs)
      (+ (car coefs) (* x (polynomial (- n 1) x (cdr coefs))))))

(polynomial 2 2 (list 1 2 3))
(polynomial 2 x (list 1 2 3))
(polynomial 2 x (list a b c))
(polynomial n x (list a b c))  ; if now works correctly but the unwinding does not stop ==> we must bound it 