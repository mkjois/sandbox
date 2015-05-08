#lang racket

; No new features are implementated in this file.  It illustrates: 
;   sub1 expects number, not a symbolic value
;   fib will unroll forever


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


; This is a polynomials program

(define-YL-var a b c n x)

(define (polynomial n x coefs)
  (if (= n 0)
      (car coefs)
      (+ (car coefs) (* x (polynomial (- n 1) x (cdr coefs))))))

; Question: which of these will correctly translate to a formula?

(polynomial 2 2 (list 1 2 3))
(polynomial 2 x (list 1 2 3))
(polynomial 2 x (list a b c))
(polynomial n x (list a b c))  ; not handled correctly 