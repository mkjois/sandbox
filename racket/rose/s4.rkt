#lang racket

(require macro-debugger/stepper)

; problems illustrated:
;    how to unwind function calls
(require (rename-in racket (+ orig+)(- orig-)(= orig=)(if orig-if)))

(define (+ x y)
  (if (and (number? x) (number? y))
      (orig+ x y)
      `(+ ,x ,y)))

(define (- x y)
  (if (and (number? x) (number? y))
      (orig- x y)
      `(- ,x ,y)))

(define (= x y)
  (if (and (number? x) (number? y))
      (orig= x y)
      `(= ,x ,y)))

; bad version
;(define-syntax-rule (if e t f)  ; Quiz: can this macro be a call?
;  `(ite ,e ,t ,f))              ; is partial evaluation based on e necessary?

(define-syntax-rule (if e t f)  ; Quiz: can this macro be a call?
  (orig-if (boolean? e)         ; Quiz: should we add a conjunct that e is true on rhs?
           (orig-if e t f)      ; Quiz: is partial evaluation based on e necessary? Yes because in def + we use own if not the orig if
           `(ite ,e ,t ,f)))    ; Quiz: influence of target formula language F.  How would we would have to translate if F contained no (ite)?

(if #t 1 2)
(if #f 1 2)
(if (= 1 1) 1 2)
(if (= 1 2) 1 2)
(if (= 1 'x) 1 2)

(define (fib x) 
  (case x 
    [(0 1) 1] 
    [else (+ (fib (- x 1)) (fib (- x 2)))]
  ))
  
(define (fib-with-if x) ; fib should also work with our redefined if
  (if (= x 0) 1
      (if (= x 1) 1
          (+ (fib (- x 1)) (fib (- x 2))))))
  

; recursion unwinding
(define (unwind-safe-fib x limit)   
  (define-syntax-rule (check-limit body) ; try this with a function first, then with a macro
    (if (> limit 0)
        body
        0 ; TODO an assertion should come here
        ))

  (check-limit 
   (case x 
     [(0 1) 1] 
     [else (+ (unwind-safe-fib (- x 1)(sub1 limit)) (unwind-safe-fib (- x 2) (sub1 limit)))]  ; OK to use sub1 since k will always be concrete
     )))

(define (unwind-safe-fib2 x limit)   ; ideally, the limit should be ignored when x is concrete 
  (orig-if (= limit 0)
      -100
      ; not safe to use (case x) here, as case has not been lifted
      (if (= x 0) 1
          (if (= x 1) 1
              (+ (unwind-safe-fib2 (- x 1) (sub1 limit)) 
                 (unwind-safe-fib2 (- x 2) (sub1 limit)))  ; OK to use sub1 since k will always be concrete
              ))))

; fully evaluated
(+ 1 (fib 6))
(+ 1 (fib-with-if 6))

; show that we don't need to be always symbolic
(+ 'x (fib 5))
(+ 'x (fib-with-if 5))

; this does not terminate

(unwind-safe-fib2 4 5)  ; play with a range of values for limit
