#lang racket

; problems illustrated:
;    create assertion to detect insufficient unrolling

(require rosette/lang/control)
         
(require rosette/config/config rosette/config/log 
         rosette/base/value rosette/base/bool rosette/base/num
         (rename-in rosette/base/phi [phi ite])
         rosette/solver/solution rosette/kodkod/kodkod
         (only-in rosette/lang/define define-symbolic))

(current-log-handler 
 (log-handler #:name 'sym-demo-handler   ; handler name (optional)
              #:info any/c))             ; handle any message with priorty level 'info or higher

(require (only-in racket (+ orig+)(- orig-)(= orig=)(if orig-if)))

(define (+ x y) (sym/+ x y))
(define (- x y) (sym/- x y))
(define (= x y) (sym/= x y))
(define-syntax-rule (if e t f) (sym/if e t f))

(define (fib x) 
  (case x 
    [(0 1) 1] 
    [else (+ (fib (- x 1)) (fib (- x 2)))]
  ))
  
(define (fib-with-if x) ; fib should also work with our redefined if
  (if (= x 0) 1
      (if (= x 1) 1
          (+ (fib (- x 1)) (fib (- x 2))))))
  
(define (unwind-safe-fib2 x limit)   ; ideally, the limit should be ignored when x is concrete 
  (orig-if (= limit 0)
      -100  ; TODO: assertion comes here
      ; not safe to use (case x) here, as case has not been lifted
      (if (= x 0) 1
          (if (= x 1) 1
              (+ (unwind-safe-fib2 (- x 1) (sub1 limit)) 
                 (unwind-safe-fib2 (- x 2) (sub1 limit)))  ; OK to use sub1 since k will always be concrete
              ))))

(configure [bitwidth 16])
(define-symbolic x sym/number?)
(define-symbolic y sym/number?)
(define-symbolic z sym/number?)

(define formula1 (= (+ x y) 1))
(define formula2 (= 144 (unwind-safe-fib2 z 5)))

formula2

(define solver (new kodkod%))

(send solver assert formula2) 

(let ([sol (send solver solve)]) ; get a solution
  (sleep 1) 
  (unless (sat? sol)
    (error 'sym-demo "Expected a solution but none found!"))
  (printf "\nSOLUTION:\n")
  (for ([var (list x y z)])
    (printf "~s := ~s\n" (sym-name var) (sol var))))
