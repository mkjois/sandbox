#lang racket

(require rosette/config/config rosette/config/log
         rosette/base/value rosette/base/bool rosette/base/num
         (rename-in rosette/base/phi [phi ite])
         rosette/solver/solution rosette/kodkod/kodkod
         (only-in rosette/lang/define define-symbolic))

; Rosette's verbosity is controlled through log handlers, which
; determine which messages to print to the REPL console.  Each message
; has one of the following levels:  fatal, error, warning, info, and 
; debug.  The default logger only prints fatal, error and warning 
; messages.  To see info messages as well, customize the current log
; handler as follows:
(current-log-handler 
 (log-handler #:name 'sym-demo-handler   ; handler name (optional)
              #:info any/c))             ; handle any message with priorty level 'info or higher
  

; (define-symbolic x sym/number?) declares a bitvector variable x.
;
; All bitvector variables have the same length (number of bits). 
; Bitvector length can be configured using (configure [bitwidth n]), where
; n is between 2 and 32, inclusive.  You can print the current value 
; of all configuration parameters using (print-configuration).
;
; The following operators are currently defined on bitvectors:
; ite, sym/=, sym/<, sym/<=, sym/>, sym/>=, sym/+, sym/-, sym/*, sym//, 
; and sym/expt.  The meaning of each is analogous to that of its
; Racket counterpart.
(configure [bitwidth 8])
(define-symbolic x sym/number?)
(define-symbolic y sym/number?)
(define-symbolic z sym/number?)

;(define-symbolic a sym/boolean?) declares a boolean variable x.
; Operators defined on booleans include: ite, sym/false?, sym/eq?, 
; =>, <=>, !, &&, ||.  The first two correspond to their Racket 
; counterparts, and the remaining ones stand for implication, 
; bi-implication, negation, conjunction and disjunction, respectively.
(define-symbolic a sym/boolean?)
(define-symbolic b sym/boolean?)

; The following creates a new instance of the kodkod solver, adds three satisfiable 
; constraints to it, solves them, and prints the solution.  The expression (send solver assert ...)
; is Racket syntax for invoking the method "assert" on the solver object with the given arguments.
; For a detailed description of the solver API, please see rosette/solver/solver.rkt.
(define solver (new kodkod%))

(send solver assert (sym/= x (ite a y z))) 
(send solver assert (sym/< x z))
(send solver assert (sym/> z (sym/+ y 4)))

; * (sol var), where sol is a solution struct (see rosette/solver/solution.rkt), 
;   retrieves the value of  the variable var in the given solution.  Variables that are not 
;   constrained evaluate to themselves, since their value is irrelevant to satisfiability.  
;   In our example, (sol b) is b itself.
; * (sym-name var) retrieves the full name of the symbolic variable var, including its source location.
; * (sleep 1) suspends the current thread for 1 second, so that the log handler gets a chance to 
;   output all of its messages before we print the solution (optional)
(let ([sol (send solver solve)]) ; get a solution
  (sleep 1) 
  (unless (sat? sol)
    (error 'sym-demo "Expected a solution but none found!"))
  (printf "\nSOLUTION:\n")
  (for ([var (list x y z a b)])
    (printf "~s := ~s\n" (sym-name var) (sol var))))

(newline)

; The next example clears the solver's state (i.e., it removes the constraints added above),
; adds four constraints that are unsatisfiable, debugs them, and prints a minimal unsatisfiable
; core.  For fun, we'll also decorate the constraints added to the solver with information about
; where they originated, to get a more meaningful core.  Symbolic values, as determined by the 
; sym? predicate, are decorated using (sym-track-origin value origin), where the origin of a constraint
; can be anything that is meaningful to the application.  In our case, we use a constraint's position 
; (syntax) in the source file as its origin.
(define-syntax (track stx)
  (syntax-case stx ()
    [(_ constraint) #`(let ([val constraint])
                        (if (sym? val)
                            (sym-track-origin constraint #'constraint)
                            val))]))

(send solver clear)

(send solver assert (track (sym/= x (ite (&& a b) y z)))) 
(send solver assert (track (sym/< x z)))
(send solver assert (track (sym/> z (sym/+ y 8))))
(send solver assert (track (|| (! a) (! b))))

; * (core sol) retrieves the list of assertions that comprise a minimial unsat core of the problem.
; * (sym-origin val) retrieves the origin, if any, of a given symbolic value.
(let ([sol (send solver debug)]) ; get a core
  (sleep 1) 
  (unless (unsat? sol)
    (error 'sym-demo "Expected no solution but found one!"))
  (printf "\nMINIMAL UNSATISFIABLE CORE:\n")
  (for ([assertion (core sol)])   
    (printf "~a\n" (sym-origin assertion))))

; Use (send solver shutdown) to kill a solver process, if it's no longer need or 
; if it is taking  too long to find a solution.  Every solver process is 
; automatically terminated at the end of the REPL session in which it is running.
(send solver shutdown)
