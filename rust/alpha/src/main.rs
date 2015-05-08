#![feature(collections,io)]

use std::cmp::Ordering;
use Validation::Pass;
use Validation::Fail;
use std::old_io;

fn main() {

  // variables, pattern matching on assignment
  let (mut x, y): (i32, i32) = (5, 6);
  if x > 3 {
    x = 12
  } else {
    x = 2
  }
  println!("Changing x to {}", x);
  println!("x is {}, y is {}, x + y is {}", x, y, x+y);

  // keyword-driven expressions evaluate to a value
  let z = if x < 7 { 4+8*7 } else {2-9*3};
  println!("z is {}, xyz is {}", z, x*y*z);

  // functions
  print_number(24);
  let result = sum_of_squares(x, y);
  println!("{}^2 + {}^2 = {}", x, y, result);
  let a = gcd(x, y);
  let b = gcd(135, 54);
  println!("gcd({}, {}) is {}", x, y, a);
  println!("gcd(135, 54) is {}", b);

  // Markdown documentation
  doakes("Surprise");
  doakes("Large fries");
  doakes("Blue eyes");

  // tuples, pattern matching with them, and using them in functions
  let mut u: (i32, &str) = (7, "machine");
  let v: (i32, &str) = u;
  u = (8, "learning");
  let (u1, u2) = u;
  println!("u is ({}, {})", u1, u2);
  let (v1, v2) = v;
  println!("v is ({}, {})", v1, v2);
  let (a, b, c, d) = powers(5);
  println!("{}, {}, {}, {}", a, b, c, d);

  // structs
  let junseok = Person {name: "Junseok Lee".to_string(), age: 21, ht_wt: (73, 160)};
  print_person(junseok);

  // tuple-structs
  let boil = Kelvin(373);
  let freeze = Kelvin(273);
  let Kelvin(boil_degrees) = boil;
  let Kelvin(mut freeze_degrees) = freeze;

  // basic enums, importing from libraries
  let order = cmp(boil_degrees, freeze_degrees);
  freeze_degrees -= 273;
  println!("Water boiling point is {} degrees Kelvin", boil_degrees);
  println!("Water freezing point is {} degrees Celsius", freeze_degrees);

  // 'match' keyword with primitives and basic enums
  match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    _ => println!("not one, two or three"),
  }
  match order {
    Ordering::Less => println!("Freezing point is higher."),
    Ordering::Greater => println!("Boiling point is higher."),
    Ordering::Equal => println!("Freezing and boiling points are equal."),
  }

  // matching with valued enums and as an expression
  let mut v: Validation = login("lame_memes");
  match v {
    Pass(message) => println!("{}", message),
    Fail(message) => println!("{}", message),
  }
  v = login("dank_memes");
  println!("{}", match v {
    Pass(message) => message,
    Fail(message) => message,
  });

  // for loop
  for x in 0..5 {
    println!("{}! = {}", x, factorial(x));
  }

  // finite while loop
  let mut x = 5;
  let mut done = false;
  while !done {
    x += x-3;
    println!("{}", x);
    if x % 5 == 0 { done = true; }
  }

  // infinite loop with 'break' and 'continue' keywords
  loop {
    if x == 0 { break; }
    x /= 2;
    println!("{}", x);
    if x != 1 { continue; }
    println!("not continuing");
  }

  // strings
  let mut s: String = "Beast".to_string();
  s.push_str(" mode!");
  print_slice(&*s); // String is a *reference* to the dynamically allocated string

  // arrays
  let a: [usize; 3] = [1, 2, 3]; // a: [usize; 3]
  let mut b: [i32; 5] = [0; 5]; // mut b: [i32; 5], 0 is the default value
  let mut c = 0;
  for &i in a.iter() { // iterating over an array returns a *reference* to the next elem
    c += b.len();
    b[i] = (i*i) as i32; // casting
    println!("Element {} of b is {}", i, b[i]);
  }
  println!("|a| times |b| is {}", c);

  // vectors, slices
  let mut v = vec![4, 5, 6, 7]; // v: Vec<i32>
  v.push(8);
  // can iterate, get size, and index into just like an array
  let mid = &v[1..4];
  for e in mid.iter() {
    println!("mid slice of v: {}", e);
  }
  println!("length of mid is {}", mid.len());

  // slicing into strings
  println!("length of 'Beast mode!' is {}", s.len());
  let ss: &str = &s[2..9];
  println!("String ss is {}", ss);
  let sss: &str = &&s[1..5];
  println!("&str sss is {}", sss);

  // standard input
  println!("Fight me, bitch...");
  let input = old_io::stdin().read_line().ok().expect("Failed to read line.");
  println!("Your response was: {}", input);
}

fn print_slice(s: &str) {
  println!("{}", s);
}

fn factorial(n: i32) -> i32 {
  if n < 2 { 1 }
  else { n * factorial(n-1) }
}

enum Validation {
  Pass(String),
  Fail(String),
}

fn login(password: &str) -> Validation {
  if password == "dank_memes" {
    Pass("Login successful!".to_string())
    //Validation::Pass("Login successful!".to_string())
  } else {
    Fail("Incorrect password.".to_string())
    //Validation::Fail("Incorrect password".to_string())
  }
}

fn cmp(a: i32, b: i32) -> Ordering {
  if a < b { Ordering::Less }
  else if a > b { Ordering::Greater }
  else { Ordering::Equal }
}

struct Kelvin(i32);

fn print_person(p: Person) {
  let (h, w) = p.ht_wt;
  println!("{}, age {}, is {}'{}\" tall and weighs {}lbs.", p.name, p.age, h / 12, h % 12, w);
}

struct Person {
  name: String,
  age: i32,
  ht_wt: (i32, i32),
}

fn powers(n: i32) -> (i32, i32, i32, i32) {
  let squared = n*n;
  let cubed = squared*n;
  let quart = cubed*n;
  (n, squared, cubed, quart)
}

/// Homage to James Doakes
/// Also playing with *doc comments*
/// ## Parameters
/// * `phrase` The phrase to start with. Ideally rhymes with *surprise*.
/// ## Usage
/// ```rust
/// let p = "Surprise";
/// doakes(p); // prints "Surprise, motherfucker!"
/// ```
fn doakes(phrase: &str) {
  println!("{}, motherfucker!", phrase);
}

fn gcd(x: i32, y: i32) -> i32 {
  if y > x {
    gcd(y, x)
  } else if y == 0 {
    x
  } else if y == 1 {
    1
  } else {
    gcd(y, x % y)
  }
}

fn sum_of_squares(x: i32, y: i32) -> i32 {
  let result = x*x + y*y;
  result
}

fn print_number(x: i32) {
  println!("The given argument is {}", x);
}
