#![feature(rand,collections)]

use std::str;
use std::rand;
use OptionalInt::Value;
use OptionalInt::Missing;

fn main() {

  // more strings
  let x: &[u8] = &[b'B', b'e', b'a', b's', b't', b' ', b'm', b'o', b'd', b'e'];
  let stack_str: &str = str::from_utf8(x).unwrap();
  println!("stack_str is '{}'", stack_str);
  println!("stack_str has length {}", strlen(stack_str));
  // '.' has higher precedence than '&'
  println!("stack_str has length {}", strlen(&stack_str.to_string()));

  for c in stack_str.graphemes(true) {
    println!("{}", c); // c: &str
  }

  // more patterns
  let x = rand::random::<u32>() % 10;
  match x {
    0 ... 4 => println!("x is less than 5"),
    n @ 5 | n @ 6 | n @ 7 => println!("x is {}", n),
    _ => println!("x is greater than 7"),
  }
  let x = Value(x);
  let y = Missing;
  match x {
    Value(i) if i > 4 => println!("i32 greater than 4"),
    Value(..) => println!("not missing, but ignored type"),
    Missing => println!("i32 missing"),
  }
  match y {
    Value(i) if i > 4 => println!("i32 greater than 4"),
    Value(..) => println!("not missing, but ignored type"),
    Missing => println!("i32 missing"),
  }
  let x = &5;
  match x {
    &val => println!("Got a value: {}", val),
  }
  let x = 5;
  match x {
    ref r => println!("Got a reference to {}", *r),
  }
  let v = ["match_this", "2"]; // could also be a vec![...]
  match &v[] {
    ["match_this", second] => println!("The second element is {}", second),
    _ => {},
  }

  // methods and chaining
  let mut r = Rect::new(3, 4); // must be mutable to call grow()
  println!("area of r is {}", r.area());
  println!("{} after guessing area", r.is_area(12));
  println!("area of r is {} after growing", r.grow(7).grow(2).area());
}

struct Rect {
  l: u32,
  w: u32,
}

impl Rect {
  // static method
  fn new(l: u32, w: u32) -> Rect {
    Rect{l: l, w: w}
  }

  fn area(&self) -> u32 {
    return self.l * self.w;
  }

  fn is_area(&self, a: u32) -> &str {
    if self.area() == a {
      "correct area"
    } else {
      "incorrect area"
    }
  }

  // returns &mut Rect for method chaining
  fn grow(&mut self, factor: u32) -> &mut Rect {
    self.l *= factor;
    self.w *= factor;
    return self;
  }
}

enum OptionalInt {
  Value(u32),
  Missing,
}

fn strlen(s: &str) -> usize {
  return s.len();
}
