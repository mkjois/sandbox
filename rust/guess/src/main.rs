#![feature(io,rand)]

use std::old_io;
use std::rand;
use std::cmp::Ordering;

fn main() {
  let secret_number = rand::random::<u32>() % 100 + 1;
  //let secret_number: u32 = rand::random() % 100 + 1;
  let mut non_number = false;

  loop {
    println!("{}", if non_number { "Input a number you lil' bitch." } else { "Guess a number..." });
    let input: String = old_io::stdin().read_line()                     // Result<String, Error>
                                       .ok()                            // Option<String>
                                       .expect("Failed to read input"); // String
    let input_option: Option<u32> = input.trim()         // String
                                         .parse::<u32>() // Option<u32>
                                         .ok();          // Result<u32, Error>
    println!("Your guess was {}", input);

    let input_number = match input_option {
      Some(num) => num,
      None => {
        non_number = true;
        continue;
      }
    };

    non_number = false;

    match cmp(input_number, secret_number) {
      Ordering::Less => println!("Too small."),
      Ordering::Greater => println!("Too large."),
      Ordering::Equal => {
        println!("Perfect!");
        return;
      }
    }
  }

  //println!("The secret number was {}", secret_number);
}

fn cmp(a: u32, b: u32) -> Ordering {
  if a < b { Ordering::Less }
  else if a > b { Ordering::Greater }
  else { Ordering::Equal }
}
