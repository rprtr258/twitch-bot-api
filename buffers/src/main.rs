use std::{
    env::args,
    process::exit,
    rc::Rc,
};
use {
    regex::Regex,
    lazy_static::lazy_static,
};

type Idx = usize;

enum BufferType {
    Float,
    U8,
    Bool,
    Idx,
}

#[derive(Clone, Debug)]
enum BufferData {
    Float(Vec<f32>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
    Idx(Vec<Idx>),
}

#[derive(Clone, Debug)]
struct Buffer {
    shape: Vec<u8>,
    name: String,
    data: BufferData,
}

impl Buffer {
    fn new(typee: BufferType, shape: Vec<u8>) -> Self {
        Buffer {
            shape,
            name: "_tmp".to_owned(),
            data: match typee {
                BufferType::Float => BufferData::Float(Vec::new()),
                BufferType::U8 => BufferData::U8(Vec::new()),
                BufferType::Bool => BufferData::Bool(Vec::new()),
                BufferType::Idx => BufferData::Idx(Vec::new()),
            }
        }
    }
    fn name(mut self, name: String) -> Self {
        self.name = name;
        return self
    }
}

type OperatorDimensions = Option<Idx>;

#[derive(Debug)]
enum Node {
    UnaryOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        argument: Box<Node>,
    },
    BinaryOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        first_argument: Box<Node>,
        second_argument: Box<Node>,
    },
    VariadicOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        arguments: Vec<Box<Node>>,
    },
    Buffer(Rc<Buffer>),
}

impl Node {
    fn eval(&self) -> Buffer {
        match self {
            Node::Buffer(buf) => (**buf).clone(),
            Node::UnaryOperator {operator, ..} => {
                unimplemented!("Unary operator '{}' is not implemented", operator);
            },
            Node::BinaryOperator {operator, ..} => {
                unimplemented!("Binary operator '{}' is not implemented", operator);
            },
            Node::VariadicOperator {operator, ..} => {
                unimplemented!("Variadic operator '{}' is not implemented", operator);
            },
        }
    }

    // TODO: move to Debug implementation
    fn to_string(&self) -> String {
        match self {
            Node::Buffer(buf) => {
                // shape_str = ";".join(map(str, self.shape))
                // type_str = self.type.name
                // return f"[{self.name}={type_str}[{shape_str}]]"
                buf.name.clone()
            },
            Node::UnaryOperator {operator, dimensions, argument} => {
                let operator_str = match dimensions {
                    Some(dims) => format!("{}#{:?}", operator, dims),
                    None => operator.to_string(),
                };
                format!("{} {}", operator_str, argument.to_string())
            },
            Node::BinaryOperator {operator, dimensions, first_argument, second_argument} => {
                let operator_str = match dimensions {
                    Some(dims) => format!("{}#{:?}", operator, dims),
                    None => operator.to_string(),
                };
                format!("({}) {} {}", first_argument.to_string(), operator_str, second_argument.to_string())
            },
            Node::VariadicOperator {operator, dimensions, arguments} => {
                let operator_str = match dimensions {
                    Some(dims) => format!("{}#{:?}", operator, dims),
                    None => operator.to_string(),
                };
                format!("{} {}", operator_str, arguments.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "))
            },
        }
    }
}

struct TokenStream(Vec<String>);

impl<'a> TokenStream {
    fn next(&mut self) -> String {
        self.0.pop().unwrap()
    }

    fn peek(&'a self) -> &'a String {
        &self.0.last().unwrap()
    }

    fn is_exhausted(&self) -> bool {
        self.0.len() == 0
    }

    fn parse_operator(&mut self, token: Option<String>) -> (String, Option<String>) {
        let operator = token.unwrap_or_else(|| self.next());
        let dimensions = if (operator == "max" || operator == "stack") && self.peek() == "#" {
            self.next(); // "#"
            // TODO: multidimensinal operators
            //if token_stream.peek() == "[":
            //    return str(parse(token_stream))
            Some(self.next())
        } else {
            None
        };
        (operator, dimensions)
    }

    // TODO: use reversed stream?
    fn parse(&mut self) -> Node {
        let token = self.next();
        if token == "(" {
            let mut left_operand = self.parse();
            while self.peek() != ")" {
                let (operator, dimensions) = self.parse_operator(None);
                let second_argument = self.parse();
                left_operand = Node::BinaryOperator {
                    operator,
                    second_argument: Box::new(second_argument),
                    first_argument: Box::new(left_operand),
                    dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
                };
            }
            self.next(); // ")"
            let (operator, dimensions) = self.parse_operator(None);
            let right_operand = self.parse();
            Node::BinaryOperator {
                operator,
                first_argument: Box::new(left_operand),
                second_argument: Box::new(right_operand),
                dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
            }
        } else if token == "7" || token == "0.4" || token == "0.5" || token == "x" || token == "y" { // operand
            // TODO: write not poop
            Node::Buffer(Rc::new(Buffer::new(BufferType::Float, match token.as_str() {
                "7" => vec![1],//vec![7.0],
                "0.4" => vec![1],//vec![0.4],
                "0.5" => vec![1],//vec![0.5],
                "x" => vec![],
                "y" => vec![],
                _ => unimplemented!(),
            }).name(token.clone())))
        } else if token == "*" || token == "+" || token == "fract" || token == "<" || token == "-" || token == "abs" || token == "max" { // unary operator
            let (operator, dimensions) = self.parse_operator(Some(token));
            let operand = self.parse();
            Node::UnaryOperator {
                operator,
                argument: Box::new(operand),
                dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
            }
        } else if token == "stack" { // variadic operator
            let (operator, dimensions) = self.parse_operator(Some(token));
            let mut operands = Vec::new();
            while !self.is_exhausted() && self.peek() != ")" {
                operands.push(Box::new(self.parse()));
            }
            Node::VariadicOperator {
                operator,
                arguments: operands.into_iter().collect(),
                dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
            }
        } else if token == "[" {
            let mut array_items = Vec::new();
            while self.peek() != "]" {
                array_items.push(self.parse());
            }
            self.next(); // "]"
            //array_items
            // TODO: insert parsed items into buffer
            Node::Buffer(Rc::new(Buffer::new(BufferType::Float, vec![])))
        } else {
            unimplemented!("wtf is '{}' you want me to parse?!, left to parse: {:?}", token, self.0.iter().rev().collect::<Vec<&String>>())
        }
    }
}

impl FromIterator<String> for TokenStream {
    fn from_iter<I>(iter: I) -> Self where I: std::iter::IntoIterator<Item=String> {
        TokenStream(iter.into_iter().collect::<Vec<String>>())
    }
}

impl std::str::FromStr for Node {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO: compile regex compile-time
        lazy_static! {
            static ref RE: Regex = Regex::new(r#"\s*(#|max|stack|abs|-|<|fract|\+|\*|[a-zA-Z0-9.]+|[()\[\]])\s*"#).unwrap();
        }
        Ok(RE
            .captures_iter(s)
            .map(|capture| capture[1].to_string())
            .collect::<Vec<String>>()
            .into_iter()
            .rev()
            .collect::<TokenStream>()
            .parse())
    }
}

fn main() {
    let argv = args().collect::<Vec<String>>();
    if argv.len() != 2 {
        eprintln!("Usage: {} EXPRESSION", argv[0]);
        exit(1);
    }
    let expression = &argv[1];
    let ast = expression.parse::<Node>().unwrap();
    println!("{}", ast.to_string());
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use super::{Node, Buffer, BufferType};

    #[test]
    fn string_to_expression() {
        assert_eq!(
            "((max#2 abs (stack#2 x y) - [0.5 0.5]) < 0.4) * fract (x + y) * 7".parse::<Node>().unwrap().to_string(),
            "((max#2 abs (stack#2 x y) - _tmp) < 0.4) * fract ((x) + y) * 7",
        );
    }

    #[test]
    fn expression_to_string() {
        let x_buffer = Rc::new(Buffer::new(BufferType::Float, vec![]/*x.shape*/).name("x".to_owned()));
        assert_eq!(
            Node::Buffer(x_buffer.clone()).to_string(),
            "x".to_owned(),
        );
        let y_buffer = Rc::new(Buffer::new(BufferType::Float, vec![]/*y.shape*/).name("y".to_owned()));
        assert_eq!(
            Node::Buffer(y_buffer.clone()).to_string(),
            "y".to_owned(),
        );
        let test_expr = Node::BinaryOperator {
            operator: "*".to_owned(),
            first_argument: Box::new(Node::BinaryOperator {
                operator: "<".to_owned(),
                first_argument: Box::new(Node::UnaryOperator {
                    operator: "max".to_owned(),
                    argument: Box::new(Node::UnaryOperator {
                        operator: "abs".to_owned(),
                        argument: Box::new(Node::BinaryOperator {
                            operator: "-".to_owned(),
                            first_argument: Box::new(Node::VariadicOperator {
                                operator: "stack".to_owned(),
                                arguments: vec![Box::new(Node::Buffer(x_buffer.clone())), Box::new(Node::Buffer(y_buffer.clone()))],
                                dimensions: Some(2),
                            }),
                            second_argument: Box::new(Node::Buffer(Rc::new(Buffer::new(BufferType::Float, vec![2]).name("[0.5 0.5]".to_owned())))),
                            dimensions: None,
                        }),
                        dimensions: None,
                    }),
                    dimensions: Some(2),
                }),
                second_argument: Box::new(Node::Buffer(Rc::new(Buffer::new(BufferType::Float, vec![1]).name("0.4".to_owned())))),
                dimensions: None,
            }),
            second_argument: Box::new(Node::UnaryOperator {
                operator: "fract".to_owned(),
                argument: Box::new(Node::BinaryOperator {
                    operator: "*".to_owned(),
                    first_argument: Box::new(Node::BinaryOperator {
                        operator: "+".to_owned(),
                        first_argument: Box::new(Node::Buffer(x_buffer.clone())),
                        second_argument: Box::new(Node::Buffer(y_buffer.clone())),
                        dimensions: None,
                    }),
                    second_argument: Box::new(Node::Buffer(Rc::new(Buffer::new(BufferType::Float, vec![1]).name("7".to_owned())))),
                    dimensions: None,
                }),
                dimensions: None,
            }),
            dimensions: None,
        };
        assert_eq!(
            test_expr.to_string(),
            "((max#2 abs (stack#2 x y) - [0.5 0.5]) < 0.4) * fract ((x) + y) * 7".to_owned(),
        );
    }
}

