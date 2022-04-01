use std::{
    env::args,
    process::exit,
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
    Buffer(Buffer),
}

impl Node {
    fn eval(&self) -> Buffer {
        match self {
            Node::Buffer(buf) => buf.clone(),
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

struct TokenStream {
    stream: Vec<String>,
}

impl<'a> TokenStream {
    fn next(&mut self) -> String {
        self.stream.pop().unwrap()
    }

    fn peek(&'a self) -> &'a String {
        &self.stream.last().unwrap()
    }

    fn is_exhausted(&self) -> bool {
        self.stream.len() == 0
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
}

// TODO: use reversed stream?
fn parse(token_stream: &mut TokenStream) -> Node {
    let token = token_stream.next();
    if token == "(" {
        let mut left_operand = parse(token_stream);
        while token_stream.peek() != ")" {
            let (operator, dimensions) = token_stream.parse_operator(None);
            let second_argument = parse(token_stream);
            left_operand = Node::BinaryOperator {
                operator,
                second_argument: Box::new(second_argument),
                first_argument: Box::new(left_operand),
                dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
            };
        }
        token_stream.next(); // ")"
        let (operator, dimensions) = token_stream.parse_operator(None);
        let right_operand = parse(token_stream);
        Node::BinaryOperator {
            operator,
            first_argument: Box::new(left_operand),
            second_argument: Box::new(right_operand),
            dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
        }
    } else if token == "7" || token == "0.4" || token == "0.5" || token == "x" || token == "y" { // operand
        // TODO: write not poop
        Node::Buffer(Buffer::new(BufferType::Float, match token.as_str() {
            "7" => vec![1],//vec![7.0],
            "0.4" => vec![1],//vec![0.4],
            "0.5" => vec![1],//vec![0.5],
            "x" => vec![],
            "y" => vec![],
            _ => unimplemented!(),
        }).name(token.clone()))
    } else if token == "*" || token == "+" || token == "fract" || token == "<" || token == "-" || token == "abs" || token == "max" { // unary operator
        let (operator, dimensions) = token_stream.parse_operator(Some(token));
        let operand = parse(token_stream);
        Node::UnaryOperator {
            operator,
            argument: Box::new(operand),
            dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
        }
    } else if token == "stack" { // variadic operator
        let (operator, dimensions) = token_stream.parse_operator(Some(token));
        let mut operands = Vec::new();
        while !token_stream.is_exhausted() && token_stream.peek() != ")" {
            operands.push(Box::new(parse(token_stream)));
        }
        Node::VariadicOperator {
            operator,
            arguments: operands.into_iter().collect(),
            dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
        }
    } else if token == "[" {
        let mut array_items = Vec::new();
        while token_stream.peek() != "]" {
            array_items.push(parse(token_stream));
        }
        token_stream.next(); // "]"
        //array_items
        // TODO: insert parsed items into buffer
        Node::Buffer(Buffer::new(BufferType::Float, vec![]))
    } else {
        unimplemented!("wtf is '{}' you want me to parse?!, left to parse: {:?}", token, token_stream.stream.iter().rev().collect::<Vec<&String>>())
    }
}

fn main() {
    let argv = args().collect::<Vec<String>>();
    if argv.len() != 2 {
        eprintln!("Usage: {} EXPRESSION", argv[0]);
        exit(1);
    }
    let expression = &argv[1];
    // TODO: compile regex compile-time
    lazy_static! {
        static ref RE: Regex = Regex::new(r#"\s*(#|max|stack|abs|-|<|fract|\+|\*|[a-zA-Z0-9.]+|[()\[\]])\s*"#).unwrap();
    }
    let mut token_stream = TokenStream {
        stream: RE
            .captures_iter(expression)
            .map(|capture| capture[1].to_string())
            .collect::<Vec<String>>()
            .into_iter()
            .rev()
            .collect::<Vec<String>>(),
    };
    println!("{}", parse(&mut token_stream).to_string());
}

#[cfg(test)]
mod tests {
    use super::{Node, Buffer, BufferType};

    #[test]
    fn expression_to_string() {
        let x_buffer = Node::Buffer(Buffer::new(BufferType::Float, vec![]/*x.shape*/).name("x".to_owned()));
        assert_eq!(
            x_buffer.to_string(),
            "x".to_owned(),
        );
        let y_buffer = Node::Buffer(Buffer::new(BufferType::Float, vec![]/*y.shape*/).name("y".to_owned()));
        assert_eq!(
            y_buffer.to_string(),
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
                                arguments: vec![Box::new(x_buffer), Box::new(y_buffer)],
                                dimensions: Some(2),
                            }),
                            second_argument: Box::new(Node::Buffer(Buffer::new(BufferType::Float, vec![2]).name("[0.5 0.5]".to_owned()))),
                            dimensions: None,
                        }),
                        dimensions: None,
                    }),
                    dimensions: Some(2),
                }),
                second_argument: Box::new(Node::Buffer(Buffer::new(BufferType::Float, vec![1]).name("0.4".to_owned()))),
                dimensions: None,
            }),
            second_argument: Box::new(Node::UnaryOperator {
                operator: "fract".to_owned(),
                argument: Box::new(Node::BinaryOperator {
                    operator: "*".to_owned(),
                    first_argument: Box::new(Node::BinaryOperator {
                        operator: "+".to_owned(),
                        first_argument: Box::new(x_buffer),
                        second_argument: Box::new(y_buffer),
                        dimensions: None,
                    }),
                    second_argument: Box::new(Node::Buffer(Buffer::new(BufferType::Float, vec![1]).name("7".to_owned()))),
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

