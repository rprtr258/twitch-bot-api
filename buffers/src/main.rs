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
enum Node<'a> {
    UnaryOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        argument: &'a Node<'a>,
    },
    BinaryOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        first_argument: &'a Node<'a>,
        second_argument: &'a Node<'a>,
    },
    VariadicOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        arguments: Vec<&'a Node<'a>>,
    },
    Buffer(Buffer),
}

impl Node<'_> {
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

fn main() {
    let argv = args().collect::<Vec<String>>();
    if argv.len() != 2 {
        eprintln!("Usage: {} EXPRESSION", argv[0]);
        exit(1);
    }
    let expression = &argv[1];
    // TODO: compile regex compile-time
    lazy_static! {
        static ref RE: Regex = Regex::new(r#"\s*([a-zA-Z0-9.]*|[()])\s*"#).unwrap();
    }
    let tokens = RE.captures_iter(expression)
        .map(|capture| capture[1].to_string())
        .collect::<Vec<String>>();
    println!("{:?}", tokens);
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
            first_argument: &Node::BinaryOperator {
                operator: "<".to_owned(),
                first_argument: &Node::UnaryOperator {
                    operator: "max".to_owned(),
                    argument: &Node::UnaryOperator {
                        operator: "abs".to_owned(),
                        argument: &Node::BinaryOperator {
                            operator: "-".to_owned(),
                            first_argument: &Node::VariadicOperator {
                                operator: "stack".to_owned(),
                                arguments: vec![&x_buffer, &y_buffer],
                                dimensions: Some(2),
                            },
                            second_argument: &Node::Buffer(Buffer::new(BufferType::Float, vec![2]).name("[0.5 0.5]".to_owned())),
                            dimensions: None,
                        },
                        dimensions: None,
                    },
                    dimensions: Some(2),
                },
                second_argument: &Node::Buffer(Buffer::new(BufferType::Float, vec![1]).name("0.4".to_owned())),
                dimensions: None,
            },
            second_argument: &Node::UnaryOperator {
                operator: "fract".to_owned(),
                argument: &Node::BinaryOperator {
                    operator: "*".to_owned(),
                    first_argument: &Node::BinaryOperator {
                        operator: "+".to_owned(),
                        first_argument: &x_buffer,
                        second_argument: &y_buffer,
                        dimensions: None,
                    },
                    second_argument: &Node::Buffer(Buffer::new(BufferType::Float, vec![1]).name("7".to_owned())),
                    dimensions: None,
                },
                dimensions: None,
            },
            dimensions: None,
        };
        assert_eq!(
            test_expr.to_string(),
            "((max#2 abs (stack#2 x y) - [0.5 0.5]) < 0.4) * fract ((x) + y) * 7".to_owned(),
        );
    }
}

