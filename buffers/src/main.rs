use std::{
    env::args,
    io::Write,
    ops::Deref,
    process::exit,
    rc::Rc,
};
use {
    lazy_static::lazy_static,
    regex::Regex,
};

type Idx = usize;

// TODO: change to generics?
#[derive(Clone, Debug)]
enum BufferData {
    Float(Vec<f32>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
    Idx(Vec<Idx>),
}

impl BufferData {
    fn len(&self) -> usize {
        match self {
            BufferData::Float(ref dd) => dd.len(),
            BufferData::U8(ref dd) => dd.len(),
            BufferData::Bool(ref dd) => dd.len(),
            BufferData::Idx(ref dd) => dd.len(),
        }
    }
}

impl From<Vec<f32>>   for BufferData { fn from(data: Vec<f32>  ) -> Self {BufferData::Float(data)}}
impl From<Vec<u8>>    for BufferData { fn from(data: Vec<u8>   ) -> Self {BufferData::U8   (data)}}
impl From<Vec<bool>>  for BufferData { fn from(data: Vec<bool> ) -> Self {BufferData::Bool (data)}}
impl From<Vec<usize>> for BufferData { fn from(data: Vec<usize>) -> Self {BufferData::Idx  (data)}}

#[derive(Clone, Debug)]
struct Buffer {
    shape: Vec<u8>, // TODO: make more than u8, also change Idx
    name: Option<String>,
    data: BufferData,
}

enum BufferType {
    Float,
    U8,
    Bool,
    Idx,
}

impl Buffer {
    fn new(typee: BufferType, shape: Vec<u8>) -> Self {
        Buffer {
            shape,
            data: match typee {
                BufferType::Float => BufferData::Float(Vec::new()),
                BufferType::U8 => BufferData::U8(Vec::new()),
                BufferType::Bool => BufferData::Bool(Vec::new()),
                BufferType::Idx => BufferData::Idx(Vec::new()),
            },
            name: None,
        }
    }

    fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        return self
    }
}

fn stack_along(array: Vec<Buffer>, dim: usize) -> Buffer {
    {
        if dim > array.len() {
            panic!("stack dim must be >=0 and <=len(array), but it's {} when len(array)={}", dim, array.len());
        }
        let first_item = &array[0];
        if !array.iter().skip(1).all(|buf| match (&first_item.data, &buf.data) {
            (BufferData::Float(_), BufferData::Float(_)) => true,
            (BufferData::U8(_), BufferData::U8(_)) => true,
            (BufferData::Bool(_), BufferData::Bool(_)) => true,
            (BufferData::Idx(_), BufferData::Idx(_)) => true,
            _ => false,
        } && first_item.shape == buf.shape) {
            panic!("all items must have the same type and shape");
        }
    }
    let mut resulting_shape = Vec::with_capacity(array.len() + 1);
    let (mut shapes, datas): (Vec<Vec<u8>>, Vec<BufferData>) = array
        .into_iter()
        .map(|Buffer {shape, data, ..}| (shape, data))
        .unzip();
    let dim_along = shapes.len().try_into().unwrap();
    let stack_dim = shapes.remove(shapes.len() - 1);
    for x in stack_dim.iter().take(dim) {
        resulting_shape.push(*x);
    }
    resulting_shape.push(dim_along);
    for x in stack_dim.iter().skip(dim) {
        resulting_shape.push(*x);
    }
    // stack them along 0
    // TODO: stack not only along 0
    Buffer {
        shape: resulting_shape,
        data: match datas[0] {
            BufferData::Float(_) => {
                datas
                    .into_iter()
                    .flat_map(|elem| {
                        match elem {
                            BufferData::Float(elem_data) => elem_data,
                            _ => unreachable!(),
                        }
                    })
                    .collect::<Vec<f32>>()
                    .into()
            },
            BufferData::U8(_) => {
                datas
                    .into_iter()
                    .flat_map(|elem| {
                        match elem {
                            BufferData::U8(elem_data) => elem_data,
                            _ => unreachable!(),
                        }
                    })
                    .collect::<Vec<u8>>()
                    .into()
            },
            BufferData::Bool(_) => {
                datas
                    .into_iter()
                    .flat_map(|elem| {
                        match elem {
                            BufferData::Bool(elem_data) => elem_data,
                            _ => unreachable!(),
                        }
                    })
                    .collect::<Vec<bool>>()
                    .into()
            },
            BufferData::Idx(_) => {
                datas
                    .into_iter()
                    .flat_map(|elem| {
                        match elem {
                            BufferData::Idx(elem_data) => elem_data,
                            _ => unreachable!(),
                        }
                    })
                    .collect::<Vec<Idx>>()
                    .into()
            },
        },
        name: None,
    }
}

impl Deref for Buffer {
    type Target = BufferData;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

fn dump_buffer(buf: &Buffer, filename: &str) -> std::io::Result<()> {
    // TODO: reset file content
    let mut file = std::fs::File::options()
        .create(true)
        .write(true)
        .open(filename)?;
    // type
    file.write(match buf.data {
        BufferData::Float(_) => &[0x00],
        BufferData::U8(_) => &[0x01],
        BufferData::Bool(_) => &[0x02],
        BufferData::Idx(_) => &[0x03],
    })?;
    // shape length
    file.write(&[buf.shape.len().try_into().unwrap()])?;
    // shape
    file.write(&buf.shape.iter().map(|x| (*x).try_into().unwrap()).collect::<Vec<u8>>().as_slice())?;
    // data
    println!("{:?}", buf.data);
    file.write(&match &buf.data {
        BufferData::Float(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
        BufferData::U8(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
        BufferData::Bool(data) => data.iter().map(|x| if *x {0x01u8} else {0x00u8}).collect::<Vec<u8>>(),
        BufferData::Idx(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
    }.as_slice())?;
    Ok(())
}

type OperatorDimensions = Option<Idx>;

#[derive(Debug)]
enum BinaryOperatorType {
    Multiplication,
    // TODO: implement >
    // TODO: implement -
    // TODO: implement +
}

#[derive(Debug)]
enum Node {
    UnaryOperator {
        operator: String, // TODO; replace with enum
        dimensions: OperatorDimensions, // TODO: dimensions should be property of operator
        argument: Box<Node>,
    },
    BinaryOperator {
        operator: BinaryOperatorType,
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
                // TODO: implement abs
                // TODO: implement fract
                // TODO: implement max#2
                unimplemented!("Unary operator '{}' is not implemented", operator);
            },
            Node::BinaryOperator {operator, first_argument, second_argument, ..} => {
                match operator {
                    BinaryOperatorType::Multiplication => {
                        let first_buf = first_argument.eval();
                        let second_buf = second_argument.eval();
                        Buffer {
                            data: match (first_buf.data, second_buf.data) {
                                // * :: (float, float) -> float
                                (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x * y)
                                    .collect()
                                ),
                                // * :: (idx, idx) -> idx
                                (BufferData::Idx(fd), BufferData::Idx(sd)) => BufferData::Idx(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x * y)
                                    .collect()
                                ),
                                _ => unimplemented!(),
                            },
                            shape: first_buf.shape,
                            name: None,
                        }
                    },
                }
            },
            Node::VariadicOperator {operator, ..} => {
                // TODO: implement stack#2
                unimplemented!("Variadic operator '{}' is not implemented", operator);
            },
        }
    }

    // TODO: move to Debug implementation
    fn to_string(&self) -> String {
        match self {
            Node::Buffer(buf) => {
                // shape_str = ",".join(map(str, self.shape))
                // type_str = self.type.name
                // TODO: ({name}@)?{T}[{shape}]
                buf.name.as_ref().unwrap_or(&"no_name".to_owned()).clone()
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
                    Some(dims) => format!("{:?}#{:?}", operator, dims),
                    None => format!("{:?}", operator),
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

    fn parse_buffer(&mut self, token: Option<String>) -> Buffer {
        let token = token.unwrap_or_else(|| self.next());
        if token == "[" { // array
            let mut array = Vec::new();
            while self.peek() != "]" {
                array.push(self.parse_buffer(None));
            }
            self.next(); // "]"
            if array.len() == 0 {
                panic!("Buffer cannot be empty");
            }
            stack_along(array, 0)
        } else { // single item operand
            fn lookup_buffer(_: String) -> Result<Buffer, String> {
                // TODO: lookup known/cached buffers first
                Ok(Buffer::new(BufferType::Float, vec![]))
            }
            // TODO: regexes
            // TODO: linear/monadic interface
            token
                .parse::<bool>()
                .map(|x| Buffer {
                    data: BufferData::Bool(vec![x]),
                    shape: vec![],
                    name: None,
                })
                .map_err(|e| e.to_string())
                .or_else(|_| token
                    .parse::<Idx>()
                    .map(|x| Buffer {
                        data: BufferData::Idx(vec![x]),
                        shape: vec![],
                        name: None,
                    })
                    .map_err(|e| e.to_string())
                    .or_else(|_| token
                        .parse::<f32>()
                        .map(|x| Buffer {
                            data: BufferData::Float(vec![x]),
                            shape: vec![],
                            name: None,
                        })
                        .map_err(|e| e.to_string())
                        .or_else(|_|
                            if token.starts_with("0x") {
                                Ok(1)
                            } else {
                                Err("ZHOPA".to_string())
                            }
                            .map(|x| Buffer {
                                data: BufferData::U8(vec![x]),
                                shape: vec![],
                                name: None,
                            })
                            .map_err(|e| e.to_string())
                            .or_else(lookup_buffer)
                        )
                    )
                )
                .unwrap()
                .name(token.clone())
        }
    }

    // TODO: use reversed stream?
    fn parse(&mut self) -> Node {
        if self.peek() == "[" {
            return Node::Buffer(Rc::new(self.parse_buffer(None)))
        }
        let token = self.next();
        if token == "(" {
            let mut left_operand = self.parse();
            while self.peek() != ")" {
                let (operator, dimensions) = self.parse_operator(None);
                let binary_operator = match operator.as_str() {
                    "*" => BinaryOperatorType::Multiplication,
                    _ => unimplemented!("unknown binary operator: {}", operator),
                };
                let second_argument = self.parse();
                left_operand = Node::BinaryOperator {
                    operator: binary_operator,
                    second_argument: Box::new(second_argument),
                    first_argument: Box::new(left_operand),
                    dimensions: dimensions.map(|d| d.parse::<usize>().unwrap()),
                };
            }
            self.next(); // ")"
            left_operand
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
        } else {
            Node::Buffer(Rc::new(self.parse_buffer(Some(token))))
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
    // TODO: fix "2 * 2"
    let ast = expression.parse::<Node>().unwrap();
    println!("{}", ast.to_string());
    // TODO: check ast correctness
    let res = ast.eval();
    println!("{:?}", res);
    dump_buffer(&res, "test.buf").unwrap();
    // format!("{}.buf", buf.name)
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use super::{Node, Buffer, BufferType};

    #[test]
    fn string_to_expression() {
        assert_eq!(
            "((max#2 abs (stack#2 x y) - [0.5 0.5]) < 0.4) * fract (x + y) * 7".parse::<Node>().unwrap().to_string(),
            "((max#2 abs (stack#2 x y) - no_name) < 0.4) * fract ((x) + y) * 7",
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

