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

type Idx = u16;

// TODO: change to generics?
#[derive(Clone, Debug, PartialEq)]
enum BufferData {
    Float(Vec<f32>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
    Idx(Vec<Idx>),
}

impl From<Vec<f32>>  for BufferData { fn from(data: Vec<f32> ) -> Self {BufferData::Float(data)}}
impl From<Vec<u8>>   for BufferData { fn from(data: Vec<u8>  ) -> Self {BufferData::U8   (data)}}
impl From<Vec<bool>> for BufferData { fn from(data: Vec<bool>) -> Self {BufferData::Bool (data)}}
impl From<Vec<u16>>  for BufferData { fn from(data: Vec<u16> ) -> Self {BufferData::Idx  (data)}}

#[derive(Clone, Debug, PartialEq)]
struct Buffer {
    shape: Vec<Idx>,
    name: Option<String>,
    data: BufferData,
}

#[allow(dead_code)]
enum BufferType {
    Float,
    U8,
    Bool,
    Idx,
}

impl Buffer {
    fn new(typee: BufferType, shape: Vec<Idx>) -> Self {
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
    let (mut shapes, datas): (Vec<Vec<Idx>>, Vec<BufferData>) = array
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

#[allow(dead_code, unused_variables, unused_mut)]
fn load_buffer(filename: &str) -> std::io::Result<Buffer> {
    let mut file = std::fs::File::open(filename)?;
    //let mut buf = Vec::with_capacity(file.size());
    //file.read(&buf);
    //let shape_len = buf[1];
    //let shape = buf
    //let data = match buf[0] {
    //    0x00 => BufferData::Float(),
    //}
    //Ok(Buffer {
    //    shape,
    //    data,
    //    name: filename,
    //})
    unimplemented!()
}

type OperatorDimensions = Option<Idx>;

#[derive(Debug)]
enum BinaryOperatorType {
    Multiplication,
    Greater,
    Less,
    Minus,
    Plus,
    Stack,
}

impl BinaryOperatorType {
    // TODO: impl From
    fn from_str(op: &str) -> Result<BinaryOperatorType, String> {
        match op {
            "*" => Ok(BinaryOperatorType::Multiplication),
            ">" => Ok(BinaryOperatorType::Greater),
            "<" => Ok(BinaryOperatorType::Less),
            "-" => Ok(BinaryOperatorType::Minus),
            "+" => Ok(BinaryOperatorType::Plus),
            "stack" => Ok(BinaryOperatorType::Stack),
            _ => Err(format!("binary operator '{}' is not implemented", op)),
        }
    }

    // TODO: impl Debug
    #[allow(dead_code)]
    fn to_string(&self) -> String {
        match self {
            BinaryOperatorType::Multiplication => "*",
            BinaryOperatorType::Greater => ">",
            BinaryOperatorType::Less => "<",
            BinaryOperatorType::Minus => "-",
            BinaryOperatorType::Plus => "+",
            BinaryOperatorType::Stack => "stack",
        }.to_owned()
    }
}

#[derive(Debug)]
#[allow(dead_code)]
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
                let first_buf = first_argument.eval();
                let second_buf = second_argument.eval();
                match operator {
                    BinaryOperatorType::Multiplication => {
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
                    BinaryOperatorType::Greater => {
                        Buffer {
                            data: match (first_buf.data, second_buf.data) {
                                // > :: (float, float) -> bool
                                (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Bool(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x > y)
                                    .collect()
                                ),
                                // > :: (idx, idx) -> bool
                                (BufferData::Idx(fd), BufferData::Idx(sd)) => BufferData::Bool(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x > y)
                                    .collect()
                                ),
                                _ => unimplemented!(),
                            },
                            shape: first_buf.shape,
                            name: None,
                        }
                    },
                    BinaryOperatorType::Less => {
                        Buffer {
                            data: match (first_buf.data, second_buf.data) {
                                // < :: (float, float) -> bool
                                (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Bool(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x < y)
                                    .collect()
                                ),
                                // < :: (idx, idx) -> bool
                                (BufferData::Idx(fd), BufferData::Idx(sd)) => BufferData::Bool(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x < y)
                                    .collect()
                                ),
                                _ => unimplemented!(),
                            },
                            shape: first_buf.shape,
                            name: None,
                        }
                    },
                    BinaryOperatorType::Minus => {
                        Buffer {
                            data: match (first_buf.data, second_buf.data) {
                                // - :: (float, float) -> float
                                (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x - y)
                                    .collect()
                                ),
                                _ => unimplemented!(),
                            },
                            shape: first_buf.shape,
                            name: None,
                        }
                    },
                    BinaryOperatorType::Plus => {
                        Buffer {
                            data: match (first_buf.data, second_buf.data) {
                                // * :: (float, float) -> float
                                (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x + y)
                                    .collect()
                                ),
                                // * :: (idx, idx) -> idx
                                (BufferData::Idx(fd), BufferData::Idx(sd)) => BufferData::Idx(fd
                                    .iter()
                                    .zip(sd.iter())
                                    .map(|(x, y)| x + y)
                                    .collect()
                                ),
                                _ => unimplemented!(),
                            },
                            shape: first_buf.shape,
                            name: None,
                        }
                    },
                    BinaryOperatorType::Stack => {
                        stack_along(vec![first_buf, second_buf], 0)
                    }
                }
            },
        }
    }

    // TODO: move to Debug implementation
    #[allow(dead_code)]
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
                format!("({}) {} {}", first_argument.to_string(), format!("{}{}", operator.to_string(), match dimensions {
                    Some(dims) => format!("#{:?}", dims),
                    None => "".to_owned(),
                }), second_argument.to_string())
            },
        }
    }
}

struct TokenStream(Vec<String>);

#[derive(Debug)]
enum OperatorOrOperand {
    Operator(String),
    Operand(Node),
}

impl<'a> TokenStream {
    fn next(&mut self) -> Option<String> {
        self.0.pop()
    }

    fn peek(&'a self) -> Option<&'a String> {
        self.0.last()
    }

    fn parse_operator(&mut self, operator: String) -> (String, Option<String>) {
        // TODO: multidimensinal operators??
        if operator.starts_with("max#") {
            ("max".to_owned(), Some(operator[4..].to_owned()))
        } else if operator.starts_with("stack#") {
            ("stack".to_owned(), Some(operator[6..].to_owned()))
        } else {
            (operator, None)
        }
    }

    fn parse_buffer(&mut self) -> Buffer {
        let token = self.next().unwrap();
        if token == "[" { // array
            let mut array = Vec::new();
            while self.peek().unwrap() != "]" {
                array.push(self.parse_buffer());
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

    fn parse_operator_or_operand(&mut self) -> OperatorOrOperand {
        fn is_operator(token: &str) -> bool {
            token == "fract" || token == "-" || token == "abs" || token.starts_with("max") || token.starts_with("stack")
        }
        let token = self.peek().unwrap();
        if token == "(" {
            if self.next().unwrap() != "(" {
                unreachable!();
            }
            let res = OperatorOrOperand::Operand(self.parse());
            if self.next().unwrap() != ")" {
                panic!("Expected ')' after end of expression");
            }
            res
        } else if token == "[" || !is_operator(token) {
            OperatorOrOperand::Operand(Node::Buffer(Rc::new(self.parse_buffer())))
        } else {
            OperatorOrOperand::Operator(self.next().unwrap())
        }
    }

    fn parse(&mut self) -> Node {
        // TODO: disallow operator_named identifiers
        #[derive(Debug)]
        enum UnaryOp {
            UnaryOperator(String),
            BinaryOperator(String, Node),
        }
        let mut unary_ops_stack = Vec::new();
        let first_argument = loop {
            unary_ops_stack.push(
                match self.parse_operator_or_operand() {
                    OperatorOrOperand::Operator(op) => {
                        UnaryOp::UnaryOperator(op)
                    },
                    OperatorOrOperand::Operand(x) => {
                        match self.peek() {
                            Some(binary_operator) if binary_operator == ")" => break x,
                            None => break x,
                            _ => {},
                        }
                        UnaryOp::BinaryOperator(self.next().unwrap(), x)
                    },
                }
            );
        };
        // self.next(); // TODO: assert ")" here and similar places
        unary_ops_stack
            .into_iter()
            .rev()
            .fold(
                first_argument,
                |acc, op| {
                    match op {
                        UnaryOp::UnaryOperator(operator) => {
                            let (op, dims) = self.parse_operator(operator);
                            let dims = dims.map(|d| d.parse::<Idx>().unwrap());
                            Node::UnaryOperator {
                                operator: op,
                                argument: Box::new(acc),
                                dimensions: dims,
                            }
                        },
                        UnaryOp::BinaryOperator(operator, buf) => {
                            let (op, dims) = self.parse_operator(operator);
                            let dims = dims.map(|d| d.parse::<Idx>().unwrap());
                            Node::BinaryOperator {
                                operator: BinaryOperatorType::from_str(op.as_str()).unwrap(),
                                first_argument: Box::new(buf),
                                second_argument: Box::new(acc),
                                dimensions: dims,
                            }
                        },
                    }
                }
            )
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
        // TODO: assure every character of string is parsed
        lazy_static! {
            static ref RE: Regex = Regex::new(r#"\s*((max|stack|abs|-|<|>|fract|\+|\*)(#\d*(\.5)?)?|[a-zA-Z0-9.]+|[()\[\]])\s*"#).unwrap();
        }
        Ok(RE
            .captures_iter(s)
            .map(|capture| capture[1].to_string())
            .collect::<Vec<String>>()
            .into_iter()
            .rev()
            .collect::<TokenStream>()
            .parse())
        // TODO: check stream ended
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
    // TODO: check ast correctness
    let res = ast.eval();
    dump_buffer(&res, "test.buf").unwrap();
    // dump_buffer(&res, format!("{}.buf", buf.name)).unwrap();
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use super::{Node, Buffer, BufferType, BufferData, BinaryOperatorType};

    #[test]
    fn string_to_expression() {
        assert_eq!(
            "(.4 > max#2 abs (x stack#2 y) - [.5 .5]) * 7. * fract x + y"
                .parse::<Node>()
                .unwrap()
                .to_string(),
            "((.4) > max#2 abs ((x) stack#2 y) - no_name) * (7.) * fract (x) + y",
        );
    }

    #[test]
    fn string_to_expression_simple() {
        assert_eq!(
            "(.4 > (x stack#2 y) - [.5 .5])"
                .parse::<Node>()
                .unwrap()
                .to_string(),
            "(.4) > ((x) stack#2 y) - no_name",
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
            operator: BinaryOperatorType::Multiplication,
            first_argument: Box::new(Node::BinaryOperator {
                operator: BinaryOperatorType::Less,
                first_argument: Box::new(Node::UnaryOperator {
                    operator: "max".to_owned(),
                    argument: Box::new(Node::UnaryOperator {
                        operator: "abs".to_owned(),
                        argument: Box::new(Node::BinaryOperator {
                            operator: BinaryOperatorType::Minus,
                            first_argument: Box::new(Node::BinaryOperator {
                                operator: BinaryOperatorType::Stack,
                                first_argument: Box::new(Node::Buffer(x_buffer.clone())),
                                second_argument: Box::new(Node::Buffer(y_buffer.clone())),
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
                    operator: BinaryOperatorType::Multiplication,
                    first_argument: Box::new(Node::BinaryOperator {
                        operator: BinaryOperatorType::Plus,
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
            "((max#2 abs ((x) stack#2 y) - [0.5 0.5]) < 0.4) * fract ((x) + y) * 7".to_owned(),
        );
    }

    #[test]
    fn adamar_product() {
        assert_eq!(
            "([[1 2][3 4]] * [[1 2][3 4]])".parse::<Node>().unwrap().eval(),
            Buffer {
                shape: vec![2, 2],
                data: BufferData::Idx(vec![1, 4, 9, 16]),
                name: None,
            },
        );
    }

    #[test]
    fn test_eval_4() {
        assert_eq!(
            "2*2".parse::<Node>().unwrap().eval(),
            Buffer {
                shape: vec![],
                data: BufferData::Idx(vec![4]),
                name: None,
            },
        );
    }

    #[test]
    fn test_eval_6() {
        assert_eq!(
            "2+2*2".parse::<Node>().unwrap().eval(),
            Buffer {
                shape: vec![],
                data: BufferData::Idx(vec![6]),
                name: None,
            },
        );
    }

    #[test]
    fn test_eval_8() {
        assert_eq!(
            "2*2+2".parse::<Node>().unwrap().eval(),
            Buffer {
                shape: vec![],
                data: BufferData::Idx(vec![8]),
                name: None,
            },
        );
    }
}

