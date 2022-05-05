use std::{
    env::args,
    // io::Write,
    process::exit,
};
use {
    lazy_static::lazy_static,
    regex::Regex,
    tch::Tensor,
};

#[derive(Debug)]
enum Buffer {
    Named(String),
    Literal(Tensor),
}

fn lookup_buffer(s: &str) -> Result<Tensor, String> {
    // TODO: lookup known/cached buffers first
    const N: i64 = 400;
    let i = Tensor::arange(N, (tch::Kind::Double, tch::Device::Cpu)) / N as f64;
    let x = Tensor::repeat(&i, &[N]).reshape(&[N, N]);
    match s {
        "x" => Ok(x),
        "y" => Ok(x.transpose(0, 1)),
        _ => Err(format!("Buffer '{}' was not found", s))
    }
}

// TODO: stack along -0.5, 0, 0.5, 1, ..., n-0.5, n, n+0.5
fn stack_along(array: &[Tensor], dim: usize) -> Tensor {
    // {
    //     if dim > array.len() {
    //         panic!("stack dim must be >=0 and <=len(array), but it's {} when len(array)={}", dim, array.len());
    //     }
    //     let first_item = &array[0];
    //     if !array.iter().skip(1).all(|buf| match (&first_item, &buf) {
    //         (Buffer::Float(_), Buffer::Float(_)) => true,
    //         (Buffer::U8(_), Buffer::U8(_)) => true,
    //         (Buffer::Bool(_), Buffer::Bool(_)) => true,
    //         (Buffer::Idx(_), Buffer::Idx(_)) => true,
    //         _ => false,
    //     } && first_item.shape() == buf.shape()) {
    //         panic!("all items must have the same type and shape");
    //     }
    // }
    Tensor::stack(&array[..], dim.try_into().unwrap())
    // let unstacked_shape = array[0].shape();
    // match array[0] {
    //     BufferData::Float(_) => BufferData::Float(FloatBuffer::Stacked(
    //             unstacked_shape,
    //             dim,
    //             array
    //                 .into_iter()
    //                 .map(|buf| match buf {
    //                     BufferData::Float(float_buf) => Box::new(float_buf),
    //                     _ => unreachable!(),
    //                 })
    //                 .collect()
    //     )),
    //     _ => unimplemented!(),
    // }
}

// fn dump_buffer(buf: &Buffer, filename: &str) -> std::io::Result<()> {
//     // TODO: reset file content
//     let mut file = std::fs::File::options()
//         .create(true)
//         .write(true)
//         .open(filename)?;
//     // type
//     file.write(match buf {
//         Buffer::Float(_) => &[0x00],
//         Buffer::U8(_) => &[0x01],
//         Buffer::Bool(_) => &[0x02],
//         Buffer::Idx(_) => &[0x03],
//     })?;
//     // shape length
//     file.write(&[buf.rank().try_into().unwrap()])?;
//     // shape
//     file.write(&buf.shape().iter().map(|x| (*x).try_into().unwrap()).collect::<Vec<u8>>().as_slice())?;
//     // data
//     println!("{:?}", buf);
//     file.write(&match &buf {
//         Buffer::Float(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
//         _ => unimplemented!(),
//         // BufferData::U8(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
//         // BufferData::Bool(data) => data.iter().map(|x| if *x {0x01u8} else {0x00u8}).collect::<Vec<u8>>(),
//         // BufferData::Idx(data) => data.iter().map(|x| x % 0xFFFF).flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
//     }.as_slice())?;
//     Ok(())
// }

//fn load_buffer(filename: &str) -> std::io::Result<Buffer> {
//    let mut file = std::fs::File::open(filename)?;
//    //let mut buf = Vec::with_capacity(file.size());
//    //file.read(&buf);
//    //let shape_len = buf[1];
//    //let shape = buf
//    //let data = match buf[0] {
//    //    0x00 => BufferData::Float(),
//    //}
//    //Ok(Buffer {
//    //    shape,
//    //    data,
//    //})
//    unimplemented!()
//}

type OperatorDimensions = Option<usize>;

#[derive(Debug)]
enum OperatorType {
    Multiplication,
    Greater,
    Less,
    Minus,
    Plus,
    Divide,
    Stack,
    Max,
    Abs,
    Fract,
    Zeros,
    Sin,
    Cos,
}

impl OperatorType {
    fn from_str(s: &str) -> Result<OperatorType, String> {
        match s {
            ">"     => Ok(OperatorType::Greater       ),
            "<"     => Ok(OperatorType::Less          ),
            "-"     => Ok(OperatorType::Minus         ),
            "+"     => Ok(OperatorType::Plus          ),
            "*"     => Ok(OperatorType::Multiplication),
            "/"     => Ok(OperatorType::Divide        ),
            "stack" => Ok(OperatorType::Stack         ),
            "max"   => Ok(OperatorType::Max           ),
            "abs"   => Ok(OperatorType::Abs           ),
            "fract" => Ok(OperatorType::Fract         ),
            "zeros" => Ok(OperatorType::Zeros         ),
            "sin"   => Ok(OperatorType::Sin           ),
            "cos"   => Ok(OperatorType::Cos           ),
            _ => Err(format!("operator '{}' is not implemented", s)),
        }
    }

    // TODO: impl Debug
    #[allow(dead_code)]
    fn to_str(&self) -> String {
        match self {
            OperatorType::Greater        => ">"    ,
            OperatorType::Less           => "<"    ,
            OperatorType::Minus          => "-"    ,
            OperatorType::Plus           => "+"    ,
            OperatorType::Multiplication => "*"    ,
            OperatorType::Divide         => "/",
            OperatorType::Stack          => "stack",
            OperatorType::Max            => "max"  ,
            OperatorType::Abs            => "abs"  ,
            OperatorType::Fract          => "fract",
            OperatorType::Zeros          => "zeros",
            OperatorType::Sin            => "zeros",
            OperatorType::Cos            => "zeros",
        }.to_owned()
    }
}

#[derive(Debug)]
struct Operator {
    typee: OperatorType,
    dimensions: OperatorDimensions,
}

impl Operator {
    // TODO: impl Parse
    fn from_str(op: &str) -> Result<Operator, String> {
        // TODO: multidimensinal operators??
        let (operator_type, dimensions) = match op.find('#') {
            Some(i) => (&op[..i], Some(op[i+1..].parse::<usize>().unwrap())),
            None => (op, None),
        };
        let typee = OperatorType::from_str(operator_type)?;
        Ok(Operator {typee, dimensions})
    }

    fn to_str(&self) -> String {
        format!(
            "{}{}",
            self.typee.to_str(),
            self.dimensions
                .map(|dims| format!("#{:?}", dims))
                .unwrap_or("".to_owned()),
        )
    }
}

#[derive(Debug)]
#[allow(dead_code)]
enum Node {
    UnaryOperator {
        operator: Operator,
        argument: Box<Node>,
    },
    BinaryOperator {
        operator: Operator,
        first_argument: Box<Node>,
        second_argument: Box<Node>,
    },
    Buffer(Buffer),
}

impl Node {
    fn eval(self) -> Tensor {
        match self {
            Node::Buffer(buf) => match buf {
                Buffer::Literal(x) => x,
                Buffer::Named(name) => lookup_buffer(name.as_str()).unwrap(),
            },
            Node::UnaryOperator {operator, argument, ..} => {
                let arg = argument.eval();
                // println!("op={} a=[{:?}]{:?}", operator.to_str(), arg.shape(), arg);
                match operator.typee {
                    // max#k :: float[a1,..,ak,..,an] -> float[a1,..,~ak,..,an]
                    OperatorType::Max => match operator.dimensions {
                        None => arg.max(),
                        Some(k) => arg.max_dim(k.try_into().unwrap(), false).0,
                    },
                    // abs :: float[*sh] -> float[*sh]
                    OperatorType::Abs => arg.abs(),
                    // fract :: float[*sh] -> float[*sh]
                    OperatorType::Fract => arg.frac(),
                    OperatorType::Zeros => if arg.dim() != 1 {
                        panic!("can't eval zeros of tensor with shape {:?}", arg.size());
                    } else {
                        Tensor::zeros(
                            &arg
                                .iter::<f64>()
                                .unwrap()
                                .map(|x| x as i64)
                                .collect::<Vec<i64>>()[..],
                            (tch::Kind::Double, tch::Device::Cpu)
                        )
                    },
                    OperatorType::Sin => arg.sin(),
                    OperatorType::Cos => arg.cos(),
                    ref t => unimplemented!("Unary operator '{}' is not implemented", t.to_str()),
                }
            },
            Node::BinaryOperator {operator, first_argument, second_argument, ..} => {
                let fd = first_argument.eval();
                let sd = second_argument.eval();
                // println!("op={} f=[{:?}]{:?} s=[{:?}]{:?}", operator.to_str(), fd.shape(), fd, sd, sd);
                // let first_shape = first_buf.shape();
                match operator.typee {
                    // * :: (float, float) -> float
                    OperatorType::Multiplication => fd * sd,
                    // / :: (float, float) -> float
                    OperatorType::Divide => fd / sd,
                    // > :: (float, float) -> bool
                    OperatorType::Greater => fd.greater_tensor(&sd),
                    // < :: (float, float) -> bool
                    OperatorType::Less => fd.less_tensor(&sd),
                    // - :: (float, float) -> float
                    OperatorType::Minus => fd - sd,
                    // max :: (float[..], float[..]) -> float[..]
                    OperatorType::Max => fd.maximum(&sd),
                    // + :: (float, float) -> float
                    OperatorType::Plus => fd + sd,
                    OperatorType::Stack => stack_along(&[fd, sd], operator.dimensions.unwrap_or(0)),
                    ref t => unimplemented!("Binary operator {} is not implemented", t.to_str()),
                }
            },
        }
    }

    // TODO: move to Debug implementation
    #[allow(dead_code)]
    fn to_str(&self) -> String {
        match self {
            Node::Buffer(buf) => {
                // shape_str = ",".join(map(str, self.shape))
                // type_str = self.type.name
                // TODO: ({name}@)?{T}[{shape}]
                match buf {
                    Buffer::Named(name) => name.to_string(),
                    Buffer::Literal(data) => if data.dim() == 0 {
                        format!("{}", data.double_value(&[]))
                    } else {
                        // TODO: join
                        let data_len = data.size().into_iter().fold(1, |a, x| a * x);
                        let mut res = "[".to_owned();
                        for i in 0..data_len {
                            res += format!("{}", data.double_value(&[i])).as_str();
                            if i + 1 < data_len {
                                res += " ";
                            }
                        }
                        res += "]";
                        res.to_owned()
                    }
                }
            },
            Node::UnaryOperator {operator, argument} => format!("{} {}", operator.to_str(), argument.to_str()),
            Node::BinaryOperator {operator, first_argument, second_argument} => format!("({}) {} {}", first_argument.to_str(), operator.to_str(), second_argument.to_str()),
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

    fn parse_buffer(&mut self) -> Buffer {
        let token = self.next().unwrap();
        if token == "[" { // array
            let mut array = Vec::new();
            while self.peek().unwrap() != "]" {
                array.push(match self.parse_buffer() {
                    Buffer::Literal(x) => x,
                    _ => unimplemented!(),
                });
            }
            self.next(); // "]"
            if array.len() == 0 {
                panic!("Buffer cannot be empty");
            }
            Buffer::Literal(Tensor::stack(&array[..], 0))
        } else { // single item operand
            // TODO: regexes
            // TODO: linear/monadic interface
            // TODO: compile regex compile-time
            // TODO: assure every character of string is parsed
            lazy_static! {
                static ref RE_FLOAT: Regex = Regex::new(r#"(\d+\.\d*|\d*\.\d+)"#).unwrap();
                static ref RE_U8: Regex = Regex::new(r#"0x([0-9A-F]{2})"#).unwrap();
                static ref RE_IDX: Regex = Regex::new(r#"(\d{,4})"#).unwrap();
                static ref RE_BOOL: Regex = Regex::new(r#"(true|false)"#).unwrap();
            }
            if RE_FLOAT.is_match(&token.as_str()) {
                let data = RE_FLOAT
                    .captures_iter(&token.as_str())
                    .next()
                    .map(|capture| capture[1].to_string())
                    .and_then(|x| x.parse::<f64>().ok())
                    .unwrap();
                Buffer::Literal(Tensor::of_slice(&[data]).reshape(&[]))
            } else {
                Buffer::Named(token)
            }
        }
    }

    fn parse_operator_or_operand(&mut self) -> OperatorOrOperand {
        fn is_unary_operator(token: &str) -> bool {
            token == "fract" ||
            token == "-" ||
            token == "zeros" ||
            token == "abs" ||
            token.starts_with("max") ||
            token.starts_with("stack") ||
            token == "sin" ||
            token == "cos"
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
        } else if is_unary_operator(token) {
            OperatorOrOperand::Operator(self.next().unwrap())
        } else {
            OperatorOrOperand::Operand(Node::Buffer(self.parse_buffer()))
        }
    }

    fn parse(&mut self) -> Node {
        // TODO: disallow operator_named identifiers
        #[derive(Debug)]
        enum UnaryOp {
            UnaryOperator(Operator),
            BinaryOperator(Operator, Node),
        }
        let mut unary_ops_stack = Vec::new();
        let first_argument = loop {
            unary_ops_stack.push(
                {let x = self.parse_operator_or_operand();
                match x {
                    OperatorOrOperand::Operator(op) => UnaryOp::UnaryOperator(Operator::from_str(op.as_str()).unwrap()),
                    OperatorOrOperand::Operand(x) => {
                        match self.peek() {
                            Some(binary_operator) if binary_operator == ")" => break x,
                            None => break x,
                            _ => {},
                        }
                        UnaryOp::BinaryOperator(Operator::from_str(self.next().unwrap().as_str()).unwrap(), x)
                    },
                }}
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
                            Node::UnaryOperator {
                                operator: operator,
                                argument: Box::new(acc),
                            }
                        },
                        UnaryOp::BinaryOperator(operator, buf) => {
                            Node::BinaryOperator {
                                operator: operator,
                                first_argument: Box::new(buf),
                                second_argument: Box::new(acc),
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
            static ref RE: Regex = Regex::new(r#"\s*((max|stack|abs|zeros|-|<|>|fract|\+|\*|/|sin|cos)(#\d*(\.5)?)?|\d+(\.\d*)?|[a-zA-Z0-9.]+|[()\[\]])\s*"#).unwrap();
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

fn array_to_image(arr: Tensor) {
    let shape = arr.size();
    let mut out_data = Vec::new();
    let mut encoder = png_pong::Encoder::new(&mut out_data).into_step_enc();
    // TODO: wtf is step and delay, remove
    let step = png_pong::Step{
        raster: match arr.dim() {
            2 => png_pong::PngRaster::Gray8(pix::Raster::with_u8_buffer(shape[1] as u32, shape[0] as u32, arr
                .flip(&[0])
                .ravel()
                .iter::<f64>()
                .unwrap()
                .map(|x| x as u8)
                .collect::<Vec<u8>>()
            )),
            3 if shape[2] == 3 => png_pong::PngRaster::Rgb8(pix::Raster::with_u8_buffer(shape[1] as u32, shape[0] as u32, arr
                .flip(&[0])
                .ravel()
                .iter::<f64>()
                .unwrap()
                .map(|x| x as u8)
                .collect::<Vec<u8>>()
            )),
            _ => unimplemented!(),
        },
        delay: 0,
    };
    encoder.encode(&step).expect("Failed to add frame");
    std::fs::write("out.png", out_data).expect("Failed to save image");
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
    //println!("res shape: {:?}", res.size());
    array_to_image(res);
    //res.save("test.buf");
    // dump_buffer(&res, "test.buf").unwrap();
    // dump_buffer(&res, format!("{}.buf", buf.name)).unwrap();
}

#[cfg(test)]
mod tests {
    // use std::rc::Rc;
    use tch::Tensor;
    use super::{Node, Buffer, Operator, OperatorType};

    fn tensor(shape: &[i64], data: &[f64]) -> Tensor {
        Tensor::of_slice(data).reshape(shape)
    }

    fn scalar(x: f64) -> Tensor {
        tensor(&[], &[x])
    }

    fn parse_expr(expr: &str) -> Node {
        expr.parse().unwrap()
    }
            
    fn eval_expr(expr: &str) -> Tensor {
        parse_expr(expr).eval()
    }

    #[test]
    fn string_to_expression() {
        assert_eq!(
            parse_expr("(.4 > max#2 abs (x stack#2 y) - [.5 .5]) * 7. * fract x + y").to_str(),
            "((0.4) > max#2 abs ((x) stack#2 y) - [0.5 0.5]) * (7) * fract (x) + y",
        );
    }

    #[test]
    fn string_to_expression_simple() {
        assert_eq!(
            parse_expr("(.4 > (x stack#2 y) - [.5 .5])").to_str(),
            "(0.4) > ((x) stack#2 y) - [0.5 0.5]",
        );
    }

    #[test]
    fn expression_to_string() {
        let test_expr = Node::BinaryOperator {
            operator: Operator {typee: OperatorType::Multiplication, dimensions: None},
            first_argument: Box::new(Node::BinaryOperator {
                operator: Operator {typee: OperatorType::Less, dimensions: None},
                first_argument: Box::new(Node::UnaryOperator {
                    operator: Operator {typee: OperatorType::Max, dimensions: Some(2)},
                    argument: Box::new(Node::UnaryOperator {
                        operator: Operator {typee: OperatorType::Abs, dimensions: None},
                        argument: Box::new(Node::BinaryOperator {
                            operator: Operator {typee: OperatorType::Minus, dimensions: None},
                            first_argument: Box::new(Node::BinaryOperator {
                                operator: Operator {typee: OperatorType::Stack, dimensions: Some(2)},
                                first_argument: Box::new(Node::Buffer(Buffer::Named("x".to_owned()))),
                                second_argument: Box::new(Node::Buffer(Buffer::Named("y".to_owned()))),
                            }),
                            second_argument: Box::new(Node::Buffer(Buffer::Literal(tensor(&[2], &[0.5, 0.5])))),
                        }),
                    }),
                }),
                second_argument: Box::new(Node::Buffer(Buffer::Literal(scalar(0.4f64)))),
            }),
            second_argument: Box::new(Node::UnaryOperator {
                operator: Operator {typee: OperatorType::Fract, dimensions: None},
                argument: Box::new(Node::BinaryOperator {
                    operator: Operator {typee: OperatorType::Multiplication, dimensions: None},
                    first_argument: Box::new(Node::BinaryOperator {
                        operator: Operator {typee: OperatorType::Plus, dimensions: None},
                        first_argument: Box::new(Node::Buffer(Buffer::Named("x".to_owned()))),
                        second_argument: Box::new(Node::Buffer(Buffer::Named("y".to_owned()))),
                    }),
                    second_argument: Box::new(Node::Buffer(Buffer::Literal(scalar(7.)))),
                }),
            }),
        };
        assert_eq!(
            test_expr.to_str(),
            "((max#2 abs ((x) stack#2 y) - [0.5 0.5]) < 0.4) * fract ((x) + y) * 7".to_owned(),
        );
    }

    // // TODO: fix to Idx-s
    #[test]
    fn adamar_product() {
        assert_eq!(
            eval_expr("([[1. 2.][3. 4.]]*[[1. 2.][3. 4.]])"),
            tensor(&[2, 2], &[1., 4., 9., 16.]),
        );
    }

    #[test]
    fn test_eval_4() {
        assert_eq!(eval_expr("2.*2."), scalar(4.));
    }

    #[test]
    fn test_eval_6() {
        assert_eq!(eval_expr("2.+2.*2."), scalar(6.));
    }

    #[test]
    fn test_eval_8() {
        assert_eq!(eval_expr("2.*2.+2."), scalar(8.));
    }
}
