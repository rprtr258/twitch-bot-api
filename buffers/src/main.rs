use std::{
    env::args,
    io::Write,
    process::exit,
    rc::Rc,
};
use {
    lazy_static::lazy_static,
    regex::Regex,
};

type Idx = usize;
type Shape = Vec<Idx>;

trait Buffer<T> {
    fn get(&self, i: Idx) -> T;
    fn iter<I>(&self) -> I where I: Iterator<Item=T>;
    fn dim(&self, i: Idx) -> Idx;
    fn rank(&self) -> Idx;
    fn shape(&self) -> Vec<Idx> {
        (0..self.rank())
            .map(|i| self.dim(i))
            .collect()
    }
}

enum FloatBuffer {
    Materialized(Shape, Vec<f32>),
    Stacked(Shape, Idx, Vec<Box<FloatBuffer>>),
}

enum FloatBufferIter<'a> {
    Materialized(&'a Vec<f32>),
    Stacked(&'a Shape, Idx, &'a Vec<Box<FloatBuffer>>),
}

impl Buffer<f32> for FloatBuffer {
    fn get(&self, i: Idx) -> f32 {
        match self {
            FloatBuffer::Materialized(_, data) => data[i],
            _ => unimplemented!(),
        }
    }

    fn iter<FloatBufferIter>(&self) -> FloatBufferIter {
        match self {
            FloatBuffer::Materialized(_, data) => FloatBufferIter<'_>::Materialized(&data),
            _ => unimplemented!(),
        }
    }

    fn dim(&self, i: Idx) -> Idx {
        match self {
            FloatBuffer::Materialized(shape, _) => shape[i],
            FloatBuffer::Stacked(shape, dim, data) => if i < *dim {shape[i]} else if i == *dim {data.len()} else {shape[i - 1]},
            _ => unimplemented!(),
        }
    }

    fn rank(&self) -> Idx {
        match self {
            FloatBuffer::Materialized(shape, _) => shape.len(),
            FloatBuffer::Stacked(shape, _, _) => shape.len() + 1,
            _ => unimplemented!(),
        }
    }
}

enum U8Buffer {
    Materialized(Shape, Vec<u8>),
}

enum BoolBuffer {
    Materialized(Shape, Vec<bool>),
}

enum IdxBuffer {
    Materialized(Shape, Vec<Idx>),
}

// TODO: change to generics?
#[derive(Clone, Debug, PartialEq)]
enum BufferData {
    Float(FloatBuffer),
    U8   (U8Buffer),
    Bool (BoolBuffer),
    Idx  (IdxBuffer),
}

impl BufferData {
    fn dim(&self, i: Idx) -> Idx {
        match self {
            BufferData::Float(buf) => buf.dim(i),
            _ => unimplemented!(),
        }
    }

    fn rank(&self) -> Idx {
        match self {
            BufferData::Float(buf) => buf.rank(),
            _ => unimplemented!(),
        }
    }

    fn shape(&self) -> Vec<Idx> {
        (0..self.rank())
            .map(|i| self.dim(i))
            .collect()
    }
}

impl From<Vec<f32>>   for BufferData {fn from(data: Vec<f32>  ) -> Self {BufferData::Float(FloatBuffer::Materialized(vec![data.len()], data))}}
impl From<Vec<u8>>    for BufferData {fn from(data: Vec<u8>   ) -> Self {BufferData::U8   (   U8Buffer::Materialized(vec![data.len()], data))}}
impl From<Vec<bool>>  for BufferData {fn from(data: Vec<bool> ) -> Self {BufferData::Bool ( BoolBuffer::Materialized(vec![data.len()], data))}}
impl From<Vec<usize>> for BufferData {fn from(data: Vec<usize>) -> Self {BufferData::Idx  (  IdxBuffer::Materialized(vec![data.len()], data))}}

// TODO: stack not only along 0
fn stack_along(array: Vec<BufferData>, dim: Idx) -> BufferData {
    {
        if dim > array.len() {
            panic!("stack dim must be >=0 and <=len(array), but it's {} when len(array)={}", dim, array.len());
        }
        let first_item = &array[0];
        if !array.iter().skip(1).all(|buf| match (&first_item, &buf) {
            (BufferData::Float(_), BufferData::Float(_)) => true,
            (BufferData::U8(_), BufferData::U8(_)) => true,
            (BufferData::Bool(_), BufferData::Bool(_)) => true,
            (BufferData::Idx(_), BufferData::Idx(_)) => true,
            _ => false,
        } && first_item.shape() == buf.shape()) {
            panic!("all items must have the same type and shape");
        }
    }
    let unstacked_shape = array[0].shape();
    match array[0] {
        BufferData::Float(_) => BufferData::Float(FloatBuffer::Stacked(
                unstacked_shape,
                dim,
                array
                    .into_iter()
                    .map(|buf| match buf {
                        BufferData::Float(float_buf) => Box::new(float_buf),
                        _ => unreachable!(),
                    })
                    .collect()
        )),
        _ => unimplemented!(),
    }
}

fn dump_buffer(buf: &BufferData, filename: &str) -> std::io::Result<()> {
    // TODO: reset file content
    let mut file = std::fs::File::options()
        .create(true)
        .write(true)
        .open(filename)?;
    // type
    file.write(match buf {
        BufferData::Float(_) => &[0x00],
        BufferData::U8(_) => &[0x01],
        BufferData::Bool(_) => &[0x02],
        BufferData::Idx(_) => &[0x03],
    })?;
    // shape length
    file.write(&[buf.rank().try_into().unwrap()])?;
    // shape
    file.write(&buf.shape().iter().map(|x| (*x).try_into().unwrap()).collect::<Vec<u8>>().as_slice())?;
    // data
    println!("{:?}", buf);
    file.write(&match &buf {
        BufferData::Float(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
        _ => unimplemented!(),
        // BufferData::U8(data) => data.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
        // BufferData::Bool(data) => data.iter().map(|x| if *x {0x01u8} else {0x00u8}).collect::<Vec<u8>>(),
        // BufferData::Idx(data) => data.iter().map(|x| x % 0xFFFF).flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>(),
    }.as_slice())?;
    Ok(())
}

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

type OperatorDimensions = Option<Idx>;

#[derive(Debug)]
enum OperatorType {
    Multiplication,
    Greater,
    Less,
    Minus,
    Plus,
    Stack,
    Max,
    Abs,
    Fract,
}

impl OperatorType {
    fn from_str(s: &str) -> Result<OperatorType, String> {
        match s {
            "*"     => Ok(OperatorType::Multiplication),
            ">"     => Ok(OperatorType::Greater       ),
            "<"     => Ok(OperatorType::Less          ),
            "-"     => Ok(OperatorType::Minus         ),
            "+"     => Ok(OperatorType::Plus          ),
            "stack" => Ok(OperatorType::Stack         ),
            "max"   => Ok(OperatorType::Max           ),
            "abs"   => Ok(OperatorType::Abs           ),
            "fract" => Ok(OperatorType::Fract         ),
            _ => Err(format!("operator '{}' is not implemented", s)),
        }
    }

    // TODO: impl Debug
    #[allow(dead_code)]
    fn to_str(&self) -> String {
        match self {
            OperatorType::Multiplication => "*"    ,
            OperatorType::Greater        => ">"    ,
            OperatorType::Less           => "<"    ,
            OperatorType::Minus          => "-"    ,
            OperatorType::Plus           => "+"    ,
            OperatorType::Stack          => "stack",
            OperatorType::Max            => "max"  ,
            OperatorType::Abs            => "abs"  ,
            OperatorType::Fract          => "fract",
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
            Some(i) => (&op[..i], Some(op[i+1..].parse::<Idx>().unwrap())),
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
    Buffer(Rc<BufferData>),
}

impl Node {
    fn eval(&self) -> BufferData {
        match self {
            Node::Buffer(buf) => (**buf).clone(),
            Node::UnaryOperator {operator, argument, ..} => {
                let arg = argument.eval();
                println!("op={} a=[{:?}]{:?}", operator.to_str(), arg.shape, arg.data);
                // TODO: implement abs
                // TODO: implement fract
                // TODO: implement max#2
                match operator.typee {
                    OperatorType::Max => {
                        match arg.data {
                            // max#k :: float[a1,..,ak,..,an] -> float[a1,..,~ak,..,an]
                            BufferData::Float(d) => {
                                let dim = operator.dimensions.unwrap();
                                let mut new_shape = arg.shape.clone();
                                new_shape.remove(dim);
                                // TODO: not only dim=2
                                let shift = arg.shape[0] * arg.shape[1];
                                println!("shift={}", shift);
                                let mut new_data = Vec::with_capacity(d.len() / arg.shape[dim]);
                                for j in 0..shift {
                                    let mut res = d[j];
                                    for i in 1..arg.shape[dim] {
                                        res = res.max(d[j + i * shift]);
                                    }
                                    new_data.push(res);
                                }
                                BufferData::Float(FloatBuffer::Materialized(new_shape, new_data))
                            },
                            _ => unimplemented!(),
                        }
                    },
                    OperatorType::Abs => {
                        match arg.data {
                            // abs :: float[*sh] -> float[*sh]
                            BufferData::Float(d) => BufferData::Float(FloatBuffer::Materialized(arg.shape, d.into_iter().map(|x| x.abs()).collect())),
                            _ => unimplemented!(),
                        }
                    },
                    ref t => unimplemented!("Unary operator '{}' is not implemented", t.to_str()),
                }
            },
            Node::BinaryOperator {operator, first_argument, second_argument, ..} => {
                let first_buf = first_argument.eval();
                let second_buf = second_argument.eval();
                println!("op={} f=[{:?}]{:?} s=[{:?}]{:?}", operator.to_str(), first_buf.shape, first_buf.data, second_buf.shape, second_buf.data);
                match operator.typee {
                    OperatorType::Multiplication => match (first_buf.data, second_buf.data) {
                        // * :: (float, float) -> float
                        (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(FloatBuffer::Materialized(
                                first_buf.shape,
                                fd
                                    .iter()
                                    .zip(sd.iter()) // TODO: check validity of zip
                                    .map(|(x, y)| x * y)
                                    .collect()
                        )),
                        _ => unimplemented!(),
                    },
                    OperatorType::Greater => match (first_buf.data, second_buf.data) {
                        // > :: (float, float) -> bool
                        (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(FloatBuffer::Materialized(
                                first_buf.shape,
                                fd
                                    .iter()
                                    .zip(sd.iter()) // TODO: check validity of zip
                                    .map(|(x, y)| x > y)
                                    .collect()
                        )),
                        _ => unimplemented!(),
                    },
                    OperatorType::Less => match (first_buf.data, second_buf.data) {
                        // < :: (float, float) -> bool
                        (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(FloatBuffer::Materialized(
                                fd.shape,
                                fd
                                    .iter()
                                    .zip(sd.iter()) // TODO: check validity of zip
                                    .map(|(x, y)| x < y)
                                    .collect()
                        )),
                        _ => unimplemented!(),
                    },
                    OperatorType::Minus => match (first_buf.data, second_buf.data) {
                        // - :: (float, float) -> float
                        (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(FloatBuffer::Materialized(
                            first_buf.shape,
                            fd
                                .iter()
                                .zip(sd.iter())
                                .map(|(x, y)| x - y)
                                .collect()
                        )),
                        _ => unimplemented!(),
                    },
                    OperatorType::Plus => match (first_buf.data, second_buf.data) {
                        // + :: (float, float) -> float
                        (BufferData::Float(fd), BufferData::Float(sd)) => BufferData::Float(FloatBuffer::Materialized(
                            first_buf.shape,
                            fd
                                .iter()
                                .zip(sd.iter())
                                .map(|(x, y)| x + y)
                                .collect()
                        )),
                    },
                    OperatorType::Stack => {
                        stack_along(vec![first_buf, second_buf], operator.dimensions.unwrap_or(0))
                    },
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
                "no_name".to_owned()
            },
            Node::UnaryOperator {operator, argument} => {
                format!("{} {}", operator.to_str(), argument.to_str())
            },
            Node::BinaryOperator {operator, first_argument, second_argument} => {
                format!("({}) {} {}", first_argument.to_str(), operator.to_str(), second_argument.to_str())
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

    fn parse_buffer(&mut self) -> BufferData {
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
            fn lookup_buffer(s: String) -> Result<BufferData, String> {
                // TODO: lookup known/cached buffers first
                match s.as_str() {
                    "x" => Ok(BufferData::Float(FloatBuffer::Materialized(
                        vec![3, 3],
                        BufferData::Float(vec![-1., -1., -1., -1., -1., 0., 0., 0., 0., 0., 1.,1.,1.,1.,1.,]),
                    ))),
                    "y" => Ok(BufferData::Float(FloatBuffer::Materialized(
                        vec![3, 3],
                        BufferData::Float(vec![1., 0., -1., 1., 0., -1., 1., 0., -1.]),
                    ))),
                    _ => Err(format!("Buffer '{}' was not found", s))
                }
            }
            // TODO: regexes
            // TODO: linear/monadic interface
            // TODO: compile regex compile-time
            // TODO: assure every character of string is parsed
            lazy_static! {
                static ref re_float: Regex = Regex::new(r#"(\d*\.\d*)"#).unwrap();
                static ref re_u8: Regex = Regex::new(r#"0x([0-9A-F]{2})"#).unwrap();
                static ref re_idx: Regex = Regex::new(r#"(\d{,4})"#).unwrap();
                static ref re_bool: Regex = Regex::new(r#"(true|false)"#).unwrap();
            }
            if re_float.is_match(token) {
                BufferData::Float(FloatBuffer::Materialized(vec![], re_float.captures_iter(token).take(1).collect()[0].parse()))
            } else {
                lookup_buffer(token)
            }
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
            UnaryOperator(Operator),
            BinaryOperator(Operator, Node),
        }
        let mut unary_ops_stack = Vec::new();
        let first_argument = loop {
            unary_ops_stack.push(
                match self.parse_operator_or_operand() {
                    OperatorOrOperand::Operator(op) => {
                        UnaryOp::UnaryOperator(Operator::from_str(op.as_str()).unwrap())
                    },
                    OperatorOrOperand::Operand(x) => {
                        match self.peek() {
                            Some(binary_operator) if binary_operator == ")" => break x,
                            None => break x,
                            _ => {},
                        }
                        UnaryOp::BinaryOperator(Operator::from_str(self.next().unwrap().as_str()).unwrap(), x)
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
    use super::{Node, BufferData, Operator, OperatorType, FloatBuffer};

    #[test]
    fn string_to_expression() {
        assert_eq!(
            "(.4 > max#2 abs (x stack#2 y) - [.5 .5]) * 7. * fract x + y"
                .parse::<Node>()
                .unwrap()
                .to_str(),
            "((.4) > max#2 abs ((no_name) stack#2 no_name) - no_name) * (7.) * fract (x) + y",
        );
    }

    #[test]
    fn string_to_expression_simple() {
        assert_eq!(
            "(.4 > (x stack#2 y) - [.5 .5])"
                .parse::<Node>()
                .unwrap()
                .to_str(),
            "(.4) > ((x) stack#2 y) - no_name",
        );
    }

    #[test]
    fn expression_to_string() {
        let x_buffer = Rc::new(BufferData::Float(FloatBuffer::Materialized(vec![]/*x.shape*/, vec![])));
        let y_buffer = Rc::new(BufferData::Float(FloatBuffer::Materialized(vec![]/*y.shape*/, vec![])));
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
                                first_argument: Box::new(Node::Buffer(x_buffer.clone())),
                                second_argument: Box::new(Node::Buffer(y_buffer.clone())),
                            }),
                            second_argument: Box::new(Node::Buffer(Rc::new(BufferData::Float(FloatBuffer::Materialize(vec![2], vec![0.5, 0.5]))))),
                        }),
                    }),
                }),
                second_argument: Box::new(Node::Buffer(Rc::new(BufferData::Float(FloatBuffer::Materialize(vec![], vec![0.4]))))),
            }),
            second_argument: Box::new(Node::UnaryOperator {
                operator: Operator {typee: OperatorType::Fract, dimensions: None},
                argument: Box::new(Node::BinaryOperator {
                    operator: Operator {typee: OperatorType::Multiplication, dimensions: None},
                    first_argument: Box::new(Node::BinaryOperator {
                        operator: Operator {typee: OperatorType::Plus, dimensions: None},
                        first_argument: Box::new(Node::Buffer(x_buffer.clone())),
                        second_argument: Box::new(Node::Buffer(y_buffer.clone())),
                    }),
                    second_argument: Box::new(Node::Buffer(Rc::new(BufferData::Float(FloatBuffer::Materialize(vec![], vec![7.]))))),
                }),
            }),
        };
        assert_eq!(
            test_expr.to_str(),
            "((max#2 abs ((x) stack#2 y) - [0.5 0.5]) < 0.4) * fract ((x) + y) * 7".to_owned(),
        );
    }

    #[test]
    fn adamar_product() {
        assert_eq!(
            "([[1 2][3 4]] * [[1 2][3 4]])".parse::<Node>().unwrap().eval(),
            BufferData::Float(FloatBuffer::Materialized(
                vec![2, 2],
                vec![1, 4, 9, 16],
            )),
        );
    }

    #[test]
    fn test_eval_4() {
        assert_eq!(
            "2*2".parse::<Node>().unwrap().eval(),
            BufferData::Float(FloatBuffer::Materialized(
                vec![],
                vec![4],
            )),
        );
    }

    #[test]
    fn test_eval_6() {
        assert_eq!(
            "2+2*2".parse::<Node>().unwrap().eval(),
            BufferData::Float(FloatBuffer::Materialized(
                vec![],
                vec![6],
            )),
        );
    }

    #[test]
    fn test_eval_8() {
        assert_eq!(
            "2*2+2".parse::<Node>().unwrap().eval(),
            BufferData::Float(FloatBuffer::Materialized(
                vec![],
                vec![8],
            )),
        );
    }
}

