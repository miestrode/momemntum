use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Index, IndexMut, Mul};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape {
    dimensions: Box<[usize]>,
}

impl Debug for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(
            &self
                .dimensions()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("x"),
        )
    }
}

impl Shape {
    pub fn elements(&self) -> usize {
        self.dimensions().iter().copied().reduce(Mul::mul).unwrap()
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Shape {
            dimensions: value.into(),
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Shape {
            dimensions: value.into(),
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape {
            dimensions: value.into_boxed_slice(),
        }
    }
}

impl From<Box<[usize]>> for Shape {
    fn from(value: Box<[usize]>) -> Self {
        Shape { dimensions: value }
    }
}

impl From<Shape> for Vec<usize> {
    fn from(value: Shape) -> Self {
        value.dimensions.to_vec()
    }
}

impl From<Shape> for Box<[usize]> {
    fn from(value: Shape) -> Self {
        value.dimensions
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
}

impl Debug for Layout {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}f32", self.shape())
    }
}

impl<T: Into<Shape>> From<T> for Layout {
    fn from(value: T) -> Self {
        Self {
            shape: value.into(),
        }
    }
}

impl Layout {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ExprId(pub(crate) usize);

impl Debug for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl From<ExprId> for usize {
    fn from(value: ExprId) -> Self {
        value.0
    }
}

impl From<usize> for ExprId {
    fn from(value: usize) -> Self {
        ExprId(value)
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) data: Box<[f32]>,
    pub(crate) layout: Layout,
}

impl Tensor {
    pub fn from_parts(data: Box<[f32]>, layout: Layout) -> Self {
        Self { data, layout }
    }

    pub fn from_element(element: f32, layout: Layout) -> Self {
        Self {
            data: vec![element; layout.shape().elements()].into_boxed_slice(),
            layout,
        }
    }
}

#[derive(Clone)]
pub enum Op {
    Add,
    Mul,
}

impl Op {
    fn infer_layout(&self, children: &[&Layout]) -> Layout {
        match self {
            Op::Add => children[0].clone(),
            Op::Mul => children[0].clone(),
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Op::Add => "add",
            Op::Mul => "mul",
        })
    }
}

#[derive(Clone)]
pub(crate) enum ExprBody {
    Op { op: Op, children: Vec<ExprId> },
    VarTensor { layout: Layout, input: bool },
    ConstTensor(Tensor),
}

impl Debug for ExprBody {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ExprBody::Op { op, children } => {
                let mut result = f.debug_tuple(&op.to_string());

                for child in children {
                    result.field(child);
                }

                result.finish()
            }
            ExprBody::VarTensor { .. } => f.write_str("?"),
            ExprBody::ConstTensor(tensor) => Debug::fmt(&tensor.data, f),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprInfo {
    pub(crate) expression: ExprBody,
    pub(crate) usages: HashMap<ExprId, usize>,
    pub(crate) layout: Layout,
}

#[derive(Default)]
pub struct Graph {
    pub(crate) inputs: Vec<ExprId>,
    pub(crate) expressions: Vec<ExprInfo>,
    pub(crate) outputs: Vec<ExprId>,
}

impl Index<ExprId> for Graph {
    type Output = ExprInfo;

    fn index(&self, index: ExprId) -> &Self::Output {
        &self.expressions[index.0]
    }
}

impl IndexMut<ExprId> for Graph {
    fn index_mut(&mut self, index: ExprId) -> &mut Self::Output {
        &mut self.expressions[index.0]
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn add_usage(&mut self, expr: ExprId, user: ExprId) {
        self[expr]
            .usages
            .entry(user)
            .and_modify(|instances| *instances += 1)
            .or_insert(1);
    }

    fn add_expr_untracked(&mut self, expression: ExprBody) {
        let layout = match &expression {
            ExprBody::Op { op, children } => op.infer_layout(
                &children
                    .iter()
                    .copied()
                    .map(|expr| &self[expr].layout)
                    .collect::<Vec<_>>(),
            ),
            ExprBody::VarTensor { layout, .. } => layout.clone(),
            ExprBody::ConstTensor(tensor) => tensor.layout.clone(),
        };

        self.expressions.push(ExprInfo {
            expression,
            layout,
            usages: HashMap::new(),
        });
    }

    fn add_expr(&mut self, expr: ExprBody) -> ExprId {
        let id = ExprId::from(self.expressions.len());

        match &expr {
            ExprBody::Op { children, .. } => {
                for child in children.iter().copied() {
                    self.add_usage(child, id);
                }
            }
            ExprBody::VarTensor { input, .. } => {
                if *input {
                    self.inputs.push(id);
                }
            }
            ExprBody::ConstTensor(..) => {}
        }

        self.add_expr_untracked(expr);

        id
    }

    pub fn add_input(&mut self, layout: Layout) -> ExprId {
        self.add_expr(ExprBody::VarTensor {
            layout,
            input: true,
        })
    }

    pub fn add_variable(&mut self, layout: Layout) -> ExprId {
        self.add_expr(ExprBody::VarTensor {
            layout,
            input: false,
        })
    }

    pub fn add_constant(&mut self, tensor: Tensor) -> ExprId {
        self.add_expr(ExprBody::ConstTensor(tensor))
    }

    pub fn add_op(&mut self, op: Op, children: &[ExprId]) -> ExprId {
        self.add_expr(ExprBody::Op {
            op,
            children: children.to_owned(),
        })
    }

    pub fn add_output(&mut self, expr: ExprId) {
        self.outputs.push(expr);
    }
}

// impl Graph {
//     fn visit_children(
//         &mut self,
//         gradient_map: &mut HashMap<ExprRef, ExprRef>,
//         expression: ExprRef,
//     ) {
//         match &self[expression.id].expression {
//             ExprBody::UnaryOperation { input, .. } => {
//                 self.visit(gradient_map, *input);
//             }
//             &ExprBody::BinaryOperation {
//                 left_input,
//                 right_input,
//                 ..
//             } => {
//                 let (first_input, second_input) = if left_input.id < right_input.id {
//                     (right_input, left_input)
//                 } else {
//                     (left_input, right_input)
//                 };
//
//                 self.visit(gradient_map, first_input);
//                 self.visit(gradient_map, second_input);
//             }
//             ExprBody::VarTensor { .. } => {}
//             ExprBody::ConstTensor { .. } => {}
//         }
//     }
//
//     fn visit(&mut self, gradient_map: &mut HashMap<ExprRef, ExprRef>, expression: ExprRef) {
//         if gradient_map.contains_key(&expression) {
//             return;
//         }
//
//         gradient_map.insert(
//             expression,
//             self[expression.id()]
//                 .usages
//                 .clone()
//                 .into_iter()
//                 .map(|(usage, count)| vec![usage; count])
//                 .flatten()
//                 .filter(|usage| gradient_map.contains_key(usage))
//                 .map(|usage| {
//                     let gradient = gradient_map[&usage];
//                     let partial_derivative =
//                         match self[usage.id()].expression {
//                             ExprBody::UnaryOperation { operation, .. } => match operation {
//                                 UnaryOperation::Sin => expression.untracked().cos(),
//                                 UnaryOperation::Cos => -expression.untracked().sin(),
//                                 UnaryOperation::Negation => self
//                                     .add_constant(Tensor::from_element(-1.0, expression.layout())),
//                             },
//                             ExprBody::BinaryOperation {
//                                 left_input,
//                                 right_input,
//                                 operation,
//                             } => match operation {
//                                 BinaryOperation::Addition => self
//                                     .add_constant(Tensor::from_element(1.0, expression.layout())),
//                                 BinaryOperation::Multiplication => {
//                                     if left_input == expression {
//                                         right_input
//                                     } else {
//                                         left_input
//                                     }
//                                 }
//                             },
//                             ExprBody::VarTensor { .. } => unreachable!(),
//                             ExprBody::ConstTensor { .. } => unreachable!(),
//                         };
//
//                     gradient.untracked() * partial_derivative.untracked()
//                 })
//                 .reduce(|accumulator, expression| accumulator.untracked() + expression.untracked())
//                 .unwrap(),
//         );
//
//         self.visit_children(gradient_map, expression);
//     }
//
//     pub fn gradients(&mut self, expression: ExprRef) -> HashMap<ExprRef, ExprRef> {
//         let mut gradient_map = HashMap::with_capacity(expression.id().0 + 1);
//
//         let layout = expression.layout();
//         gradient_map.insert(
//             expression,
//             self.add_constant(Tensor::from_element(1.0, layout)),
//         );
//
//         self.visit_children(&mut gradient_map, expression);
//
//         gradient_map
//     }
// }

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}) -> ({}) {{",
            self.inputs
                .iter()
                .map(|input| format!("{input:?}: {:?}", self[*input].layout))
                .collect::<Vec<_>>()
                .join(", "),
            self.outputs
                .iter()
                .map(|output| format!("{output:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;

        if f.alternate() {
            writeln!(f)?;
        } else {
            write!(f, " ")?;
        }

        f.write_str(
            &self
                .expressions
                .iter()
                .zip((0..self.expressions.len()).map(ExprId))
                .filter(|(info, _)| {
                    !matches!(info.expression, ExprBody::VarTensor { input: true, .. })
                })
                .map(|(node, id)| {
                    format!(
                        "{}{id:?}: {:?} = {:?};",
                        if f.alternate() { "    " } else { "" },
                        node.layout,
                        node.expression
                    )
                })
                .collect::<Vec<_>>()
                .join(if f.alternate() { "\n" } else { " " }),
        )?;

        if f.alternate() {
            writeln!(f)?;
        } else {
            write!(f, " ")?;
        }

        write!(f, "}}")
    }
}
