use std::{
    fmt::{self, Debug, Display, Formatter},
    iter, mem,
};

use serde::{Deserialize, Serialize};

pub(crate) type DimId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub(crate) dims: Box<[usize]>,
    pub(crate) strides: Box<[usize]>,
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(
            &self
                .dims()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("x"),
        )
    }
}

impl Shape {
    fn contiguous(dims: Box<[usize]>) -> Self {
        Self {
            strides: (1..dims.len())
                .map(|start_dim| dims[start_dim..].iter().product())
                .chain(iter::once(1))
                .collect(),
            dims,
        }
    }

    pub fn elements(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Shape::contiguous(value.into())
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Shape::contiguous(value.into())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape::contiguous(value.into_boxed_slice())
    }
}

impl From<Box<[usize]>> for Shape {
    fn from(value: Box<[usize]>) -> Self {
        Shape::contiguous(value)
    }
}

impl From<Shape> for Vec<usize> {
    fn from(value: Shape) -> Self {
        value.dims.to_vec()
    }
}

impl From<Shape> for Box<[usize]> {
    fn from(value: Shape) -> Self {
        value.dims
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub(crate) shape: Shape,
}

impl Display for Layout {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}f32", self.shape())
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
    pub fn scalar() -> Self {
        Self::from([])
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn strides(&self) -> &[usize] {
        self.shape().strides()
    }

    pub fn elements(&self) -> usize {
        self.shape().elements()
    }

    pub fn size(&self) -> usize {
        self.elements() * mem::size_of::<f32>()
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        Self { shape }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub(crate) data: Box<[f32]>,
    pub(crate) layout: Layout,
}

impl Tensor {
    pub fn from_scalar(value: f32) -> Self {
        Self::from_parts(Box::new([value]), Layout::scalar())
    }

    pub fn from_parts(data: Box<[f32]>, layout: Layout) -> Self {
        Self { data, layout }
    }

    pub fn reshape(mut self, shape: Shape) -> Self {
        self.layout.shape = shape;

        self
    }
}
