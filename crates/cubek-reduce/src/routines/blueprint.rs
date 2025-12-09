use crate::{BoundChecksInner, LineMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceBlueprint {
    pub line_mode: LineMode,
    pub bound_checks: bool,
    pub kind: ReduceBlueprintKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceBlueprintKind {
    Unit,
    Plane(PlaneReduceBlueprint),
    Cube(CubeReduceBlueprint),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeReduceBlueprint {
    pub accumulator_size: u32,
    pub bound_checks_inner: BoundChecksInner,
    pub use_planes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneReduceBlueprint {
    pub bound_checks_inner: BoundChecksInner,
}
