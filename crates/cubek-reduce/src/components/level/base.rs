use crate::LineMode;
use cubecl::prelude::*;

// If line mode is parallel, fill a line with `x, x+1, ... x+ line_size - 1` where `x = first`.
// If line mode is perpendicular, fill a line with `x, x, ... x` where `x = first`.
#[cube]
pub(crate) fn fill_coordinate_line(
    first: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> Line<u32> {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let mut coordinates = Line::empty(line_size);
            #[unroll]
            for j in 0..line_size {
                coordinates[j] = first + j;
            }
            coordinates
        }
        LineMode::Perpendicular => Line::empty(line_size).fill(first),
    }
}
