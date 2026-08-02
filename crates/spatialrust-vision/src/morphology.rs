//! CPU mathematical morphology with explicit structuring elements and borders.

use pulp::Arch;
use rayon::prelude::*;
use spatialrust_image::{Image, ImageView};

use crate::border::{fetch, map_index};
use crate::dispatch::{
    bounded_workers, items_per_worker, should_parallelize, LARGE_PARALLEL_COMPONENTS, ROWS_PER_TILE,
};
use crate::{BorderMode, PixelComponent, VisionError, VisionResult};

/// Built-in structuring-element geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphologyShape {
    /// Every element in the bounding rectangle is active.
    Rect,
    /// The anchor row and column are active.
    Cross,
    /// A filled ellipse inscribed in the bounding rectangle.
    Ellipse,
    /// A filled Manhattan-distance diamond.
    Diamond,
}

/// A validated binary neighborhood mask and anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuringElement {
    width: usize,
    height: usize,
    anchor_x: usize,
    anchor_y: usize,
    mask: Vec<bool>,
}

impl StructuringElement {
    /// Creates a built-in element anchored at its integer center.
    pub fn try_new(shape: MorphologyShape, width: usize, height: usize) -> VisionResult<Self> {
        Self::try_new_with_anchor(shape, width, height, width / 2, height / 2)
    }

    /// Creates a built-in element with an explicit anchor.
    pub fn try_new_with_anchor(
        shape: MorphologyShape,
        width: usize,
        height: usize,
        anchor_x: usize,
        anchor_y: usize,
    ) -> VisionResult<Self> {
        validate_dimensions(width, height, anchor_x, anchor_y)?;
        let mut mask = vec![false; width * height];
        match shape {
            MorphologyShape::Rect => mask.fill(true),
            MorphologyShape::Cross => {
                for y in 0..height {
                    for x in 0..width {
                        mask[y * width + x] = x == anchor_x || y == anchor_y;
                    }
                }
            }
            MorphologyShape::Ellipse => fill_ellipse(&mut mask, width, height),
            MorphologyShape::Diamond => {
                let rx = anchor_x.max(width - 1 - anchor_x).max(1);
                let ry = anchor_y.max(height - 1 - anchor_y).max(1);
                for y in 0..height {
                    for x in 0..width {
                        let dx = x.abs_diff(anchor_x) as f64 / rx as f64;
                        let dy = y.abs_diff(anchor_y) as f64 / ry as f64;
                        mask[y * width + x] = dx + dy <= 1.0 + f64::EPSILON;
                    }
                }
            }
        }
        Self::try_from_mask(width, height, anchor_x, anchor_y, mask)
    }

    /// Creates an element from a row-major binary mask.
    pub fn try_from_mask(
        width: usize,
        height: usize,
        anchor_x: usize,
        anchor_y: usize,
        mask: Vec<bool>,
    ) -> VisionResult<Self> {
        validate_dimensions(width, height, anchor_x, anchor_y)?;
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| VisionError::InvalidParameter("structuring element overflows".into()))?;
        if mask.len() != expected {
            return Err(VisionError::ShapeMismatch(format!(
                "structuring element needs {expected} mask values, found {}",
                mask.len()
            )));
        }
        if !mask.iter().any(|&active| active) {
            return Err(VisionError::InvalidParameter(
                "structuring element must contain an active sample".into(),
            ));
        }
        Ok(Self { width, height, anchor_x, anchor_y, mask })
    }

    /// Element width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Element height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Anchor coordinate `(x, y)`.
    #[must_use]
    pub const fn anchor(&self) -> (usize, usize) {
        (self.anchor_x, self.anchor_y)
    }

    /// Row-major binary mask.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
}

/// Composite morphology operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphologyOperation {
    /// Erosion followed by dilation.
    Open,
    /// Dilation followed by erosion.
    Close,
    /// Dilation minus erosion.
    Gradient,
    /// Input minus opening.
    TopHat,
    /// Closing minus input.
    BlackHat,
}

/// Erodes each channel independently.
pub fn erode<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    repeat_extreme(input, element, iterations, border, Extreme::Minimum)
}

/// Dilates each channel independently.
pub fn dilate<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    repeat_extreme(input, element, iterations, border, Extreme::Maximum)
}

/// Applies a composite morphology operation.
pub fn morphology_ex<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    operation: MorphologyOperation,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let original = pack(input)?;
    match operation {
        MorphologyOperation::Open => {
            let eroded = erode(input, element, iterations, border)?;
            dilate(eroded.view(), element, iterations, border)
        }
        MorphologyOperation::Close => {
            let dilated = dilate(input, element, iterations, border)?;
            erode(dilated.view(), element, iterations, border)
        }
        MorphologyOperation::Gradient => {
            let high = dilate(input, element, iterations, border)?;
            let low = erode(input, element, iterations, border)?;
            subtract(high.view(), low.view())
        }
        MorphologyOperation::TopHat => {
            let opened =
                morphology_ex(input, MorphologyOperation::Open, element, iterations, border)?;
            subtract(original.view(), opened.view())
        }
        MorphologyOperation::BlackHat => {
            let closed =
                morphology_ex(input, MorphologyOperation::Close, element, iterations, border)?;
            subtract(closed.view(), original.view())
        }
    }
}

/// Erodes a grayscale `u8` image with a rectangular element in linear time.
///
/// This specialized path is independent of the rectangle area and preserves
/// the generic implementation's anchor, border, iteration, and metadata
/// semantics.
pub fn erode_rect_u8(
    input: ImageView<'_, u8, 1>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<u8, 1>> {
    let mut output = vec![0; input.width() * input.height()];
    let mut workspace = RectMorphologyWorkspace::new();
    erode_rect_u8_into(input, element, iterations, border, &mut output, &mut workspace)?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Dilates a grayscale `u8` image with a rectangular element in linear time.
pub fn dilate_rect_u8(
    input: ImageView<'_, u8, 1>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<u8, 1>> {
    let mut output = vec![0; input.width() * input.height()];
    let mut workspace = RectMorphologyWorkspace::new();
    dilate_rect_u8_into(input, element, iterations, border, &mut output, &mut workspace)?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Applies composite grayscale `u8` morphology using the rectangular fast path.
pub fn morphology_rect_u8(
    input: ImageView<'_, u8, 1>,
    operation: MorphologyOperation,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<u8, 1>,
) -> VisionResult<Image<u8, 1>> {
    let mut output = vec![0; input.width() * input.height()];
    let mut workspace = RectMorphologyWorkspace::new();
    morphology_rect_u8_into(
        input,
        operation,
        element,
        iterations,
        border,
        &mut output,
        &mut workspace,
    )?;
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

/// Reusable scratch storage for packed grayscale rectangular morphology.
///
/// The workspace owns every full-image intermediate and one set of line
/// buffers per Rayon worker. Grow it once for the largest expected image and
/// element, then reuse it on the same worker thread. It never performs a
/// hidden device transfer and does not share mutable state between calls.
#[derive(Debug, Default)]
pub struct RectMorphologyWorkspace {
    current: Vec<u8>,
    horizontal: Vec<u8>,
    transposed: Vec<u8>,
    filtered: Vec<u8>,
    output: Vec<u8>,
    padded: Vec<u8>,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    line_buffers: Vec<LineBuffers>,
    centered_5x5_active: bool,
}

impl RectMorphologyWorkspace {
    /// Creates an empty workspace that grows on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current: Vec::new(),
            horizontal: Vec::new(),
            transposed: Vec::new(),
            filtered: Vec::new(),
            output: Vec::new(),
            padded: Vec::new(),
            prefix: Vec::new(),
            suffix: Vec::new(),
            line_buffers: Vec::new(),
            centered_5x5_active: false,
        }
    }

    /// Returns the largest pixel count reserved by every active full-image plane.
    #[must_use]
    pub fn capacity(&self) -> usize {
        let active =
            self.current.capacity().min(self.horizontal.capacity()).min(self.output.capacity());
        if self.centered_5x5_active {
            active
        } else {
            active.min(self.transposed.capacity()).min(self.filtered.capacity())
        }
    }

    /// Returns the number of reusable parallel line-buffer sets.
    #[must_use]
    pub fn worker_capacity(&self) -> usize {
        self.line_buffers.len()
    }

    /// Returns the reusable element count reserved by every active line buffer.
    #[must_use]
    pub fn line_capacity(&self) -> usize {
        if self.line_buffers.is_empty() {
            return self.padded.capacity().min(self.prefix.capacity()).min(self.suffix.capacity());
        }
        self.line_buffers
            .iter()
            .map(|buffers| {
                buffers
                    .padded
                    .capacity()
                    .min(buffers.prefix.capacity())
                    .min(buffers.suffix.capacity())
            })
            .min()
            .unwrap_or(0)
    }
}

/// Erodes into caller-owned packed output using reusable scratch storage.
pub fn erode_rect_u8_into(
    input: ImageView<'_, u8, 1>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<u8, 1>,
    output: &mut [u8],
    workspace: &mut RectMorphologyWorkspace,
) -> VisionResult<()> {
    validate_output_len(output, input.width(), input.height())?;
    run_rect_sequence(input, element, border, &[(Extreme::Minimum, iterations)], workspace)?;
    output.copy_from_slice(&workspace.current);
    Ok(())
}

/// Dilates into caller-owned packed output using reusable scratch storage.
pub fn dilate_rect_u8_into(
    input: ImageView<'_, u8, 1>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<u8, 1>,
    output: &mut [u8],
    workspace: &mut RectMorphologyWorkspace,
) -> VisionResult<()> {
    validate_output_len(output, input.width(), input.height())?;
    run_rect_sequence(input, element, border, &[(Extreme::Maximum, iterations)], workspace)?;
    output.copy_from_slice(&workspace.current);
    Ok(())
}

/// Applies composite rectangular morphology into caller-owned packed output.
///
/// `output` must contain exactly `input.width() * input.height()` elements.
/// Safe Rust borrowing requires input and output storage not to overlap.
#[allow(clippy::too_many_arguments)]
pub fn morphology_rect_u8_into(
    input: ImageView<'_, u8, 1>,
    operation: MorphologyOperation,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<u8, 1>,
    output: &mut [u8],
    workspace: &mut RectMorphologyWorkspace,
) -> VisionResult<()> {
    validate_output_len(output, input.width(), input.height())?;
    match operation {
        MorphologyOperation::Open => run_rect_sequence(
            input,
            element,
            border,
            &[(Extreme::Minimum, iterations), (Extreme::Maximum, iterations)],
            workspace,
        )?,
        MorphologyOperation::Close => run_rect_sequence(
            input,
            element,
            border,
            &[(Extreme::Maximum, iterations), (Extreme::Minimum, iterations)],
            workspace,
        )?,
        MorphologyOperation::Gradient => {
            run_rect_sequence(
                input,
                element,
                border,
                &[(Extreme::Maximum, iterations)],
                workspace,
            )?;
            output.copy_from_slice(&workspace.current);
            run_rect_sequence(
                input,
                element,
                border,
                &[(Extreme::Minimum, iterations)],
                workspace,
            )?;
            for (high, &low) in output.iter_mut().zip(&workspace.current) {
                *high = high.saturating_sub(low);
            }
            return Ok(());
        }
        MorphologyOperation::TopHat => {
            pack_u8_into(input, output);
            run_rect_sequence(
                input,
                element,
                border,
                &[(Extreme::Minimum, iterations), (Extreme::Maximum, iterations)],
                workspace,
            )?;
            for (original, &opened) in output.iter_mut().zip(&workspace.current) {
                *original = original.saturating_sub(opened);
            }
            return Ok(());
        }
        MorphologyOperation::BlackHat => {
            run_rect_sequence(
                input,
                element,
                border,
                &[(Extreme::Maximum, iterations), (Extreme::Minimum, iterations)],
                workspace,
            )?;
            copy_subtract_input(input, &workspace.current, output);
            return Ok(());
        }
    }
    output.copy_from_slice(&workspace.current);
    Ok(())
}

#[derive(Clone, Copy)]
enum Extreme {
    Minimum,
    Maximum,
}

fn validate_rect(element: &StructuringElement) -> VisionResult<()> {
    if element.mask.iter().all(|&active| active) {
        Ok(())
    } else {
        Err(VisionError::InvalidParameter(
            "rectangular morphology fast path requires a full rectangular mask".into(),
        ))
    }
}

fn run_rect_sequence(
    input: ImageView<'_, u8, 1>,
    element: &StructuringElement,
    border: BorderMode<u8, 1>,
    stages: &[(Extreme, usize)],
    workspace: &mut RectMorphologyWorkspace,
) -> VisionResult<()> {
    validate_rect(element)?;
    let (width, height) = (input.width(), input.height());
    workspace.current.resize(width * height, 0);
    pack_u8_into(input, &mut workspace.current);
    for &(extreme, iterations) in stages {
        for _ in 0..iterations {
            let current = std::mem::take(&mut workspace.current);
            workspace.apply(
                &current,
                width,
                height,
                element.width,
                element.height,
                element.anchor_x,
                element.anchor_y,
                border,
                extreme,
            );
            workspace.current = current;
            std::mem::swap(&mut workspace.current, &mut workspace.output);
        }
    }
    Ok(())
}

fn validate_output_len(output: &[u8], width: usize, height: usize) -> VisionResult<()> {
    let len = width
        .checked_mul(height)
        .ok_or_else(|| VisionError::InvalidDimensions("morphology output size overflows".into()))?;
    if output.len() != len {
        return Err(VisionError::ShapeMismatch(format!(
            "morphology output needs {len} elements, found {}",
            output.len()
        )));
    }
    Ok(())
}

fn pack_u8_into(input: ImageView<'_, u8, 1>, output: &mut [u8]) {
    for y in 0..input.height() {
        let start = y * input.width();
        output[start..start + input.width()]
            .copy_from_slice(input.row(y).expect("input row in bounds"));
    }
}

fn copy_subtract_input(input: ImageView<'_, u8, 1>, high: &[u8], output: &mut [u8]) {
    for y in 0..input.height() {
        let start = y * input.width();
        for ((value, &closed), &original) in output[start..start + input.width()]
            .iter_mut()
            .zip(&high[start..start + input.width()])
            .zip(input.row(y).expect("input row in bounds"))
        {
            *value = closed.saturating_sub(original);
        }
    }
}

impl RectMorphologyWorkspace {
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        input: &[u8],
        width: usize,
        height: usize,
        kernel_width: usize,
        kernel_height: usize,
        anchor_x: usize,
        anchor_y: usize,
        border: BorderMode<u8, 1>,
        extreme: Extreme,
    ) {
        let len = width * height;
        self.centered_5x5_active = false;
        if kernel_width == 5
            && kernel_height == 5
            && anchor_x == 2
            && anchor_y == 2
            && matches!(border, BorderMode::Replicate)
        {
            self.apply_centered_5x5(input, width, height, extreme);
            return;
        }
        if should_parallelize(len, width.max(height), LARGE_PARALLEL_COMPONENTS) {
            self.apply_parallel(
                input,
                width,
                height,
                kernel_width,
                kernel_height,
                anchor_x,
                anchor_y,
                border,
                extreme,
            );
            return;
        }
        self.horizontal.resize(len, 0);
        for y in 0..height {
            let start = y * width;
            filter_line(
                &input[start..start + width],
                &mut self.horizontal[start..start + width],
                kernel_width,
                anchor_x,
                border,
                extreme,
                &mut self.padded,
                &mut self.prefix,
                &mut self.suffix,
            );
        }

        self.transposed.resize(len, 0);
        transpose_blocked(&self.horizontal, &mut self.transposed, width, height);
        self.filtered.resize(len, 0);
        for x in 0..width {
            let start = x * height;
            filter_line(
                &self.transposed[start..start + height],
                &mut self.filtered[start..start + height],
                kernel_height,
                anchor_y,
                border,
                extreme,
                &mut self.padded,
                &mut self.prefix,
                &mut self.suffix,
            );
        }
        self.output.resize(len, 0);
        transpose_blocked(&self.filtered, &mut self.output, height, width);
    }

    fn apply_centered_5x5(&mut self, input: &[u8], width: usize, height: usize, extreme: Extreme) {
        let len = width * height;
        self.centered_5x5_active = true;
        self.horizontal.resize(len, 0);
        if should_parallelize(len, width.max(height), LARGE_PARALLEL_COMPONENTS) {
            let workers = bounded_workers(
                len,
                width.max(height),
                LARGE_PARALLEL_COMPONENTS,
                rayon::current_num_threads(),
            );
            self.line_buffers.resize_with(workers, LineBuffers::default);
            let line_len = width.max(height) + 4;
            for buffers in &mut self.line_buffers {
                buffers.padded.reserve(line_len.saturating_sub(buffers.padded.len()));
                buffers.prefix.reserve(line_len.saturating_sub(buffers.prefix.len()));
                buffers.suffix.reserve(line_len.saturating_sub(buffers.suffix.len()));
            }
        }
        let arch = Arch::new();
        if should_parallelize(len, height, LARGE_PARALLEL_COMPONENTS) {
            self.horizontal.par_chunks_mut(width * ROWS_PER_TILE).enumerate().for_each(
                |(block, outputs)| {
                    arch.dispatch(|| {
                        for (row, output) in outputs.chunks_mut(width).enumerate() {
                            let y = block * ROWS_PER_TILE + row;
                            filter_centered_5_replicate(
                                &input[y * width..(y + 1) * width],
                                output,
                                extreme,
                            );
                        }
                    });
                },
            );
        } else {
            arch.dispatch(|| {
                for (source, output) in input.chunks(width).zip(self.horizontal.chunks_mut(width)) {
                    filter_centered_5_replicate(source, output, extreme);
                }
            });
        }

        self.output.resize(len, 0);
        if should_parallelize(len, height, LARGE_PARALLEL_COMPONENTS) {
            self.output.par_chunks_mut(width * ROWS_PER_TILE).enumerate().for_each(
                |(block, outputs)| {
                    arch.dispatch(|| {
                        for (row, output) in outputs.chunks_mut(width).enumerate() {
                            filter_vertical_centered_5_replicate(
                                &self.horizontal,
                                width,
                                height,
                                block * ROWS_PER_TILE + row,
                                output,
                                extreme,
                            );
                        }
                    });
                },
            );
        } else {
            arch.dispatch(|| {
                for (y, output) in self.output.chunks_mut(width).enumerate() {
                    filter_vertical_centered_5_replicate(
                        &self.horizontal,
                        width,
                        height,
                        y,
                        output,
                        extreme,
                    );
                }
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_parallel(
        &mut self,
        input: &[u8],
        width: usize,
        height: usize,
        kernel_width: usize,
        kernel_height: usize,
        anchor_x: usize,
        anchor_y: usize,
        border: BorderMode<u8, 1>,
        extreme: Extreme,
    ) {
        let len = width * height;
        let workers = bounded_workers(
            len,
            width.max(height),
            LARGE_PARALLEL_COMPONENTS,
            rayon::current_num_threads(),
        );
        self.line_buffers.resize_with(workers, LineBuffers::default);
        self.horizontal.resize(len, 0);
        let horizontal_rows = items_per_worker(height, workers);
        self.horizontal
            .par_chunks_mut(horizontal_rows * width)
            .zip(self.line_buffers.par_iter_mut())
            .enumerate()
            .for_each(|(chunk, (outputs, buffers))| {
                let first_y = chunk * horizontal_rows;
                for (local_y, output) in outputs.chunks_mut(width).enumerate() {
                    let start = (first_y + local_y) * width;
                    filter_line(
                        &input[start..start + width],
                        output,
                        kernel_width,
                        anchor_x,
                        border,
                        extreme,
                        &mut buffers.padded,
                        &mut buffers.prefix,
                        &mut buffers.suffix,
                    );
                }
            });

        self.transposed.resize(len, 0);
        self.transposed.par_chunks_mut(height).enumerate().for_each(|(x, output)| {
            for (y, value) in output.iter_mut().enumerate() {
                *value = self.horizontal[y * width + x];
            }
        });
        self.filtered.resize(len, 0);
        let vertical_rows = items_per_worker(width, workers);
        self.filtered
            .par_chunks_mut(vertical_rows * height)
            .zip(self.line_buffers.par_iter_mut())
            .enumerate()
            .for_each(|(chunk, (outputs, buffers))| {
                let first_x = chunk * vertical_rows;
                for (local_x, output) in outputs.chunks_mut(height).enumerate() {
                    let start = (first_x + local_x) * height;
                    filter_line(
                        &self.transposed[start..start + height],
                        output,
                        kernel_height,
                        anchor_y,
                        border,
                        extreme,
                        &mut buffers.padded,
                        &mut buffers.prefix,
                        &mut buffers.suffix,
                    );
                }
            });
        self.output.resize(len, 0);
        self.output.par_chunks_mut(width).enumerate().for_each(|(y, output)| {
            for (x, value) in output.iter_mut().enumerate() {
                *value = self.filtered[x * height + y];
            }
        });
    }
}

fn filter_centered_5_replicate(input: &[u8], output: &mut [u8], extreme: Extreme) {
    if input.len() < 5 {
        for (x, value) in output.iter_mut().enumerate() {
            let mut result = input[x.saturating_sub(2)];
            for offset in 1..5 {
                let source = (x + offset).saturating_sub(2).min(input.len() - 1);
                result = extreme_u8(result, input[source], extreme);
            }
            *value = result;
        }
        return;
    }
    output[0] = extreme_u8(extreme_u8(input[0], input[1], extreme), input[2], extreme);
    output[1] = extreme_u8(
        extreme_u8(extreme_u8(input[0], input[1], extreme), input[2], extreme),
        input[3],
        extreme,
    );
    for x in 2..input.len() - 2 {
        let left = extreme_u8(input[x - 2], input[x - 1], extreme);
        let middle = extreme_u8(input[x], input[x + 1], extreme);
        output[x] = extreme_u8(extreme_u8(left, middle, extreme), input[x + 2], extreme);
    }
    let last = input.len() - 1;
    output[last - 1] = extreme_u8(
        extreme_u8(extreme_u8(input[last - 3], input[last - 2], extreme), input[last - 1], extreme),
        input[last],
        extreme,
    );
    output[last] =
        extreme_u8(extreme_u8(input[last - 2], input[last - 1], extreme), input[last], extreme);
}

fn filter_vertical_centered_5_replicate(
    input: &[u8],
    width: usize,
    height: usize,
    y: usize,
    output: &mut [u8],
    extreme: Extreme,
) {
    let y0 = y.saturating_sub(2);
    let y1 = y.saturating_sub(1);
    let y3 = (y + 1).min(height - 1);
    let y4 = (y + 2).min(height - 1);
    let row0 = &input[y0 * width..(y0 + 1) * width];
    let row1 = &input[y1 * width..(y1 + 1) * width];
    let row2 = &input[y * width..(y + 1) * width];
    let row3 = &input[y3 * width..(y3 + 1) * width];
    let row4 = &input[y4 * width..(y4 + 1) * width];
    for x in 0..width {
        let upper = extreme_u8(row0[x], row1[x], extreme);
        let lower = extreme_u8(row3[x], row4[x], extreme);
        output[x] = extreme_u8(extreme_u8(upper, row2[x], extreme), lower, extreme);
    }
}

#[derive(Debug, Default)]
struct LineBuffers {
    padded: Vec<u8>,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn filter_line(
    input: &[u8],
    output: &mut [u8],
    kernel: usize,
    anchor: usize,
    border: BorderMode<u8, 1>,
    extreme: Extreme,
    padded: &mut Vec<u8>,
    prefix: &mut Vec<u8>,
    suffix: &mut Vec<u8>,
) {
    if input.is_empty() {
        return;
    }
    let padded_len = input.len() + kernel - 1;
    padded.resize(padded_len, 0);
    prefix.resize(padded_len, 0);
    suffix.resize(padded_len, 0);
    let constant = match border {
        BorderMode::Constant([value]) => value,
        _ => 0,
    };
    for (index, value) in padded.iter_mut().enumerate() {
        let source = index as isize - anchor as isize;
        *value = map_index(source, input.len(), border).map_or(constant, |mapped| input[mapped]);
    }

    for block_start in (0..padded_len).step_by(kernel) {
        let block_end = (block_start + kernel).min(padded_len);
        prefix[block_start] = padded[block_start];
        for index in block_start + 1..block_end {
            prefix[index] = extreme_u8(prefix[index - 1], padded[index], extreme);
        }
        suffix[block_end - 1] = padded[block_end - 1];
        for index in (block_start..block_end - 1).rev() {
            suffix[index] = extreme_u8(suffix[index + 1], padded[index], extreme);
        }
    }
    for (index, value) in output.iter_mut().enumerate() {
        *value = extreme_u8(suffix[index], prefix[index + kernel - 1], extreme);
    }
}

#[inline(always)]
fn extreme_u8(left: u8, right: u8, extreme: Extreme) -> u8 {
    match extreme {
        Extreme::Minimum => left.min(right),
        Extreme::Maximum => left.max(right),
    }
}

fn transpose_blocked(input: &[u8], output: &mut [u8], width: usize, height: usize) {
    const BLOCK: usize = 32;
    for y0 in (0..height).step_by(BLOCK) {
        for x0 in (0..width).step_by(BLOCK) {
            let y1 = (y0 + BLOCK).min(height);
            let x1 = (x0 + BLOCK).min(width);
            for y in y0..y1 {
                for x in x0..x1 {
                    output[x * height + y] = input[y * width + x];
                }
            }
        }
    }
}

fn repeat_extreme<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    iterations: usize,
    border: BorderMode<T, CHANNELS>,
    extreme: Extreme,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = pack(input)?;
    for _ in 0..iterations {
        output = extreme_once(output.view(), element, border, extreme)?;
    }
    Ok(output)
}

fn extreme_once<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
    element: &StructuringElement,
    border: BorderMode<T, CHANNELS>,
    extreme: Extreme,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            let mut values = [0.0; CHANNELS];
            let mut initialized = false;
            for ey in 0..element.height {
                for ex in 0..element.width {
                    if !element.mask[ey * element.width + ex] {
                        continue;
                    }
                    let pixel = fetch(
                        input,
                        x as isize + ex as isize - element.anchor_x as isize,
                        y as isize + ey as isize - element.anchor_y as isize,
                        border,
                    );
                    if !initialized {
                        values = pixel.map(PixelComponent::to_f64);
                        initialized = true;
                    } else {
                        for channel in 0..CHANNELS {
                            let candidate = pixel[channel].to_f64();
                            let ordering = candidate.total_cmp(&values[channel]);
                            if matches!(extreme, Extreme::Minimum) && ordering.is_lt()
                                || matches!(extreme, Extreme::Maximum) && ordering.is_gt()
                            {
                                values[channel] = candidate;
                            }
                        }
                    }
                }
            }
            output.extend(values.map(T::from_f64));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn subtract<T: PixelComponent, const CHANNELS: usize>(
    left: ImageView<'_, T, CHANNELS>,
    right: ImageView<'_, T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    if left.width() != right.width() || left.height() != right.height() {
        return Err(VisionError::ShapeMismatch(
            "morphology subtraction dimensions must match".into(),
        ));
    }
    let mut output = Vec::with_capacity(left.width() * left.height() * CHANNELS);
    for y in 0..left.height() {
        for x in 0..left.width() {
            let a = left.get(x, y).expect("left coordinate in bounds");
            let b = right.get(x, y).expect("right coordinate in bounds");
            output.extend(std::array::from_fn::<_, CHANNELS, _>(|channel| {
                T::from_f64(a[channel].to_f64() - b[channel].to_f64())
            }));
        }
    }
    Ok(Image::try_new_with_metadata(left.width(), left.height(), output, left.metadata())?)
}

fn pack<T: PixelComponent, const CHANNELS: usize>(
    input: ImageView<'_, T, CHANNELS>,
) -> VisionResult<Image<T, CHANNELS>> {
    let mut output = Vec::with_capacity(input.width() * input.height() * CHANNELS);
    for y in 0..input.height() {
        for x in 0..input.width() {
            output.extend_from_slice(input.get(x, y).expect("input coordinate in bounds"));
        }
    }
    Ok(Image::try_new_with_metadata(input.width(), input.height(), output, input.metadata())?)
}

fn validate_dimensions(
    width: usize,
    height: usize,
    anchor_x: usize,
    anchor_y: usize,
) -> VisionResult<()> {
    if width == 0 || height == 0 {
        return Err(VisionError::InvalidParameter(
            "structuring element dimensions must be non-zero".into(),
        ));
    }
    if anchor_x >= width || anchor_y >= height {
        return Err(VisionError::InvalidParameter(format!(
            "structuring element anchor ({anchor_x}, {anchor_y}) is outside {width}x{height}"
        )));
    }
    width
        .checked_mul(height)
        .ok_or_else(|| VisionError::InvalidParameter("structuring element overflows".into()))?;
    Ok(())
}

fn fill_ellipse(mask: &mut [bool], width: usize, height: usize) {
    let center_x = width / 2;
    let center_y = height / 2;
    let radius_y = center_y.max(1) as f64;
    for y in 0..height {
        let dy = y.abs_diff(center_y) as f64;
        if dy > radius_y {
            continue;
        }
        let extent =
            (center_x as f64 * (1.0 - dy * dy / (radius_y * radius_y)).sqrt()).round() as usize;
        let start = center_x.saturating_sub(extent);
        let end = (center_x + extent).min(width - 1);
        for x in start..=end {
            mask[y * width + x] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dilate, dilate_rect_u8, erode, erode_rect_u8, morphology_ex, morphology_rect_u8,
        morphology_rect_u8_into, MorphologyOperation, MorphologyShape, RectMorphologyWorkspace,
        StructuringElement,
    };
    use crate::BorderMode;
    use spatialrust_image::{Image, ImageRegion};

    #[test]
    fn built_in_masks_have_expected_3x3_layouts() {
        let rect = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let cross = StructuringElement::try_new(MorphologyShape::Cross, 3, 3).unwrap();
        let ellipse = StructuringElement::try_new(MorphologyShape::Ellipse, 3, 3).unwrap();
        let diamond = StructuringElement::try_new(MorphologyShape::Diamond, 3, 3).unwrap();
        assert_eq!(rect.mask().iter().filter(|&&v| v).count(), 9);
        assert_eq!(cross.mask().iter().filter(|&&v| v).count(), 5);
        assert_eq!(ellipse.mask().iter().filter(|&&v| v).count(), 5);
        assert_eq!(diamond.mask().iter().filter(|&&v| v).count(), 5);
    }

    #[test]
    fn erode_and_dilate_match_known_extrema_on_roi() {
        let parent = Image::<u8, 1>::try_new(5, 3, (0..15).collect()).unwrap();
        let roi = parent.view().subview(ImageRegion::new(1, 0, 3, 3)).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let low = erode(roi, &element, 1, BorderMode::Replicate).unwrap();
        let high = dilate(roi, &element, 1, BorderMode::Replicate).unwrap();
        assert_eq!(low[(1, 1)][0], 1);
        assert_eq!(high[(1, 1)][0], 13);
    }

    #[test]
    fn opening_removes_isolated_impulse() {
        let image = Image::<u16, 1>::try_new(3, 3, vec![0, 0, 0, 0, 100, 0, 0, 0, 0]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let output = morphology_ex(
            image.view(),
            MorphologyOperation::Open,
            &element,
            1,
            BorderMode::Constant([0]),
        )
        .unwrap();
        assert!(output.as_slice().iter().all(|&value| value == 0));
    }

    #[test]
    fn gradient_is_dilate_minus_erode_for_float() {
        let image = Image::<f32, 1>::try_new(3, 1, vec![1.0, 4.0, 9.0]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 1).unwrap();
        let output = morphology_ex(
            image.view(),
            MorphologyOperation::Gradient,
            &element,
            1,
            BorderMode::Replicate,
        )
        .unwrap();
        assert_eq!(output.as_slice(), &[3.0, 8.0, 5.0]);
    }

    #[test]
    fn zero_iterations_return_packed_copy() {
        let image = Image::<u8, 3>::from_pixel(2, 2, [1, 2, 3]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        assert_eq!(erode(image.view(), &element, 0, BorderMode::Replicate).unwrap(), image);
    }

    #[test]
    fn invalid_masks_are_rejected() {
        assert!(StructuringElement::try_from_mask(2, 2, 0, 0, vec![false; 4]).is_err());
        assert!(StructuringElement::try_from_mask(2, 2, 0, 0, vec![true; 3]).is_err());
    }

    #[test]
    fn rectangular_u8_fast_path_matches_generic_for_anchors_borders_and_iterations() {
        let mut state = 0x9e37_79b9_u32;
        let pixels = (0..63)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        let image = Image::<u8, 1>::try_new(9, 7, pixels).unwrap();
        let elements = [
            StructuringElement::try_new_with_anchor(MorphologyShape::Rect, 1, 1, 0, 0).unwrap(),
            StructuringElement::try_new_with_anchor(MorphologyShape::Rect, 4, 2, 0, 1).unwrap(),
            StructuringElement::try_new_with_anchor(MorphologyShape::Rect, 5, 3, 4, 0).unwrap(),
            StructuringElement::try_new_with_anchor(MorphologyShape::Rect, 11, 9, 5, 4).unwrap(),
        ];
        let borders = [
            BorderMode::Constant([37]),
            BorderMode::Replicate,
            BorderMode::Reflect,
            BorderMode::Reflect101,
            BorderMode::Wrap,
        ];

        for element in &elements {
            for &border in &borders {
                for iterations in 0..=2 {
                    assert_eq!(
                        erode_rect_u8(image.view(), element, iterations, border).unwrap(),
                        erode(image.view(), element, iterations, border).unwrap()
                    );
                    assert_eq!(
                        dilate_rect_u8(image.view(), element, iterations, border).unwrap(),
                        dilate(image.view(), element, iterations, border).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn rectangular_u8_composites_match_generic() {
        let image = Image::<u8, 1>::try_new(
            7,
            5,
            (0..35).map(|index| ((index * 47 + index * index * 3) % 256) as u8).collect(),
        )
        .unwrap();
        let element =
            StructuringElement::try_new_with_anchor(MorphologyShape::Rect, 4, 3, 1, 2).unwrap();
        for operation in [
            MorphologyOperation::Open,
            MorphologyOperation::Close,
            MorphologyOperation::Gradient,
            MorphologyOperation::TopHat,
            MorphologyOperation::BlackHat,
        ] {
            for iterations in 0..=2 {
                assert_eq!(
                    morphology_rect_u8(
                        image.view(),
                        operation,
                        &element,
                        iterations,
                        BorderMode::Replicate,
                    )
                    .unwrap(),
                    morphology_ex(
                        image.view(),
                        operation,
                        &element,
                        iterations,
                        BorderMode::Replicate,
                    )
                    .unwrap()
                );
            }
        }
    }

    #[test]
    fn rectangular_fast_path_rejects_sparse_masks() {
        let image = Image::<u8, 1>::from_pixel(3, 3, [1]).unwrap();
        let cross = StructuringElement::try_new(MorphologyShape::Cross, 3, 3).unwrap();
        assert!(erode_rect_u8(image.view(), &cross, 1, BorderMode::Replicate).is_err());
    }

    #[test]
    fn rectangular_into_matches_allocating_path_for_strided_input() {
        let parent = Image::<u8, 1>::try_new(
            13,
            9,
            (0..117).map(|index| ((index * 61 + 17) & 255) as u8).collect(),
        )
        .unwrap();
        let input = parent.view().subview(ImageRegion::new(2, 1, 9, 7)).unwrap();
        let element =
            StructuringElement::try_new_with_anchor(MorphologyShape::Rect, 4, 6, 1, 4).unwrap();
        let mut workspace = RectMorphologyWorkspace::new();
        let mut output = vec![0; input.width() * input.height()];

        for operation in [
            MorphologyOperation::Open,
            MorphologyOperation::Close,
            MorphologyOperation::Gradient,
            MorphologyOperation::TopHat,
            MorphologyOperation::BlackHat,
        ] {
            morphology_rect_u8_into(
                input,
                operation,
                &element,
                2,
                BorderMode::Reflect101,
                &mut output,
                &mut workspace,
            )
            .unwrap();
            assert_eq!(
                output,
                morphology_rect_u8(input, operation, &element, 2, BorderMode::Reflect101,)
                    .unwrap()
                    .into_vec()
            );
        }
    }

    #[test]
    fn rectangular_workspace_reuses_full_image_and_worker_capacity() {
        let image = Image::<u8, 1>::try_new(
            1000,
            1000,
            (0..1_000_000).map(|index| ((index * 29 + index / 1000) & 255) as u8).collect(),
        )
        .unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 5, 5).unwrap();
        let mut workspace = RectMorphologyWorkspace::new();
        let mut output = vec![0; 1_000_000];
        morphology_rect_u8_into(
            image.view(),
            MorphologyOperation::Open,
            &element,
            1,
            BorderMode::Replicate,
            &mut output,
            &mut workspace,
        )
        .unwrap();
        let capacity = workspace.capacity();
        let workers = workspace.worker_capacity();
        let line_capacity = workspace.line_capacity();
        assert!(capacity >= output.len());
        assert!(workers >= 1);
        assert!(line_capacity >= 1004);

        morphology_rect_u8_into(
            image.view(),
            MorphologyOperation::Close,
            &element,
            1,
            BorderMode::Replicate,
            &mut output,
            &mut workspace,
        )
        .unwrap();
        assert_eq!(workspace.capacity(), capacity);
        assert_eq!(workspace.worker_capacity(), workers);
        assert_eq!(workspace.line_capacity(), line_capacity);
    }

    #[test]
    fn rectangular_into_validates_output_length() {
        let image = Image::<u8, 1>::from_pixel(4, 3, [9]).unwrap();
        let element = StructuringElement::try_new(MorphologyShape::Rect, 3, 3).unwrap();
        let mut workspace = RectMorphologyWorkspace::new();
        assert!(morphology_rect_u8_into(
            image.view(),
            MorphologyOperation::Open,
            &element,
            1,
            BorderMode::Replicate,
            &mut [0; 11],
            &mut workspace,
        )
        .is_err());
    }
}
